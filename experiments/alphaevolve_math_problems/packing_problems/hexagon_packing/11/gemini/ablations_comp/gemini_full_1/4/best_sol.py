# EVOLVE-BLOCK-START
import numpy as np
import math
import random
import time
from shapely.geometry import Polygon
from scipy.optimize import differential_evolution
import numba

# --- Numba-accelerated Geometric & Objective Functions (from Inspiration 1) ---

@numba.njit(fastmath=True, cache=True)
def get_hex_vertices_numba(center_x, center_y, side_length, angle_degrees):
    """Generates hexagon vertices as a NumPy array for Numba-accelerated functions."""
    vertices = np.empty((6, 2), dtype=np.float64)
    start_angle_rad = np.deg2rad(angle_degrees)
    for i in range(6):
        angle_rad = start_angle_rad + i * np.pi / 3
        x = center_x + side_length * np.cos(angle_rad)
        y = center_y + side_length * np.sin(angle_rad)
        vertices[i, 0] = x
        vertices[i, 1] = y
    return vertices

@numba.njit(fastmath=True, cache=True)
def project_vertices_numba(vertices, axis):
    """Projects vertices onto an axis and returns the min and max projection."""
    min_proj = np.dot(vertices[0], axis)
    max_proj = min_proj
    for i in range(1, vertices.shape[0]):
        proj = np.dot(vertices[i], axis)
        if proj < min_proj:
            min_proj = proj
        elif proj > max_proj:
            max_proj = proj
    return min_proj, max_proj

@numba.njit(fastmath=True, cache=True)
def get_hexagon_penetration_numba(verts1, verts2):
    """Fast SAT check that returns penetration depth, or 0 if no overlap."""
    axes = np.empty((6, 2), dtype=np.float64)
    # Get 3 unique axes from each hexagon's edges (un-normalized)
    for i in range(3):
        p1 = verts1[i]; p2 = verts1[i + 1]
        edge = p2 - p1
        axes[i, 0] = -edge[1]
        axes[i, 1] = edge[0]

        p1 = verts2[i]; p2 = verts2[i + 1]
        edge = p2 - p1
        axes[i+3, 0] = -edge[1]
        axes[i+3, 1] = edge[0]
    
    min_penetration = np.inf

    for i in range(6):
        axis = axes[i]
        
        min1, max1 = project_vertices_numba(verts1, axis)
        min2, max2 = project_vertices_numba(verts2, axis)

        if max1 < min2 or max2 < min1:
            return 0.0

        overlap = min(max1, max2) - max(min1, min2)
        
        axis_len_sq = axis[0]**2 + axis[1]**2
        if axis_len_sq < 1e-12: continue

        penetration = overlap / np.sqrt(axis_len_sq)
        min_penetration = min(min_penetration, penetration)

    return min_penetration

@numba.njit(fastmath=True, cache=True)
def get_outer_hex_side_length_numba(all_vertices, outer_hex_orientation_degrees):
    """Numba-accelerated version to find the enclosing hexagon's side length."""
    if all_vertices.shape[0] == 0:
        return np.inf, 0.0, 0.0

    centroid_x = np.mean(all_vertices[:, 0])
    centroid_y = np.mean(all_vertices[:, 1])
    
    shifted_points = np.empty_like(all_vertices)
    shifted_points[:, 0] = all_vertices[:, 0] - centroid_x
    shifted_points[:, 1] = all_vertices[:, 1] - centroid_y

    angle_rad = np.deg2rad(-outer_hex_orientation_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    rotated_points = np.empty_like(shifted_points)
    rotated_points[:, 0] = shifted_points[:, 0] * cos_a - shifted_points[:, 1] * sin_a
    rotated_points[:, 1] = shifted_points[:, 0] * sin_a + shifted_points[:, 1] * cos_a

    max_proj_y = np.max(np.abs(rotated_points[:, 1]))

    cos_30, sin_30 = np.cos(np.pi/6), np.sin(np.pi/6)
    max_proj_30 = np.max(np.abs(rotated_points[:, 0] * cos_30 + rotated_points[:, 1] * sin_30))
    
    cos_150, sin_150 = np.cos(5*np.pi/6), np.sin(5*np.pi/6)
    max_proj_150 = np.max(np.abs(rotated_points[:, 0] * cos_150 + rotated_points[:, 1] * sin_150))
    
    min_apothem = max(max_proj_y, max_proj_30, max_proj_150)
    return min_apothem * 2.0 / np.sqrt(3.0), centroid_x, centroid_y

ANGLES_TO_CHECK_DE = np.linspace(0, 30, 15)

@numba.njit(fastmath=True, cache=True)
def objective_numba_core(individual, HEX_SIDE_LENGTH, N_HEXAGONS, angles_to_check):
    """The core JIT-compiled objective function with a smooth, graded penalty for overlaps."""
    centers = np.empty((N_HEXAGONS, 2), dtype=np.float64)
    angles = np.empty(N_HEXAGONS, dtype=np.float64)
    for i in range(N_HEXAGONS):
        centers[i, 0] = individual[i*3]
        centers[i, 1] = individual[i*3 + 1]
        angles[i] = individual[i*3 + 2]

    all_inner_verts = np.empty((N_HEXAGONS, 6, 2), dtype=np.float64)
    for i in range(N_HEXAGONS):
        all_inner_verts[i] = get_hex_vertices_numba(centers[i, 0], centers[i, 1], HEX_SIDE_LENGTH, angles[i])

    total_penalty = 0.0
    has_overlap = False
    DIAMETER_SQ = (2 * HEX_SIDE_LENGTH)**2
    for i in range(N_HEXAGONS):
        for j in range(i + 1, N_HEXAGONS):
            dist_sq = ((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            if dist_sq < DIAMETER_SQ:
                penetration = get_hexagon_penetration_numba(all_inner_verts[i], all_inner_verts[j])
                if penetration > 1e-9:
                    has_overlap = True
                    # The graded penalty is crucial: it's proportional to the square of the penetration.
                    total_penalty += penetration * penetration

    if has_overlap:
        return 100.0 + total_penalty * 1000.0 # Return a large value + graded penalty

    flat_verts = all_inner_verts.reshape(N_HEXAGONS * 6, 2)
    min_R_overall = np.inf
    for angle in angles_to_check:
        R, _, _ = get_outer_hex_side_length_numba(flat_verts, angle)
        if R < min_R_overall:
            min_R_overall = R
    return min_R_overall

def objective_DE(individual, HEX_SIDE_LENGTH, N_HEXAGONS):
    """Python wrapper for the Numba objective function."""
    return objective_numba_core(individual, HEX_SIDE_LENGTH, N_HEXAGONS, ANGLES_TO_CHECK_DE)

# --- Original Geometric Helper Functions (for final high-precision evaluation) ---

def create_hexagon_polygon(center_x, center_y, side_length, angle_degrees):
    """Creates a shapely Polygon object for a regular hexagon."""
    vertices = []
    start_angle_rad = math.radians(angle_degrees)
    for i in range(6):
        angle_rad = start_angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle_rad)
        y = center_y + side_length * math.sin(angle_rad)
        vertices.append((x, y))
    return Polygon(vertices)

def get_outer_hexagon_side_length_shapely(inner_hex_polygons, outer_hex_orientation_degrees=0):
    """Calculates the minimum side length of a regular hexagon that contains all `inner_hex_polygons`."""
    all_vertices = []
    for poly in inner_hex_polygons:
        all_vertices.extend(poly.exterior.coords[:-1]) 
    if not all_vertices: return float('inf'), 0, 0
    points = np.array(all_vertices)
    centroid_x = np.mean(points[:, 0]); centroid_y = np.mean(points[:, 1])
    shifted_points = points - np.array([centroid_x, centroid_y])
    angle_rad = math.radians(-outer_hex_orientation_degrees)
    rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad),  math.cos(angle_rad)]])
    rotated_points = shifted_points @ rotation_matrix.T
    max_proj_y = np.max(np.abs(rotated_points[:, 1])) 
    cos_30, sin_30 = np.cos(math.radians(30)), np.sin(math.radians(30))
    max_proj_30 = np.max(np.abs(rotated_points[:, 0] * cos_30 + rotated_points[:, 1] * sin_30))
    cos_150, sin_150 = np.cos(math.radians(150)), np.sin(math.radians(150))
    max_proj_150 = np.max(np.abs(rotated_points[:, 0] * cos_150 + rotated_points[:, 1] * sin_150))
    min_apothem = max(max_proj_y, max_proj_30, max_proj_150)
    outer_side_length = min_apothem * 2 / np.sqrt(3)
    return outer_side_length, centroid_x, centroid_y

# --- Initial Population Generator ---
def create_initial_population(popsize, bounds, N_HEXAGONS=11):
    """Creates a custom initial population for DE based on a strong initial guess."""
    _sqrt3 = np.sqrt(3)
    base_pos_xy = [
        (0.0, 0.0), (0.0, 2.0), (0.0, -2.0),
        (_sqrt3, 1.0), (_sqrt3, -1.0),
        (-_sqrt3, 1.0), (-_sqrt3, -1.0),
        (_sqrt3, 3.0), (-_sqrt3, 3.0),
        (2.0 * _sqrt3, 0.0), (-2.0 * _sqrt3, 0.0),
    ]
    base_individual = []
    for x, y in base_pos_xy:
        base_individual.extend([x, y, 0.0]) # Use 0 angle as base
    base_individual = np.array(base_individual)
    ndim = N_HEXAGONS * 3
    initial_pop_array = np.zeros((popsize, ndim))
    for i in range(popsize):
        individual = np.copy(base_individual)
        for j in range(ndim):
            param_min, param_max = bounds[j]
            if j % 3 < 2: # x or y coordinate
                perturbation = random.uniform(-0.5, 0.5)
                individual[j] += perturbation
                individual[j] = np.clip(individual[j], param_min, param_max)
            else: # angle
                perturbation = random.uniform(0, 360)
                individual[j] += perturbation
                individual[j] %= 360.0
        initial_pop_array[i] = individual
    return initial_pop_array

def hexagon_packing_11():
    """
    Uses Differential Evolution with a Numba-accelerated objective function to find an optimal arrangement.
    This approach is heavily inspired by the top-performing inspiration programs.
    """
    random.seed(42)
    np.random.seed(42)

    HEX_SIDE_LENGTH = 1.0
    N_HEXAGONS = 11
    POS_MIN, POS_MAX = -5.0, 5.0
    ANGLE_MIN, ANGLE_MAX = 0.0, 360.0

    bounds = []
    for _ in range(N_HEXAGONS):
        bounds.extend([(POS_MIN, POS_MAX), (POS_MIN, POS_MAX), (ANGLE_MIN, ANGLE_MAX)])
    
    # Increased budget for a more thorough search, leveraging the fast Numba objective function.
    popsize = 70
    maxiter = 16000 # Drastically increased from inspiration's 6000 to use more of the time budget.

    initial_population = create_initial_population(popsize, bounds, N_HEXAGONS)

    result = differential_evolution(
        objective_DE,
        bounds,
        args=(HEX_SIDE_LENGTH, N_HEXAGONS),
        init=initial_population,
        maxiter=maxiter,
        popsize=popsize,
        recombination=0.8,
        tol=1e-7, # Tighten tolerance
        polish=True,
        workers=-1,
        updating='deferred',
        strategy='best1bin',
        seed=42,
        disp=False
    )
    
    best_individual = result.x
    
    # Use original, high-precision shapely functions for final evaluation and centering.
    inner_hex_polygons_final = []
    for i in range(N_HEXAGONS):
        x, y, theta = best_individual[i*3], best_individual[i*3 + 1], best_individual[i*3 + 2]
        inner_hex_polygons_final.append(create_hexagon_polygon(x, y, HEX_SIDE_LENGTH, theta))
    
    # Use a very fine step for the final angle check for maximum precision.
    angles_to_check_fine = np.arange(0, 30.1, 0.05) 
    min_R_overall = float('inf')
    best_cx, best_cy = 0, 0
    best_outer_angle = 0
    
    for angle in angles_to_check_fine:
        R, cx, cy = get_outer_hexagon_side_length_shapely(inner_hex_polygons_final, angle)
        if R < min_R_overall:
            min_R_overall = R
            best_cx, best_cy = cx, cy
            best_outer_angle = angle

    final_outer_hex_side_length = min_R_overall

    final_inner_hex_data = []
    for i in range(N_HEXAGONS):
        x, y, theta = best_individual[i*3], best_individual[i*3 + 1], best_individual[i*3 + 2]
        final_inner_hex_data.append([x - best_cx, y - best_cy, theta])

    final_inner_hex_data = np.array(final_inner_hex_data)
    final_outer_hex_data = np.array([0, 0, best_outer_angle])

    return final_inner_hex_data, final_outer_hex_data, final_outer_hex_side_length


# EVOLVE-BLOCK-END
