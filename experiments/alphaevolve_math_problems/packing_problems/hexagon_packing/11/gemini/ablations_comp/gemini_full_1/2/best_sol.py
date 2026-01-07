# EVOLVE-BLOCK-START
import numpy as np
import math # Used for np.deg2rad in Numba functions
import random
import time
from scipy.optimize import differential_evolution
import numba # For JIT compilation of geometric functions

# Constants
UNIT_HEX_SIDE_LENGTH = 1.0
SQRT3 = np.sqrt(3.0) # Numba compatible float
HUGE_PENALTY = 1e10  # Penalty for overlaps or invalid configurations

# --- Numba-accelerated Geometric & Objective Functions ---

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
            return 0.0 # Found a separating axis, no overlap

        overlap = min(max1, max2) - max(min1, min2)
        
        axis_len_sq = axis[0]**2 + axis[1]**2
        if axis_len_sq < 1e-12: continue # Avoid division by zero for tiny axes

        penetration = overlap / np.sqrt(axis_len_sq)
        min_penetration = min(min_penetration, penetration)

    return min_penetration

@numba.njit(fastmath=True, cache=True)
def get_min_outer_hex_side_for_point_numba(px, py):
    """
    Calculates the minimum side length `R` of a regular hexagon,
    centered at (0,0) with 0 rotation (flat top/bottom),
    that contains the point (px, py).
    """
    r1 = 2 * np.abs(py) / SQRT3
    r2 = np.abs(px + py / SQRT3)
    r3 = np.abs(px - py / SQRT3)
    return max(r1, r2, r3)

@numba.njit(fastmath=True, cache=True)
def calculate_outer_R_from_params_numba(all_inner_verts_flat, outer_cx, outer_cy, outer_angle_deg):
    """
    Calculates the minimum outer hexagon side length required to contain
    all inner hexagon vertices, given the outer hexagon's explicit center and rotation.
    """
    min_R_outer = 0.0
    
    angle_rad = -np.deg2rad(outer_angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    for i in range(all_inner_verts_flat.shape[0]):
        vx, vy = all_inner_verts_flat[i, 0], all_inner_verts_flat[i, 1]
        
        tx = vx - outer_cx
        ty = vy - outer_cy

        rotated_tx = tx * cos_a - ty * sin_a
        rotated_ty = tx * sin_a + ty * cos_a
        
        required_R_for_point = get_min_outer_hex_side_for_point_numba(rotated_tx, rotated_ty)
        min_R_outer = max(min_R_outer, required_R_for_point)
    return min_R_outer

@numba.njit(fastmath=True, cache=True)
def objective_numba_core(individual, HEX_SIDE_LENGTH, N_HEXAGONS):
    """
    The core JIT-compiled objective function with high-fidelity SAT-based penalty.
    """
    centers = np.empty((N_HEXAGONS, 2), dtype=np.float64)
    angles = np.empty(N_HEXAGONS, dtype=np.float64)
    for i in range(N_HEXAGONS):
        centers[i, 0] = individual[i*3]
        centers[i, 1] = individual[i*3 + 1]
        angles[i] = individual[i*3 + 2]

    outer_cx = individual[N_HEXAGONS * 3]
    outer_cy = individual[N_HEXAGONS * 3 + 1]
    outer_angle_deg = individual[N_HEXAGONS * 3 + 2]

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
                    total_penalty += penetration * penetration

    if has_overlap:
        return 100.0 + total_penalty * 1000.0

    flat_verts = all_inner_verts.reshape(N_HEXAGONS * 6, 2)
    min_R_overall = calculate_outer_R_from_params_numba(flat_verts, outer_cx, outer_cy, outer_angle_deg)
    
    return min_R_overall

def objective_DE_numba(individual, HEX_SIDE_LENGTH, N_HEXAGONS):
    """
    Python wrapper for the Numba objective function, passed to differential_evolution.
    """
    return objective_numba_core(individual, HEX_SIDE_LENGTH, N_HEXAGONS)

def create_initial_population(popsize, bounds_inner_hex, N_HEXAGONS=11):
    """
    Creates a custom initial population for DE, with all individuals being
    perturbations of a strong initial guess.
    """
    _sqrt3 = np.sqrt(3.0) 
    base_pos_xy = [
        (0.0, 0.0), (0.0, 2.0), (0.0, -2.0),
        (_sqrt3, 1.0), (_sqrt3, -1.0),
        (-_sqrt3, 1.0), (-_sqrt3, -1.0),
        (_sqrt3, 3.0), (-_sqrt3, 3.0),
        (2.0 * _sqrt3, 0.0), (-2.0 * _sqrt3, 0.0),
    ]
    
    base_individual = []
    for x, y in base_pos_xy:
        base_individual.extend([x, y, 0.0])
    base_individual = np.array(base_individual)

    ndim_inner = N_HEXAGONS * 3
    initial_pop_array = np.zeros((popsize, ndim_inner))

    for i in range(popsize):
        individual = np.copy(base_individual)
        for j in range(ndim_inner):
            param_min, param_max = bounds_inner_hex[j]
            
            if j % 3 < 2:
                perturbation = random.uniform(-0.75, 0.75) 
                individual[j] += perturbation
                individual[j] = np.clip(individual[j], param_min, param_max)
            else:
                perturbation = random.uniform(0, 360)
                individual[j] += perturbation
                individual[j] %= 360.0
        initial_pop_array[i] = individual
        
    return initial_pop_array

def hexagon_packing_11():
    """
    Uses Numba-accelerated Differential Evolution to find an optimal arrangement.
    """
    random.seed(42)
    np.random.seed(42)

    N_HEXAGONS = 11

    POS_MIN, POS_MAX = -5.0, 5.0
    ANGLE_MIN, ANGLE_MAX = 0.0, 360.0

    bounds_inner_hex = []
    for _ in range(N_HEXAGONS):
        bounds_inner_hex.extend([(POS_MIN, POS_MAX), (POS_MIN, POS_MAX), (ANGLE_MIN, ANGLE_MAX)])
    
    outer_hex_center_x_bounds = (-1.0, 1.0)
    outer_hex_center_y_bounds = (-1.0, 1.0)
    outer_hex_angle_bounds = (0.0, 360.0)
    
    full_bounds = bounds_inner_hex + [outer_hex_center_x_bounds, outer_hex_center_y_bounds, outer_hex_angle_bounds]

    popsize = 150 
    maxiter = 27500

    initial_population_inner = create_initial_population(popsize, bounds_inner_hex, N_HEXAGONS)
    
    initial_population_full = np.empty((popsize, N_HEXAGONS * 3 + 3))
    for i in range(popsize):
        initial_population_full[i, :N_HEXAGONS * 3] = initial_population_inner[i]
        initial_population_full[i, N_HEXAGONS * 3:] = [0.0, 0.0, 0.0]

    result = differential_evolution(
        objective_DE_numba,
        full_bounds,
        args=(UNIT_HEX_SIDE_LENGTH, N_HEXAGONS),
        init=initial_population_full,
        maxiter=maxiter,
        popsize=popsize,
        recombination=0.9,
        tol=1e-6,
        polish=True,
        workers=-1,
        updating='deferred',
        strategy='randtobest1bin',
        mutation=(0.6, 1.2),
        seed=42,
        disp=False
    )
    
    best_individual = result.x
    final_outer_hex_side_length = result.fun 

    optimized_inner_hex_params = best_individual[:N_HEXAGONS * 3].reshape(N_HEXAGONS, 3)
    optimized_outer_hex_params = best_individual[N_HEXAGONS * 3:]
    best_outer_cx, best_outer_cy, best_outer_angle = optimized_outer_hex_params

    final_inner_hex_data = np.empty_like(optimized_inner_hex_params)
    for i in range(N_HEXAGONS):
        final_inner_hex_data[i, 0] = optimized_inner_hex_params[i, 0] - best_outer_cx
        final_inner_hex_data[i, 1] = optimized_inner_hex_params[i, 1] - best_outer_cy
        final_inner_hex_data[i, 2] = optimized_inner_hex_params[i, 2]

    final_outer_hex_data = np.array([0.0, 0.0, best_outer_angle])

    return final_inner_hex_data, final_outer_hex_data, final_outer_hex_side_length


# EVOLVE-BLOCK-END
