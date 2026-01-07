# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import time # Added for performance metrics
from numba import jit # Import numba for JIT compilation
import os # New import from inspirations

# Constants for inner unit hexagons
HEX_SIDE_LENGTH = 1.0
SQRT3 = np.sqrt(3.0)

# Cap workers to avoid overwhelming system, inspired by Inspiration Programs
try:
    N_WORKERS = min(os.cpu_count(), 8) if os.cpu_count() is not None else -1
except (AttributeError, TypeError, NotImplementedError):
    N_WORKERS = -1

# Numba-optimized functions for Separating Axis Theorem (SAT) overlap check
@jit(nopython=True)
def get_hexagon_vertices_array_numba(center_x, center_y, angle_degrees, side_length=HEX_SIDE_LENGTH):
    """
    Returns the vertices of a regular hexagon as a (6, 2) numpy array, optimized for Numba.
    At angle_degrees=0, it's a pointy-top hexagon (one vertex on the positive x-axis).
    """
    angle_rad = np.deg2rad(angle_degrees)
    vertices = np.empty((6, 2), dtype=np.float64)
    for i in range(6):
        # Start angle for pointy-top is 0 degrees (along x-axis)
        current_angle = np.deg2rad(i * 60) + angle_rad
        vertices[i, 0] = center_x + side_length * np.cos(current_angle)
        vertices[i, 1] = center_y + side_length * np.sin(current_angle)
    return vertices

@jit(nopython=True)
def get_hexagon_axes_numba(angle_degrees):
    """
    Returns the 3 unique edge normal vectors for a hexagon with given orientation, optimized for Numba.
    These are the axes to check for SAT.
    """
    axes = np.empty((3, 2), dtype=np.float64)
    for i in range(3):
        # Perpendicular to edge of pointy-top hex rotated by angle_degrees
        edge_angle_rad = np.deg2rad(angle_degrees + i * 60 + 90) 
        axes[i, 0] = np.cos(edge_angle_rad)
        axes[i, 1] = np.sin(edge_angle_rad)
    return axes

@jit(nopython=True)
def check_overlap_sat_numba(hex1_center, hex1_angle, hex2_center, hex2_angle, side_length=HEX_SIDE_LENGTH):
    """
    Checks for overlap between two unit regular hexagons using the Separating Axis Theorem (SAT).
    Returns True if overlapping, False otherwise. Optimized for Numba.
    """
    vertices1 = get_hexagon_vertices_array_numba(hex1_center[0], hex1_center[1], hex1_angle, side_length)
    vertices2 = get_hexagon_vertices_array_numba(hex2_center[0], hex2_center[1], hex2_angle, side_length)

    axes1 = get_hexagon_axes_numba(hex1_angle)
    axes2 = get_hexagon_axes_numba(hex2_angle)
    
    # Combine axes from both hexagons.
    for axis in np.concatenate((axes1, axes2)):
        min1 = np.inf
        max1 = -np.inf
        for k in range(6):
            proj = vertices1[k, 0] * axis[0] + vertices1[k, 1] * axis[1]
            min1 = min(min1, proj)
            max1 = max(max1, proj)

        min2 = np.inf
        max2 = -np.inf
        for k in range(6):
            proj = vertices2[k, 0] * axis[0] + vertices2[k, 1] * axis[1]
            min2 = min(min2, proj)
            max2 = max(max2, proj)

        if max1 < min2 or max2 < min1:
            return False # Separating axis found, no overlap

    return True # No separating axis found, polygons overlap


def create_hexagon_polygon(center_x, center_y, angle_degrees, side_length=HEX_SIDE_LENGTH):
    """
    Creates a shapely Polygon for a regular hexagon. Kept for potential debugging or visualization,
    but not used in the performance-critical optimization path.
    """
    angle_rad_offset = np.deg2rad(angle_degrees)
    vertices = []
    for i in range(6):
        angle = np.deg2rad(i * 60) + angle_rad_offset
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return Polygon(vertices)


@jit(nopython=True, fastmath=True)
def _calculate_min_enclosing_hex_side_length_numba(all_vertices, num_orientation_samples=30):
    """
    Numba-jitted function to calculate the side length of the minimum regular hexagon
    that encloses all given vertices by testing different outer hexagon rotations.
    Adopted from inspiration programs for maximum performance.
    """
    if all_vertices.shape[0] == 0:
        return np.inf, 0.0

    min_R = np.inf
    optimal_theta = 0.0
    sqrt3 = np.sqrt(3.0)

    for i in range(num_orientation_samples + 1):
        theta = i * (np.pi / 6) / num_orientation_samples
        cos_m_theta = np.cos(-theta)
        sin_m_theta = np.sin(-theta)
        
        max_R_for_this_theta = 0.0
        for k in range(all_vertices.shape[0]):
            vx, vy = all_vertices[k, 0], all_vertices[k, 1]
            
            rx = vx * cos_m_theta - vy * sin_m_theta
            ry = vx * sin_m_theta + vy * cos_m_theta

            r_comp1 = np.abs(ry) * 2.0 / sqrt3
            r_comp2 = np.abs(rx + ry / sqrt3)
            r_comp3 = np.abs(rx - ry / sqrt3)
            
            current_max = r_comp1
            if r_comp2 > current_max: current_max = r_comp2
            if r_comp3 > current_max: current_max = r_comp3
            
            if current_max > max_R_for_this_theta: max_R_for_this_theta = current_max
        
        if max_R_for_this_theta < min_R:
            min_R = max_R_for_this_theta
            optimal_theta = theta
            
    return min_R, optimal_theta

def objective_function(params):
    """
    Objective function for the optimizer. This version avoids creating Shapely Polygons
    and uses a fully numba-jitted enclosing hexagon calculation for max performance.
    """    
    r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg = params

    if r1 < 0.1 or r2 < 0.1: 
        return 1e9

    inner_hex_definitions = [] 
    # Seed 1:
    for j in range(6):
        current_center_angle_degrees = phi1_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        cx = r1 * np.cos(current_center_angle_rad)
        cy = r1 * np.sin(current_center_angle_rad)
        hex_angle = (rot1_deg + current_center_angle_degrees) % 360
        inner_hex_definitions.append((cx, cy, hex_angle))
    # Seed 2:
    for j in range(6):
        current_center_angle_degrees = phi2_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        cx = r2 * np.cos(current_center_angle_rad)
        cy = r2 * np.sin(current_center_angle_rad)
        hex_angle = (rot2_deg + current_center_angle_degrees) % 360
        inner_hex_definitions.append((cx, cy, hex_angle))

    # Overlap check using Numba-optimized SAT
    MIN_OVERLAP_CENTER_DIST_SQ = (SQRT3 * HEX_SIDE_LENGTH)**2 - 1e-9 
    MAX_NON_OVERLAP_CENTER_DIST_SQ = (2 * HEX_SIDE_LENGTH)**2 + 1e-9 
    
    for i in range(len(inner_hex_definitions)):
        for j in range(i + 1, len(inner_hex_definitions)):
            hex1_center_x, hex1_center_y, hex1_angle = inner_hex_definitions[i]
            hex2_center_x, hex2_center_y, hex2_angle = inner_hex_definitions[j]
            dist_centers_sq = (hex1_center_x - hex2_center_x)**2 + (hex1_center_y - hex2_center_y)**2
            if dist_centers_sq < MIN_OVERLAP_CENTER_DIST_SQ: return 1e9
            if dist_centers_sq >= MAX_NON_OVERLAP_CENTER_DIST_SQ: continue
            if check_overlap_sat_numba(np.array([hex1_center_x, hex1_center_y]), hex1_angle, np.array([hex2_center_x, hex2_center_y]), hex2_angle, HEX_SIDE_LENGTH):
                return 1e9

    # If no overlaps, calculate outer hex side length using numba-jitted function
    all_vertices_list = []
    for cx, cy, angle in inner_hex_definitions:
        all_vertices_list.append(get_hexagon_vertices_array_numba(cx, cy, angle, HEX_SIDE_LENGTH))
    all_vertices = np.vstack(all_vertices_list)

    outer_hex_side_length, _ = _calculate_min_enclosing_hex_side_length_numba(all_vertices, num_orientation_samples=30)
    return outer_hex_side_length


def hexagon_packing_12():
    """
    Algorithmically determines an optimal arrangement of 12 unit regular hexagons within a larger regular hexagon.
    This function uses an evolutionary optimization algorithm (differential_evolution) to search for the best
    arrangement that minimizes the side length of the enclosing hexagon.

    The search space is reduced by exploiting C6 rotational symmetry.
    """
    start_time = time.time()

    bounds = [(0.5, 4.5), (0.0, 60.0), (0.0, 60.0), # Hexagon set 1 parameters
              (0.5, 4.5), (0.0, 60.0), (0.0, 60.0)] # Hexagon set 2 parameters

    result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=2000,
        popsize=50,
        tol=1e-5,
        mutation=(0.5, 1.2),
        recombination=0.8,
        seed=42,
        polish=True,
        workers=N_WORKERS,          # Use capped workers for stability
        init='latinhypercube',
        updating='deferred'
    )

    # --- Reconstruct Final Configuration ---
    best_params = result.x
    r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg = best_params

    inner_hex_data_list = []
    all_vertices_list = []

    # Generate inner hexagon data and vertices from optimized parameters
    for j in range(6):
        current_center_angle_degrees = phi1_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        cx = r1 * np.cos(current_center_angle_rad)
        cy = r1 * np.sin(current_center_angle_rad)
        hex_angle = (rot1_deg + current_center_angle_degrees) % 360
        inner_hex_data_list.append([cx, cy, hex_angle])
        all_vertices_list.append(get_hexagon_vertices_array_numba(cx, cy, hex_angle))

    for j in range(6):
        current_center_angle_degrees = phi2_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        cx = r2 * np.cos(current_center_angle_rad)
        cy = r2 * np.sin(current_center_angle_rad)
        hex_angle = (rot2_deg + current_center_angle_degrees) % 360
        inner_hex_data_list.append([cx, cy, hex_angle])
        all_vertices_list.append(get_hexagon_vertices_array_numba(cx, cy, hex_angle))

    inner_hex_data = np.array(inner_hex_data_list)
    all_vertices_final = np.vstack(all_vertices_list)

    # Calculate final outer hex side length and its optimal rotation with high sample count
    outer_hex_side_length, optimal_outer_rotation_rad = _calculate_min_enclosing_hex_side_length_numba(all_vertices_final, num_orientation_samples=360)
    
    outer_hex_data = np.array([0, 0, np.degrees(optimal_outer_rotation_rad)])
    
    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
    print(f"Optimal outer_hex_side_length: {outer_hex_side_length:.7f}")
    print(f"Optimal inv_outer_hex_side_length: {1/outer_hex_side_length:.7f}")
    print(f"Optimal parameters: {best_params}")

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
