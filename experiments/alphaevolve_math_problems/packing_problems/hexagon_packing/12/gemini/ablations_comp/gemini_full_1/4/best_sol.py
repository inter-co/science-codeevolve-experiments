# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import time # Added for performance metrics
from numba import jit # Import numba for JIT compilation

# Constants for inner unit hexagons
HEX_SIDE_LENGTH = 1.0
SQRT3 = np.sqrt(3.0)

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
    # For two hexagons, we check 6 axes (3 from each hex's orientation).
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
    Creates a shapely Polygon for a regular hexagon.
    At angle_degrees=0, it's a pointy-top hexagon (one vertex on the positive x-axis).
    This function is now primarily used for final reconstruction and outer hex calculation,
    not for fast overlap checks within the objective function.
    """
    angle_rad_offset = np.deg2rad(angle_degrees)
    vertices = []
    for i in range(6):
        angle = np.deg2rad(i * 60) + angle_rad_offset
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return Polygon(vertices)


def calculate_min_enclosing_hex_side_length(hexagons_polygons, num_orientation_samples=30):
    """
    Calculates the minimum side length of a regular hexagon that contains all given inner hexagon polygons.
    It does this by sampling different rotations for the outer hexagon.
    This version is adapted from the efficient implementation in the inspiration programs.
    """
    if not hexagons_polygons:
        return np.inf, 0.0 # Return large value if no hexagons

    # Collect all vertices from all inner hexagons
    all_vertices = np.vstack([np.array(poly.exterior.coords[:-1]) for poly in hexagons_polygons])

    min_R = np.inf
    optimal_outer_rotation_rad = 0.0

    # Iterate through a range of possible outer hexagon rotations
    # Sampling within a 30-degree range (np.pi/6) is sufficient due to symmetry.
    for i in range(num_orientation_samples + 1):
        theta = i * (np.pi / 6) / num_orientation_samples
        
        # Rotate all vertices by -theta to effectively rotate the bounding box by +theta
        cos_m_theta, sin_m_theta = np.cos(-theta), np.sin(-theta)
        rot_matrix = np.array([[cos_m_theta, -sin_m_theta], [sin_m_theta, cos_m_theta]])
        rotated_vertices = all_vertices @ rot_matrix.T
        
        rx, ry = rotated_vertices[:, 0], rotated_vertices[:, 1]

        # Calculate bounding R for a pointy-top hexagon (with 0 rotation) using the inspiration's formula
        r_comp1 = np.abs(ry) * 2 / SQRT3
        r_comp2 = np.abs(rx + ry / SQRT3)
        r_comp3 = np.abs(rx - ry / SQRT3)
        
        max_R_for_this_theta = np.max(np.maximum.reduce([r_comp1, r_comp2, r_comp3]))
        
        if max_R_for_this_theta < min_R:
            min_R = max_R_for_this_theta
            optimal_outer_rotation_rad = theta

    return min_R, optimal_outer_rotation_rad

def objective_function(params):
    """
    Objective function for the optimizer. Calculates the outer hexagon side length for a given configuration.
    A large penalty is returned if inner hexagons overlap.
    This function now uses Numba-optimized SAT for fast overlap checks.
    """    
    r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg = params

    # Penalty for generator hexagons being too close to the origin
    if r1 < 0.1 or r2 < 0.1: 
        return 1e9

    inner_hex_definitions = [] # Stores (cx, cy, hex_angle) for each hexagon

    # Generate the 12 hexagons using 6-fold rotational symmetry for two "seed" hexagons
    # Seed 1:
    for j in range(6):
        current_center_angle_degrees = phi1_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        
        cx = r1 * np.cos(current_center_angle_rad)
        cy = r1 * np.sin(current_center_angle_rad)
        
        # Hexagon's absolute angle: its own rot_deg + the angle of its center from global x-axis
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
            
            if dist_centers_sq < MIN_OVERLAP_CENTER_DIST_SQ:
                return 1e9 # Definitely overlapping, apply penalty
            if dist_centers_sq >= MAX_NON_OVERLAP_CENTER_DIST_SQ:
                continue # Definitely not overlapping, skip SAT check

            # If in ambiguous range, use Numba-optimized SAT check
            if check_overlap_sat_numba(
                np.array([hex1_center_x, hex1_center_y]), hex1_angle,
                np.array([hex2_center_x, hex2_center_y]), hex2_angle,
                HEX_SIDE_LENGTH
            ):
                return 1e9 # Overlapping according to SAT, apply penalty

    # If no overlaps, proceed to calculate outer hexagon side length
    # Only now, create Shapely Polygon objects (this is the most expensive part)
    all_inner_hex_polygons = [
        create_hexagon_polygon(cx, cy, angle, HEX_SIDE_LENGTH) for cx, cy, angle in inner_hex_definitions
    ]
    outer_hex_side_length, _ = calculate_min_enclosing_hex_side_length(all_inner_hex_polygons, num_orientation_samples=30)
    return outer_hex_side_length


def hexagon_packing_12():
    """
    Algorithmically determines an optimal arrangement of 12 unit regular hexagons within a larger regular hexagon.
    This function uses an evolutionary optimization algorithm (differential_evolution) to search for the best
    arrangement that minimizes the side length of the enclosing hexagon.

    The search space is reduced by exploiting the C6 rotational symmetry of the problem. We define two "generator"
    hexagons in a 60-degree wedge of the plane and replicate them through rotation to form the full set of 12.
    The optimization variables include radial distance, angular position, and individual rotation for each generator.

    Returns:
        inner_hex_data (np.ndarray): Shape (12, 3) for (x, y, angle_degrees) of each inner hexagon.
        outer_hex_data (np.ndarray): Shape (3,) for (x, y, angle_degrees) of the outer hexagon.
        outer_hex_side_length (float): The side length of the minimal bounding outer hexagon.
    """
    start_time = time.time()

    # Bounds for the 6 variables: [r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg]
    # r: radial distance from origin. Min 0.5 (to avoid hexes too close to center), Max 4.5 (based on SOTA R=3.94).
    # phi_deg: angular position in the 0-60 degree sector (due to C6 symmetry).
    # rot_deg: rotation of the hexagon (relative to its radial line). Due to internal hex symmetry, 0-60 degrees is sufficient.
    bounds = [(0.5, 4.5), (0.0, 60.0), (0.0, 60.0), # Hexagon set 1 parameters
              (0.5, 4.5), (0.0, 60.0), (0.0, 60.0)] # Hexagon set 2 parameters

    # Run differential evolution for global optimization. (Aggressive parameters from Inspirations 1 & 3)
    result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=2000,        # Increased significantly for better exploration
        popsize=50,          # Increased for larger population diversity
        tol=1e-5,            # Tighter tolerance for convergence
        mutation=(0.5, 1.2), # Wider mutation range
        recombination=0.8,   # Higher crossover probability
        seed=42,             # For reproducibility
        polish=True,         # Refine the best solution found
        workers=-1,          # Utilize all available CPU cores for parallel execution
        init='latinhypercube', # Use Latin Hypercube Sampling for a more diverse initial population
        updating='deferred'  # Recommended for parallel execution
    )

    # --- Reconstruct Final Configuration ---
    # Recalculate with the optimal parameters to get detailed info, including outer hex rotation
    best_params = result.x
    
    r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg = best_params

    all_inner_hex_polygons = []
    inner_hex_data_list = []

    # Generate inner hexagon data for output based on optimized parameters
    for j in range(6):
        current_center_angle_degrees = phi1_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        
        cx = r1 * np.cos(current_center_angle_rad)
        cy = r1 * np.sin(current_center_angle_rad)
        hex_angle = (rot1_deg + current_center_angle_degrees) % 360
        inner_hex_data_list.append([cx, cy, hex_angle])
        all_inner_hex_polygons.append(create_hexagon_polygon(cx, cy, hex_angle))


    for j in range(6):
        current_center_angle_degrees = phi2_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        
        cx = r2 * np.cos(current_center_angle_rad)
        cy = r2 * np.sin(current_center_angle_rad)
        hex_angle = (rot2_deg + current_center_angle_degrees) % 360
        inner_hex_data_list.append([cx, cy, hex_angle])
        all_inner_hex_polygons.append(create_hexagon_polygon(cx, cy, hex_angle))

    inner_hex_data = np.array(inner_hex_data_list)

    # Calculate final outer hex side length and its optimal rotation with high sample count
    outer_hex_side_length, optimal_outer_rotation_rad = calculate_min_enclosing_hex_side_length(all_inner_hex_polygons, num_orientation_samples=360)
    
    # Outer hexagon is centered at origin, with the optimal rotation found
    outer_hex_data = np.array([0, 0, np.degrees(optimal_outer_rotation_rad)])
    
    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
    print(f"Optimal outer_hex_side_length: {outer_hex_side_length:.7f}")
    print(f"Optimal inv_outer_hex_side_length: {1/outer_hex_side_length:.7f}")
    print(f"Optimal parameters: {best_params}")

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
