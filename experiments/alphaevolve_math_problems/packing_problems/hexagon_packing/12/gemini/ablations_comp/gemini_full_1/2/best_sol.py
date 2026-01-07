# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import differential_evolution
import time
from numba import jit # Import numba for JIT compilation

# --- Helper Functions for Hexagon Geometry and Enclosing Hexagon Calculation ---

# Unit regular hexagon side length is 1.0 throughout this problem.
HEX_SIDE_LENGTH = 1.0 # Define as constant
SQRT3 = np.sqrt(3.0) # Define as constant

# Numba-optimized functions for Separating Axis Theorem (SAT) overlap check (from Inspiration 1/3)
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

    # Manually construct all_axes to avoid np.concatenate inside nopython jit
    all_axes = np.empty((6, 2), dtype=np.float64)
    axes1 = get_hexagon_axes_numba(hex1_angle)
    axes2 = get_hexagon_axes_numba(hex2_angle)
    all_axes[:3] = axes1
    all_axes[3:] = axes2
    
    for axis in all_axes:
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


def _get_hexagon_vertices(center_x, center_y, angle_degrees, side_length=HEX_SIDE_LENGTH):
    """
    Calculates the 6 vertices of a regular hexagon.
    
    Args:
        center_x (float): X-coordinate of the hexagon's center.
        center_y (float): Y-coordinate of the hexagon's center.
        angle_degrees (float): Rotation angle of the hexagon in degrees.
                                0 degrees typically means a flat top/bottom.
        side_length (float): The side length of the hexagon.
        
    Returns:
        np.ndarray: A (6, 2) array of vertex coordinates.
    """
    angle_rad = np.radians(angle_degrees)
    vertices = []
    for i in range(6):
        # Vertices are at distance 'side_length' from the center
        # at angles offset by multiples of 60 degrees (pi/3 radians)
        angle = angle_rad + i * np.pi / 3
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def _create_hexagon_polygon(center_x, center_y, angle_degrees, side_length=HEX_SIDE_LENGTH):
    """
    Creates a shapely Polygon object for a regular hexagon.
    
    Args:
        center_x (float): X-coordinate of the hexagon's center.
        center_y (float): Y-coordinate of the hexagon's center.
        angle_degrees (float): Rotation angle of the hexagon in degrees.
        side_length (float): The side length of the hexagon.
        
    Returns:
        shapely.geometry.Polygon: The hexagon polygon.
    """
    vertices = _get_hexagon_vertices(center_x, center_y, angle_degrees, side_length)
    return Polygon(vertices)

def _calculate_min_enclosing_hex_side_length(hexagons_polygons, num_orientation_samples=60):
    """
    Calculates the side length of the minimum regular hexagon that encloses
    all given shapely hexagon polygons, and its optimal rotation angle.
    
    This function finds the minimum enclosing regular hexagon by iterating through
    a range of possible orientations for the outer hexagon and calculating
    the required side length for each orientation.
    
    Args:
        hexagons_polygons (list): A list of shapely Polygon objects representing the inner hexagons.
        num_orientation_samples (int): Number of samples to take between 0 and pi/6 radians
                                       to check for the optimal outer hexagon orientation.
        
    Returns:
        tuple: (float, float)
            - The side length of the smallest enclosing regular hexagon.
            - The rotation angle (in radians) of the outer hexagon that yields this minimum side length.
    """
    if not hexagons_polygons:
        return np.inf, 0.0 # Return infinity if no polygons are provided, indicating an invalid state
        
    all_vertices = []
    for poly in hexagons_polygons:
        # Exclude the closing point from shapely's exterior.coords for unique vertices
        all_vertices.extend(poly.exterior.coords[:-1]) 

    if not all_vertices:
        return 0.0, 0.0 # Should not happen with valid input

    all_vertices = np.array(all_vertices)
    
    min_R = np.inf # Initialize with a very large value
    optimal_outer_rotation_rad = 0.0 # Initialize optimal rotation angle

    # Iterate through a range of orientations for the outer hexagon.
    # Due to the 6-fold rotational symmetry of a regular hexagon,
    # we only need to check angles from 0 to pi/6 radians (0 to 30 degrees)
    # to find the absolute minimum enclosing hexagon.
    
    for i in range(num_orientation_samples + 1):
        # `theta` represents the rotation of the outer hexagon's "flat sides" from the x-axis.
        theta = i * (np.pi / 6) / num_orientation_samples 
        
        # Rotation by -theta for all vertices, vectorized for speed
        cos_m_theta = np.cos(-theta)
        sin_m_theta = np.sin(-theta)
        
        # Rotation matrix for [x, y] row vectors
        rot_matrix = np.array([[cos_m_theta, sin_m_theta], [-sin_m_theta, cos_m_theta]])
        rotated_vertices = all_vertices @ rot_matrix
        
        # Extract rotated x and y coordinates
        rx = rotated_vertices[:, 0]
        ry = rotated_vertices[:, 1]

        # For a point (x,y) to be contained within a 0-rotated hexagon of side R (centered at origin):
        # 1. Its absolute y-coordinate must be less than or equal to R * sqrt(3)/2.
        #    This implies R >= abs(y) * 2 / sqrt(3).
        # 2. Its absolute "hexagonal" x-coordinates (projections onto axes at 60 deg to vertical)
        #    must be less than or equal to R.
        #    This implies R >= abs(x + y/sqrt(3)) and R >= abs(x - y/sqrt(3)).
        # The maximum of these three conditions for all points gives the minimum R
        # required for this specific orientation (`theta`).
        r_comp1 = np.abs(ry) * 2 / np.sqrt(3)
        r_comp2 = np.abs(rx + ry / np.sqrt(3))
        r_comp3 = np.abs(rx - ry / np.sqrt(3))
        
        # Update the maximum R required for any vertex for the current `theta`
        # np.maximum.reduce is efficient for finding the max across multiple arrays
        max_R_for_this_theta = np.max(np.maximum.reduce([r_comp1, r_comp2, r_comp3]))
        
        # Update the overall minimum R found across all `theta` orientations
        if max_R_for_this_theta < min_R:
            min_R = max_R_for_this_theta
            optimal_outer_rotation_rad = theta

    return min_R, optimal_outer_rotation_rad

# --- Objective Function for Optimization (Symmetry-Enforced) ---

def _objective_function_symmetric(params):
    """
    Objective function enforcing C6 symmetry. It optimizes for 2 generator
    hexagons, and the remaining 10 are created by 60-degree rotations.
    This reduces the search space from 36D to 6D.
    
    Args:
        params (np.ndarray): A 1D array of 6 elements representing the
                             (r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg)
                             for 2 generator hexagons.
                             
    Returns:
        float: The outer hexagon's side length, or np.inf if overlaps occur.
    """
    num_hexagons_total = 12
    
    # Unpack parameters: [r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg]
    r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg = params
    
    # Penalty for generator hexagons being too close to the origin,
    # as this can lead to degenerate or perfectly overlapping symmetric copies.
    if r1 < 0.1 or r2 < 0.1: 
        return 1e9 # Use a large value for penalty

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
    
    # Overlap check using Numba-optimized SAT (from Inspiration 1/3)
    # MIN_OVERLAP_CENTER_DIST_SQ: If dist_sq < this, hexagons definitely overlap (vertex-to-vertex contact)
    MIN_OVERLAP_CENTER_DIST_SQ = (SQRT3 * HEX_SIDE_LENGTH)**2 - 1e-9 
    # MAX_NON_OVERLAP_CENTER_DIST_SQ: If dist_sq > this, hexagons definitely do not overlap (edge-to-edge contact)
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

    # If no overlaps, proceed to calculate outer hexagon side length.
    # Only now, create Shapely Polygon objects (this is the most expensive part).
    all_inner_hex_polygons = [
        _create_hexagon_polygon(cx, cy, angle, HEX_SIDE_LENGTH) for cx, cy, angle in inner_hex_definitions
    ]
    outer_hex_side_length, _ = _calculate_min_enclosing_hex_side_length(all_inner_hex_polygons, num_orientation_samples=30)
    return outer_hex_side_length

# --- Main Optimization Function ---

def hexagon_packing_12():
    """
    Discovers an optimal arrangement of 12 unit regular hexagons within a larger regular hexagon,
    maximizing 1/outer_hex_side_length (minimizing outer_hex_side_length).
    
    This implementation uses `scipy.optimize.differential_evolution` for global optimization,
    exploiting C6 symmetry to reduce the search space.
    
    Returns:
        tuple:
            - inner_hex_data (np.ndarray): Shape (12,3), (x, y, angle_degrees) for each inner hexagon.
            - outer_hex_data (np.ndarray): Shape (3,), (x, y, angle_degrees) for the outer hexagon.
                                            (x,y) is (0,0) as it's centered, angle_degrees is the
                                            optimal rotation found.
            - outer_hex_side_length (float): The side length of the optimal outer hexagon.
    """
    num_generators = 2 # We optimize for 2 generator hexagons
    n = num_generators * 6 # Total 12 hexagons (2 generators * 6 rotations)
    
    # Define bounds for the 2 generator hexagons (6 parameters total).
    # Using (r, phi_deg, rot_deg) parameterization from inspirations.
    # r: radial distance from origin. Min 0.5 (to avoid hexes too close to center), Max 4.5 (based on SOTA R=3.94).
    # phi_deg: angular position in the 0-60 degree sector (due to C6 symmetry).
    # rot_deg: rotation of the hexagon (relative to its radial line). Due to internal hex symmetry, 0-60 degrees is sufficient.
    bounds = [(0.5, 4.5), (0.0, 60.0), (0.0, 60.0), # Hexagon set 1 parameters
              (0.5, 4.5), (0.0, 60.0), (0.0, 60.0)] # Hexagon set 2 parameters

    # --- Optimization using differential_evolution ---
    # With a much smaller 6D search space, we can afford a larger population and more 
    # iterations to conduct a more exhaustive and precise search within the time budget.
    
    result = differential_evolution(
        func=_objective_function_symmetric, # Use the symmetry-enforced objective function
        bounds=bounds, 
        maxiter=2000,     # Increased max iterations for a more exhaustive search (from inspirations)
        popsize=50,       # Increased population size for better exploration
        tol=1e-5,         # Tighter tolerance for a more precise solution (from inspirations)
        mutation=(0.5, 1.2),# Wider mutation range can help escape local minima (from inspirations)
        recombination=0.8,# Higher recombination probability
        polish=True,      # Apply local optimization (e.g., L-BFGS-B) at the end to refine the best solution
        disp=False,       # Do not display optimization progress
        workers=-1,       # Use all available CPU cores for parallel evaluation, speeding up computation
        init='latinhypercube', # Use Latin Hypercube Sampling for initial population diversity
        updating='deferred',   # Defer population update until all individuals are evaluated in parallel
        seed=42                # Set a seed for reproducibility
    )
    
    optimal_gen_params = result.x
    # The `result.fun` is the minimized `outer_hex_side_length` from the objective function
    # (calculated with num_orientation_samples=30 within the objective)
    # outer_hex_side_length_from_de = result.fun # This variable is no longer strictly needed

    # Reconstruct the full 12 inner hexagons from the 2 optimal generators
    inner_hex_data = np.zeros((n, 3))
    # Unpack optimal parameters: [r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg]
    r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg = optimal_gen_params

    hex_idx = 0
    final_hex_polygons = [] # Collect polygons for final high-res calculation
    
    # Generate inner hexagon data for output based on optimized parameters
    # Seed 1:
    for j in range(6):
        current_center_angle_degrees = phi1_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        
        cx = r1 * np.cos(current_center_angle_rad)
        cy = r1 * np.sin(current_center_angle_rad)
        hex_angle = (rot1_deg + current_center_angle_degrees) % 360
        inner_hex_data[hex_idx] = [cx, cy, hex_angle]
        final_hex_polygons.append(_create_hexagon_polygon(cx, cy, hex_angle, HEX_SIDE_LENGTH))
        hex_idx += 1

    # Seed 2:
    for j in range(6):
        current_center_angle_degrees = phi2_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        
        cx = r2 * np.cos(current_center_angle_rad)
        cy = r2 * np.sin(current_center_angle_rad)
        hex_angle = (rot2_deg + current_center_angle_degrees) % 360
        inner_hex_data[hex_idx] = [cx, cy, hex_angle]
        final_hex_polygons.append(_create_hexagon_polygon(cx, cy, hex_angle, HEX_SIDE_LENGTH))
        hex_idx += 1
            
    # Determine the optimal rotation of the outer hexagon for the found configuration.
    # We re-run `_calculate_min_enclosing_hex_side_length` with a higher angular resolution
    # to find the specific rotation that yields the precise minimal `outer_hex_side_length`
    # and its corresponding angle.
    
    # Use a higher number of samples for the final orientation determination for accuracy
    num_orientation_samples_final = 360 # Check every 0.08 degrees (30/360) for ultimate precision
    outer_hex_side_length, optimal_outer_rotation_rad = _calculate_min_enclosing_hex_side_length(
        final_hex_polygons, num_orientation_samples=num_orientation_samples_final)
    
    # The outer hexagon is considered centered at (0,0). Its orientation is `optimal_outer_rotation_rad`.
    outer_hex_data = np.array([0, 0, np.degrees(optimal_outer_rotation_rad)])
    # The `outer_hex_side_length` is already the precisely calculated value from the above call. 

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
