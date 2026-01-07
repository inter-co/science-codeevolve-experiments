# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import time # Added for performance metrics

# Constants for inner unit hexagons
HEX_SIDE_LENGTH = 1.0
SQRT3 = np.sqrt(3.0)

def create_hexagon_polygon(center_x, center_y, side_length, angle_degrees):
    """
    Creates a shapely Polygon for a regular hexagon.
    At angle_degrees=0, it's a pointy-top hexagon (one vertex on the positive x-axis).
    """
    angle_rad_offset = np.deg2rad(angle_degrees)
    vertices = []
    for i in range(6):
        # Angle for pointy-top hexagon (vertex at (side_length, 0) when angle_degrees=0)
        angle = np.deg2rad(i * 60) # Equivalent to i * np.pi / 3
        x = center_x + side_length * np.cos(angle + angle_rad_offset)
        y = center_y + side_length * np.sin(angle + angle_rad_offset)
        vertices.append((x, y))
    return Polygon(vertices)

# Removed _get_hexagon_vertices as it was redundant and could be replaced by direct shapely calls.

def _calculate_min_enclosing_hex_side_length(hexagons_polygons, num_orientation_samples=60):
    """
    Calculates the minimum side length of a regular hexagon that contains all given inner hexagon polygons.
    It does this by sampling different rotations for the outer hexagon.
    This function is adapted from Inspiration Program 2/3 but modified to correctly handle
    a pointy-top outer hexagon at 0 degrees rotation.
    """
    if not hexagons_polygons:
        return np.inf, 0.0 # Return large value if no hexagons

    # Collect all vertices from all inner hexagons
    all_vertices = np.vstack([np.array(poly.exterior.coords[:-1]) for poly in hexagons_polygons])

    min_R = np.inf
    optimal_outer_rotation_rad = 0.0

    # Iterate through a range of possible outer hexagon rotations
    # Sampling within a 30-degree range (np.pi/6) is sufficient for finding the minimal R.
    for i in range(num_orientation_samples + 1):
        # Sample angle from 0 to pi/6 (30 degrees)
        theta = i * (np.pi / 6) / num_orientation_samples
        
        # Rotate all vertices by -theta to effectively rotate the bounding box by +theta.
        # This means the current coordinate system is aligned with a candidate outer hexagon rotated by `theta`.
        cos_m_theta, sin_m_theta = np.cos(-theta), np.sin(-theta)
        # Standard rotation matrix for rotating points (x, y) by an angle 'alpha':
        # [x_new, y_new] = [x, y] @ [[cos(alpha), sin(alpha)], [-sin(alpha), cos(alpha)]]
        # Here, alpha = -theta
        rot_matrix = np.array([[cos_m_theta, sin_m_theta], [-sin_m_theta, cos_m_theta]])
        rotated_vertices = all_vertices @ rot_matrix # Corrected rotation matrix application
        
        rx, ry = rotated_vertices[:, 0], rotated_vertices[:, 1]

        # Calculate bounding R for a pointy-top hexagon (with 0 rotation)
        # For a pointy-top hexagon of side R (centered at origin, 0 rotation):
        # A point (x,y) is contained if:
        # |x| <= R
        # |y - sqrt(3)*x| / sqrt(3) <= R  => |y/sqrt(3) - x| <= R
        # |y + sqrt(3)*x| / sqrt(3) <= R  => |y/sqrt(3) + x| <= R
        # The maximum of these three conditions for all points gives the minimal R for THIS orientation.
        r_comp1 = np.abs(rx) # Equivalent to |x| <= R
        r_comp2 = np.abs(ry - SQRT3 * rx) / SQRT3 # Equivalent to |y'| <= R for axis at 60deg
        r_comp3 = np.abs(ry + SQRT3 * rx) / SQRT3 # Equivalent to |y''| <= R for axis at 120deg
        
        max_R_for_this_theta = np.max(np.maximum.reduce([r_comp1, r_comp2, r_comp3]))
        
        if max_R_for_this_theta < min_R:
            min_R = max_R_for_this_theta
            optimal_outer_rotation_rad = theta

    return min_R, optimal_outer_rotation_rad

def objective_function(params):
    """
    Objective function for the optimizer. Calculates the outer hexagon side length for a given configuration.
    A large penalty is returned if inner hexagons overlap.
    This function is enhanced to include individual rotations for generator hexagons and improved overlap checks,
    inspired by Inspiration Programs 1 and 2.
    """
    # Unpack parameters: [r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg]
    # r: radial distance of generator hex center from global origin
    # phi_deg: angular position of generator hex center within the 0-60 degree sector (degrees)
    # rot_deg: rotation of the generator hexagon (in degrees) *relative to its radial line* (0-60 degrees)
    
    r1, phi1_deg, rot1_deg, r2, phi2_deg, rot2_deg = params

    # Penalize if generators are too close to the origin, leading to degenerate symmetric copies
    if r1 < 1e-6 or r2 < 1e-6:
        return np.inf

    all_inner_hex_polygons = []
    inner_hex_centers = [] # Storing centers to optimize overlap checks

    # Generate the 12 hexagons using 6-fold rotational symmetry for two "seed" hexagons
    # Seed 1:
    for j in range(6):
        current_center_angle_degrees = phi1_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        
        cx = r1 * np.cos(current_center_angle_rad)
        cy = r1 * np.sin(current_center_angle_rad)
        
        # Hexagon's absolute angle: its own rot_deg + the angle of its center from global x-axis
        hex_angle = (rot1_deg + current_center_angle_degrees) % 360
        
        all_inner_hex_polygons.append(create_hexagon_polygon(cx, cy, HEX_SIDE_LENGTH, hex_angle))
        inner_hex_centers.append((cx, cy))

    # Seed 2:
    for j in range(6):
        current_center_angle_degrees = phi2_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        
        cx = r2 * np.cos(current_center_angle_rad)
        cy = r2 * np.sin(current_center_angle_rad)
        
        hex_angle = (rot2_deg + current_center_angle_degrees) % 360
        
        all_inner_hex_polygons.append(create_hexagon_polygon(cx, cy, HEX_SIDE_LENGTH, hex_angle))
        inner_hex_centers.append((cx, cy))

    # Overlap check (optimized with center distance pre-check from Inspiration 1, refined with area check from IP2/3)
    # Minimum distance between centers for two non-overlapping unit hexagons is 2 * HEX_SIDE_LENGTH (i.e., 2.0)
    # Using a small tolerance for floating point comparisons to handle numerical precision.
    MIN_CENTER_DIST_SQ = (2 * HEX_SIDE_LENGTH - 1e-7)**2 # Adjusted tolerance
    
    for i in range(len(all_inner_hex_polygons)):
        for j in range(i + 1, len(all_inner_hex_polygons)):
            center1 = inner_hex_centers[i]
            center2 = inner_hex_centers[j]
            dist_centers_sq = (center1[0] - center2[0])**2 + (center1[1] - center2[1])**2
            
            # Quick check based on center distance. If centers are too close, they must overlap.
            # Only perform shapely.intersects if centers are close, as it's computationally more expensive.
            if dist_centers_sq < MIN_CENTER_DIST_SQ:
                if all_inner_hex_polygons[i].intersects(all_inner_hex_polygons[j]):
                    # Penalize only significant overlaps to allow for floating point tolerances (from IP2/3)
                    intersection_area = all_inner_hex_polygons[i].intersection(all_inner_hex_polygons[j]).area
                    if intersection_area > 1e-6: # Threshold for significant overlap
                        return np.inf # Use np.inf for maximal penalty
                        
    # Calculate outer hexagon side length by optimizing its rotation (from Inspiration 2/3)
    outer_hex_side_length, _ = _calculate_min_enclosing_hex_side_length(all_inner_hex_polygons, num_orientation_samples=60)
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

    # Run differential evolution for global optimization. (Aggressive parameters from Inspirations 1 & 2)
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
        all_inner_hex_polygons.append(create_hexagon_polygon(cx, cy, HEX_SIDE_LENGTH, hex_angle))


    for j in range(6):
        current_center_angle_degrees = phi2_deg + j * 60
        current_center_angle_rad = np.deg2rad(current_center_angle_degrees)
        
        cx = r2 * np.cos(current_center_angle_rad)
        cy = r2 * np.sin(current_center_angle_rad)
        hex_angle = (rot2_deg + current_center_angle_degrees) % 360
        inner_hex_data_list.append([cx, cy, hex_angle])
        all_inner_hex_polygons.append(create_hexagon_polygon(cx, cy, HEX_SIDE_LENGTH, hex_angle))

    inner_hex_data = np.array(inner_hex_data_list)

    # Calculate final outer hex side length and its optimal rotation with high sample count
    outer_hex_side_length, optimal_outer_rotation_rad = _calculate_min_enclosing_hex_side_length(all_inner_hex_polygons, num_orientation_samples=360)
    
    # Outer hexagon is centered at origin, with the optimal rotation found
    outer_hex_data = np.array([0, 0, np.degrees(optimal_outer_rotation_rad)])
    
    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
    print(f"Optimal outer_hex_side_length: {outer_hex_side_length:.7f}")
    print(f"Optimal inv_outer_hex_side_length: {1/outer_hex_side_length:.7f}")
    print(f"Optimal parameters: {best_params}")

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
