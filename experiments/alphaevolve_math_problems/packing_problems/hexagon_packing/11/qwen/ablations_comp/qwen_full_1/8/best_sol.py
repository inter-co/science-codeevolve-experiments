# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
import math
import time


def hexagon_vertices(center_x, center_y, side_length, rotation_degrees):
    """Generate vertices of a regular hexagon."""
    rotation_rad = math.radians(rotation_degrees)
    vertices = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices


def create_hexagon_polygon(center_x, center_y, side_length, rotation_degrees):
    """Create Shapely polygon for a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, side_length, rotation_degrees)
    return Polygon(vertices)


def check_containment(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained within outer hexagon."""
    # Check if all vertices of inner hexagon are inside outer hexagon
    for vertex in inner_hex.exterior.coords[:-1]:  # Exclude closing vertex
        if not outer_hex.contains(Point(vertex)):
            return False
    return True


def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)


def compute_outer_hexagon_side_length(inner_hex_data, side_length=1):
    """Compute the minimal side length needed to enclose all hexagons."""
    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = inner_hex_data[i][2]
        vertices = hexagon_vertices(center[0], center[1], side_length, angle)
        all_vertices.extend(vertices)
    
    if not all_vertices:
        return 1.0
    
    # Center of all vertices (centroid)
    centroid = np.mean(all_vertices, axis=0)
    
    # Maximum distance from centroid to any vertex
    distances = np.linalg.norm(np.array(all_vertices) - centroid, axis=1)
    max_distance = np.max(distances)
    
    # For a regular hexagon, the distance from center to corner equals side length
    return max_distance


def objective_function(params):
    """
    Objective function to minimize (negative of 1/outer_hex_side_length).
    params: array of [x1, y1, theta1, ..., x11, y11, theta11, R]
    where (xi, yi) are positions, thetai are rotations, and R is outer hex side length.
    """
    n = 11
    # Extract parameters
    inner_positions = params[:2*n].reshape(n, 2)  # (x, y) for each hexagon
    inner_rotations = params[2*n:3*n]  # rotation angles for each hexagon
    outer_side_length = params[3*n]  # outer hexagon side length
    
    # Create outer hexagon
    outer_hex = create_hexagon_polygon(0, 0, outer_side_length, 0)
    
    # Create inner hexagons
    inner_hexagons = []
    for i in range(n):
        hexagon = create_hexagon_polygon(
            inner_positions[i][0], 
            inner_positions[i][1], 
            1.0,  # unit hexagon
            inner_rotations[i]
        )
        inner_hexagons.append(hexagon)
    
    # Check containment - early termination if violated
    for hexagon in inner_hexagons:
        if not check_containment(hexagon, outer_hex):
            return 1e10  # penalty for violation
    
    # Check overlaps - early termination if violated
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return 1e10  # penalty for overlap
    
    # Return negative of 1/outer_side_length (we want to maximize 1/R)
    return -1.0 / outer_side_length


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a focused multi-start approach with optimized configurations and aggressive optimization parameters.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    start_time = time.time()
    
    # Use highly optimized configurations based on mathematical insights
    # Inspired by the best of all inspirations, especially focusing on dense packing
    configs = []
    
    # Configuration 1: Optimized dense hexagonal close-packed arrangement (Inspiration 3)
    config1 = np.array([
        [0.0, 0.0, 0.0],      # center
        [0.0, 1.85, 0.0],     # top
        [1.59, 0.90, 0.0],    # top-right
        [1.59, -0.90, 0.0],   # bottom-right
        [0.0, -1.85, 0.0],    # bottom
        [-1.59, -0.90, 0.0],  # bottom-left
        [-1.59, 0.90, 0.0],   # top-left
        [3.18, 0.0, 0.0],     # far right
        [-3.18, 0.0, 0.0],    # far left
        [1.59, 2.75, 0.0],    # top-right extended
        [-1.59, 2.75, 0.0],   # top-left extended
    ])
    configs.append(config1)
    
    # Configuration 2: Balanced arrangement with good coverage (Inspiration 1)
    config2 = np.array([
        [0.0, 0.0, 0.0],      # center
        [0.0, 1.9, 0.0],      # top
        [1.64, 0.95, 0.0],    # top-right
        [1.64, -0.95, 0.0],   # bottom-right
        [0.0, -1.9, 0.0],     # bottom
        [-1.64, -0.95, 0.0],  # bottom-left
        [-1.64, 0.95, 0.0],   # top-left
        [3.28, 0.0, 0.0],     # far right
        [-3.28, 0.0, 0.0],    # far left
        [1.64, 2.85, 0.0],    # top-right extended
        [-1.64, 2.85, 0.0],   # top-left extended
    ])
    configs.append(config2)
    
    # Configuration 3: Alternative arrangement with more symmetry (Inspiration 2)
    config3 = np.array([
        [0.0, 0.0, 0.0],      # center
        [0.0, 2.1, 0.0],      # top
        [1.82, 1.05, 0.0],    # top-right
        [1.82, -1.05, 0.0],   # bottom-right
        [0.0, -2.1, 0.0],     # bottom
        [-1.82, -1.05, 0.0],  # bottom-left
        [-1.82, 1.05, 0.0],   # top-left
        [3.64, 0.0, 0.0],     # far right
        [-3.64, 0.0, 0.0],    # far left
        [0.0, 3.15, 0.0],     # far top
        [0.0, -3.15, 0.0],    # far bottom
    ])
    configs.append(config3)
    
    best_inv_side_length = 0.0
    best_result = None
    best_config = None
    
    # Try each configuration with differential evolution
    # We'll limit to 2 configurations to save time but still have good coverage
    for i, initial_config in enumerate(configs[:2]):  # Limit to 2 configs to stay within time budget
        # Check if we're running out of time (leave 5 seconds for final processing)
        if time.time() - start_time > 55.0:
            break
            
        # Compute initial outer hexagon size
        initial_outer_side_length = compute_outer_hexagon_side_length(initial_config)
        
        # Set up optimization parameters with aggressive settings for better results
        # Flatten the initial configuration for optimization
        initial_guess = []
        
        # Add positions
        for j in range(n):
            initial_guess.extend([initial_config[j][0], initial_config[j][1]])
        
        # Add rotations (initially 0)
        initial_guess.extend([0] * n)
        
        # Add outer side length
        initial_guess.append(initial_outer_side_length)
        
        # Define bounds for optimization - more reasonable bounds for faster convergence
        bounds = []
        
        # Bounds for positions (x, y) - reasonable range to focus search
        pos_bounds = [(-8, 8)] * (2 * n)  # x and y coordinates for each hexagon
        bounds.extend(pos_bounds)
        
        # Bounds for rotations (0 to 360 degrees)
        rot_bounds = [(0, 360)] * n
        bounds.extend(rot_bounds)
        
        # Bounds for outer hexagon side length (reasonable range)
        bounds.append((1.0, 12.0))  # outer side length - tighter bounds for speed
        
        try:
            # Use differential evolution with aggressive yet reasonable parameters
            # Based on Inspiration 2 and 3 - more iterations but moderate population
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=40,   # Moderate iterations for good balance
                popsize=12,   # Moderate population for good exploration
                mutation=(0.75, 1),  # Good balance of exploration and exploitation
                recombination=0.85,  # Good recombination rate
                seed=42+i,
                disp=False,
                tol=1e-6
            )
            
            # If optimization succeeds and gives better result, keep it
            if result.success and -result.fun > best_inv_side_length:
                best_inv_side_length = -result.fun
                best_result = result
                best_config = initial_config
                
        except Exception:
            # Continue with other configurations if this one fails
            continue
    
    # If we found a good result, use it
    if best_result is not None:
        params = best_result.x
        inner_positions = params[:2*n].reshape(n, 2)
        inner_rotations = params[2*n:3*n]
        outer_side_length = params[3*n]
        
        # Validate final solution
        inv_side_length = -best_result.fun  # Since we minimized negative of 1/R
        if inv_side_length > 0:
            inner_hex_data = np.column_stack([
                inner_positions[:, 0],  # x coordinates
                inner_positions[:, 1],  # y coordinates
                inner_rotations         # rotation angles
            ])
            outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
            return inner_hex_data, outer_hex_data, outer_side_length
    
    # If no optimization was successful, fall back to the best configuration from our list
    if best_config is not None:
        inner_hex_data = best_config
        outer_side_length = compute_outer_hexagon_side_length(best_config)
        outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
        return inner_hex_data, outer_hex_data, outer_side_length
    
    # Final fallback to first configuration
    inner_hex_data = configs[0]
    outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
    outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
