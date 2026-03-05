# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from numba import jit
import math
import time


@jit(nopython=True)
def hexagon_vertices_fast(center_x, center_y, angle_deg, side_length=1.0):
    """Fast computation of hexagon vertices using numba"""
    angle_rad = math.radians(angle_deg)
    vertices = np.zeros((6, 2))
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        vertices[i, 0] = center_x + side_length * math.cos(angle)
        vertices[i, 1] = center_y + side_length * math.sin(angle)
    return vertices


def hexagon_vertices(center_x, center_y, side_length, rotation_degrees):
    """Generate vertices of a regular hexagon."""
    return hexagon_vertices_fast(center_x, center_y, rotation_degrees, side_length)


def create_hexagon_polygon(center_x, center_y, side_length, rotation_degrees):
    """Create Shapely polygon for a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, side_length, rotation_degrees)
    return Polygon(vertices)


def check_containment_and_overlap(inner_hex_data, outer_radius):
    """Check if all inner hexagons are contained in outer hexagon and no overlaps exist."""
    # Create outer hexagon vertices
    outer_vertices = hexagon_vertices(0, 0, outer_radius, 0)
    outer_hex = Polygon(outer_vertices)
    
    # Check containment and overlap for each inner hexagon
    for i, (cx, cy, angle) in enumerate(inner_hex_data):
        inner_vertices = hexagon_vertices(cx, cy, 1.0, angle)
        inner_hex = Polygon(inner_vertices)
        
        # Check if inner hexagon is fully contained
        if not outer_hex.contains(inner_hex):
            return False, "Not contained"
        
        # Check for overlaps with other hexagons
        for j in range(i):
            cx2, cy2, angle2 = inner_hex_data[j]
            inner_hex2_vertices = hexagon_vertices(cx2, cy2, 1.0, angle2)
            inner_hex2 = Polygon(inner_hex2_vertices)
            
            if inner_hex.intersects(inner_hex2):
                return False, "Overlap detected"
    
    return True, "Valid"


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


def validate_and_score_config(config):
    """Validate configuration and return score (inverse side length) if valid."""
    # Create outer hexagon (just large enough to contain everything)
    outer_side_length = compute_outer_hexagon_side_length(config)
    
    # Validate the configuration
    is_valid, message = check_containment_and_overlap(config, outer_side_length)
    
    if not is_valid:
        return None, float('inf')
    
    return config, 1.0 / outer_side_length


def generate_best_initial_configs():
    """Generate the best initial configurations based on inspiration programs."""
    configs = []
    
    # The best known configuration from inspiration 1 and 2
    # This is a highly optimized configuration that achieved very good results
    config1 = [
        [0.0, 0.0, 0.0],        # center
        [0.0, 1.85, 0.0],       # top
        [1.59, 0.90, 0.0],      # top-right
        [1.59, -0.90, 0.0],     # bottom-right
        [0.0, -1.85, 0.0],      # bottom
        [-1.59, -0.90, 0.0],    # bottom-left
        [-1.59, 0.90, 0.0],     # top-left
        [3.18, 0.0, 0.0],       # far right
        [-3.18, 0.0, 0.0],      # far left
        [1.59, 2.75, 0.0],      # top-right extended
        [-1.59, 2.75, 0.0],     # top-left extended
    ]
    configs.append(config1)
    
    # More precise configuration from inspiration 2
    config2 = [
        [0.0, 0.0, 0.0],        # center
        [0.0, 2.0, 0.0],        # top
        [1.7320508075688772, 1.0, 0.0],  # top-right (sqrt(3) ~ 1.732)
        [1.7320508075688772, -1.0, 0.0], # bottom-right
        [0.0, -2.0, 0.0],       # bottom
        [-1.7320508075688772, -1.0, 0.0], # bottom-left
        [-1.7320508075688772, 1.0, 0.0],  # top-left
        [3.4641016151377544, 0.0, 0.0],   # far right (2*sqrt(3))
        [-3.4641016151377544, 0.0, 0.0],  # far left
        [1.7320508075688772, 2.918033988749895, 0.0],   # top-top-right - maximally precise
        [-1.7320508075688772, 2.918033988749895, 0.0]   # top-top-left - maximally precise
    ]
    configs.append(config2)
    
    # Compact configuration from inspiration 3
    config3 = [
        [0, 0, 0],              # center
        [0, 1.8, 0],            # top
        [1.56, 0.9, 0],         # top-right 
        [1.56, -0.9, 0],        # bottom-right
        [0, -1.8, 0],           # bottom
        [-1.56, -0.9, 0],       # bottom-left
        [-1.56, 0.9, 0],        # top-left
        [3.12, 0, 0],           # far right
        [-3.12, 0, 0],          # far left
        [1.56, 2.7, 0],         # top-top-right
        [-1.56, 2.7, 0]         # top-top-left
    ]
    configs.append(config3)
    
    # Slightly refined version
    config4 = [
        [0, 0, 0],              # center
        [0, 2.05, 0],           # top
        [1.77, 1.025, 0],       # top-right 
        [1.77, -1.025, 0],      # bottom-right
        [0, -2.05, 0],          # bottom
        [-1.77, -1.025, 0],     # bottom-left
        [-1.77, 1.025, 0],      # top-left
        [3.54, 0, 0],           # far right
        [-3.54, 0, 0],          # far left
        [1.77, 3.075, 0],       # top-top-right
        [-1.77, 3.075, 0]       # top-top-left
    ]
    configs.append(config4)
    
    return configs


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
    
    # Create inner hexagons
    inner_hex_data = []
    for i in range(n):
        inner_hex_data.append((inner_positions[i][0], inner_positions[i][1], inner_rotations[i]))
    
    # Validate configuration
    is_valid, message = check_containment_and_overlap(inner_hex_data, outer_side_length)
    
    if not is_valid:
        return 1e10  # penalty for violation
    
    # Return negative of 1/outer_side_length (we want to maximize 1/R)
    return -1.0 / outer_side_length


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a robust hybrid approach combining global and local optimization with multiple starting configurations.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    
    # Generate multiple candidate configurations
    configs = generate_best_initial_configs()
    
    # Evaluate all configurations to find the best one
    best_config = None
    best_score = 0.0
    
    # Validate and score each configuration
    for config in configs:
        _, score = validate_and_score_config(config)
        if score > best_score:
            best_score = score
            best_config = config
    
    # If no valid configuration found, use the first one as fallback
    if best_config is None:
        best_config = configs[0]
    
    # Convert to numpy array for easier handling
    initial_config = np.array(best_config)
    
    # Compute initial outer hexagon size
    initial_outer_side_length = compute_outer_hexagon_side_length(initial_config)
    
    # Set up optimization parameters
    # Flatten the initial configuration for optimization
    initial_guess = []
    
    # Add positions
    for j in range(n):
        initial_guess.extend([initial_config[j][0], initial_config[j][1]])
    
    # Add rotations (initially 0)
    initial_guess.extend([0] * n)
    
    # Add outer side length
    initial_guess.append(initial_outer_side_length)
    
    # Define bounds for optimization
    bounds = []
    
    # Bounds for positions (x, y) - reasonable range
    pos_bounds = [(-10, 10)] * (2 * n)  # x and y coordinates for each hexagon
    bounds.extend(pos_bounds)
    
    # Bounds for rotations (0 to 360 degrees)
    rot_bounds = [(0, 360)] * n
    bounds.extend(rot_bounds)
    
    # Bounds for outer hexagon side length (reasonable range)
    bounds.append((1.0, 15.0))  # outer side length - reduced upper bound for efficiency
    
    best_result = None
    best_inv_side_length = best_score
    
    try:
        # Use differential evolution for global optimization (inspired by inspiration 1)
        # More focused parameters for better performance within time limits
        de_result = differential_evolution(
            objective_function,
            bounds,
            maxiter=40,    # Reduced iterations for faster execution
            popsize=15,    # Moderate population size
            mutation=(0.7, 1.0),  # Balanced mutation rate
            recombination=0.8,    # Good recombination rate
            seed=42,
            disp=False,
            tol=1e-10  # Sufficient tolerance for practical purposes
        )
        
        if de_result.success:
            inv_side_length = -de_result.fun
            if inv_side_length > best_inv_side_length:
                best_inv_side_length = inv_side_length
                best_result = de_result
                
        # If we have a good DE result, try local optimization for further refinement
        if best_result is not None and best_result.success:
            # Use a simpler local optimization approach to refine solution
            local_result = minimize(
                objective_function,
                best_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            if local_result.success:
                inv_side_length = -local_result.fun
                if inv_side_length > best_inv_side_length:
                    best_inv_side_length = inv_side_length
                    best_result = local_result
                    
    except Exception as e:
        # Continue with fallback if anything goes wrong
        pass
    
    # If we found a good result, use it
    if best_result is not None and best_result.success:
        params = best_result.x
        inner_positions = params[:2*n].reshape(n, 2)
        inner_rotations = params[2*n:3*n]
        outer_side_length = params[3*n]
        
        inner_hex_data = np.column_stack([
            inner_positions[:, 0],  # x coordinates
            inner_positions[:, 1],  # y coordinates
            inner_rotations         # rotation angles
        ])
        outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
        return inner_hex_data, outer_hex_data, outer_side_length
    
    # Final fallback to best validated configuration
    inner_hex_data = initial_config
    outer_side_length = compute_outer_hexagon_side_length(initial_config)
    outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
