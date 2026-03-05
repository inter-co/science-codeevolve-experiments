# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
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


def validate_and_score_config(config):
    """Validate configuration and return score (inverse side length) if valid."""
    # Create outer hexagon (just large enough to contain everything)
    outer_side_length = compute_outer_hexagon_side_length(config)
    outer_hex = create_hexagon_polygon(0, 0, outer_side_length, 0)
    
    # Create inner hexagons
    inner_hexagons = []
    for i in range(len(config)):
        hexagon = create_hexagon_polygon(
            config[i][0], 
            config[i][1], 
            1.0,  # unit hexagon
            config[i][2]
        )
        inner_hexagons.append(hexagon)
    
    # Check containment
    for hexagon in inner_hexagons:
        if not check_containment(hexagon, outer_hex):
            return None, float('inf')
    
    # Check overlaps
    for i in range(len(config)):
        for j in range(i+1, len(config)):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return None, float('inf')
    
    return config, 1.0 / outer_side_length


def generate_best_initial_configs():
    """Generate the best initial configurations based on inspiration programs."""
    configs = []
    
    # The best known configuration from inspiration 1 and 2
    # This is a highly optimized configuration that achieved very good results
    config1 = [
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
    configs.append(config1)
    
    # A more balanced configuration from inspiration 3
    config2 = [
        [0, 0, 0],              # center
        [0, 2.0, 0],            # top
        [1.732, 1.0, 0],        # top-right 
        [1.732, -1.0, 0],       # bottom-right
        [0, -2.0, 0],           # bottom
        [-1.732, -1.0, 0],      # bottom-left
        [-1.732, 1.0, 0],       # top-left
        [3.464, 0, 0],          # far right
        [-3.464, 0, 0],         # far left
        [1.732, 3.0, 0],        # top-top-right
        [-1.732, 3.0, 0]        # top-top-left
    ]
    configs.append(config2)
    
    # Compact configuration that balances density and feasibility
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
    
    # Slightly adjusted version for better optimization
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
    bounds.append((1.0, 20.0))  # outer side length - wider range for optimization
    
    best_result = None
    best_inv_side_length = best_score
    
    try:
        # Use a multi-stage optimization approach
        # Stage 1: Differential Evolution for global search (more thorough)
        de_result = differential_evolution(
            objective_function,
            bounds,
            maxiter=60,    # More iterations for better convergence
            popsize=20,    # Larger population for better exploration
            mutation=(0.8, 1.0),  # Good balance of exploration/exploitation
            recombination=0.9,    # High recombination rate
            seed=42,
            disp=False,
            tol=1e-12  # Very tight tolerance
        )
        
        if de_result.success:
            inv_side_length = -de_result.fun
            if inv_side_length > best_inv_side_length:
                best_inv_side_length = inv_side_length
                best_result = de_result
        
        # Stage 2: Local optimization with L-BFGS-B for fine-tuning
        local_result = minimize(
            objective_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 250, 'ftol': 1e-15, 'gtol': 1e-15},
            tol=1e-15
        )
        
        if local_result.success:
            inv_side_length = -local_result.fun
            if inv_side_length > best_inv_side_length:
                best_inv_side_length = inv_side_length
                best_result = local_result
                
        # Stage 3: Try a second local optimization with different starting point
        # Use the DE result as starting point for additional refinement
        if best_result is not None and best_result.success:
            de_start = best_result.x
            local_result2 = minimize(
                objective_function,
                de_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-15},
                tol=1e-15
            )
            
            if local_result2.success:
                inv_side_length = -local_result2.fun
                if inv_side_length > best_inv_side_length:
                    best_inv_side_length = inv_side_length
                    best_result = local_result2
                    
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
