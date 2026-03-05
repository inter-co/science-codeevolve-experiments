# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from numba import jit
import time
from scipy.optimize import differential_evolution, minimize

# Constants for regular hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APOGEE = HEX_RADIUS * np.sqrt(3)/2  # Distance from center to side midpoint
HEX_HEIGHT = 2 * HEX_APOGEE  # Height of hexagon
HEX_WIDTH = 2 * HEX_RADIUS  # Width of hexagon

@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, angle_deg, radius=HEX_RADIUS):
    """Fast computation of hexagon vertices using numba"""
    angle_rad = np.deg2rad(angle_deg)
    vertices = np.zeros((6, 2))
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        vertices[i, 0] = center_x + radius * np.cos(angle)
        vertices[i, 1] = center_y + radius * np.sin(angle)
    return vertices

def get_hexagon_vertices(center_x, center_y, angle_deg, radius=HEX_RADIUS):
    """Get vertices of a regular hexagon given center, angle, and radius"""
    return hexagon_vertices_jit(center_x, center_y, angle_deg, radius)

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

def check_containment_and_overlap(inner_hex_data, outer_radius):
    """Check if all inner hexagons are contained in outer hexagon and no overlaps exist"""
    # Create outer hexagon vertices
    outer_vertices = get_hexagon_vertices(0, 0, 0, outer_radius)
    outer_hex = Polygon(outer_vertices)
    
    # Create inner hexagons and check containment and overlaps
    inner_hexagons = []
    for i, (cx, cy, angle) in enumerate(inner_hex_data):
        inner_vertices = get_hexagon_vertices(cx, cy, angle)
        inner_hex = Polygon(inner_vertices)
        inner_hexagons.append(inner_hex)
        
        # Check if inner hexagon is fully contained
        if not check_containment(inner_hex, outer_hex):
            return False, "Not contained"
        
        # Check for overlaps with other hexagons
        for j in range(i):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return False, "Overlap detected"
    
    return True, "Valid"

def compute_outer_hexagon_side_length(inner_hex_data, side_length=1):
    """Compute the minimal side length needed to enclose all hexagons."""
    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = inner_hex_data[i][2]
        vertices = get_hexagon_vertices(center[0], center[1], angle, side_length)
        all_vertices.extend(vertices)
    
    if not all_vertices:
        return 1.0
    
    # Center of all vertices (centroid)
    centroid = np.mean(all_vertices, axis=0)
    
    # Maximum distance from centroid to any vertex
    distances = np.linalg.norm(np.array(all_vertices) - centroid, axis=1)
    max_distance = np.max(distances)
    
    # For a regular hexagon, the distance from center to corner equals side length
    # But we want the side length of the enclosing hexagon, which needs to be slightly larger
    # to properly contain all vertices (we'll use a small safety margin)
    return max_distance * 1.01  # Small margin for numerical stability

def objective_function(params):
    """
    Objective function to minimize (negative of 1/outer_hex_side_length)
    params: flattened array of [x1, y1, angle1, x2, y2, angle2, ..., x11, y11, angle11, outer_side_length]
    """
    # Extract parameters
    n = 11
    inner_params = params[:-1]  # First 33 parameters: 11 hexagons * 3 params each
    outer_side_length = params[-1]   # Last parameter: outer hexagon side length
    
    # Reshape inner hexagon parameters
    inner_hex_data = inner_params.reshape(-1, 3)
    
    # Check if the configuration is valid
    is_valid, message = check_containment_and_overlap(inner_hex_data, outer_side_length)
    
    if not is_valid:
        # Return a large penalty value for invalid configurations
        return 1e10
    
    # Return negative of 1/outer_side_length (we want to maximize 1/outer_side_length)
    return -1.0 / outer_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining multiple initial configurations with both global and local optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    start_time = time.time()
    
    # Multiple high-quality initial configurations inspired by hexagonal packing theory
    configs = []
    
    # Configuration 1: Compact hexagonal arrangement (inspired by hexagonal lattice)
    config1 = np.array([
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
    configs.append(config1)
    
    # Configuration 2: More spread-out for better optimization flexibility
    config2 = np.array([
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
    configs.append(config2)
    
    # Configuration 3: Optimized compact arrangement with better spacing
    config3 = np.array([
        [0.0, 0.0, 0.0],      # center
        [0.0, 1.8, 0.0],      # top
        [1.55, 0.9, 0.0],     # top-right
        [1.55, -0.9, 0.0],    # bottom-right
        [0.0, -1.8, 0.0],     # bottom
        [-1.55, -0.9, 0.0],   # bottom-left
        [-1.55, 0.9, 0.0],    # top-left
        [3.1, 0.0, 0.0],      # far right
        [-3.1, 0.0, 0.0],     # far left
        [1.55, 2.7, 0.0],     # top-right extended
        [-1.55, 2.7, 0.0],    # top-left extended
    ])
    configs.append(config3)
    
    # Configuration 4: Zigzag pattern for diversity
    config4 = np.array([
        [0.0, 0.0, 0.0],      # center
        [0.0, 2.0, 0.0],      # top
        [1.732, 1.0, 0.0],    # top-right
        [1.732, -1.0, 0.0],   # bottom-right
        [0.0, -2.0, 0.0],     # bottom
        [-1.732, -1.0, 0.0],  # bottom-left
        [-1.732, 1.0, 0.0],   # top-left
        [3.464, 0.0, 0.0],    # far right
        [-3.464, 0.0, 0.0],   # far left
        [0.0, 3.0, 0.0],      # far top
        [0.0, -3.0, 0.0],     # far bottom
    ])
    configs.append(config4)
    
    # Configuration 5: Precise mathematical configuration from inspirations
    config5 = np.array([
        [0.0, 0.0, 0.0],       # center
        [0.0, 2.0, 0.0],       # top
        [1.7320508075688772, 1.0, 0.0],  # top-right (sqrt(3) ~ 1.732)
        [1.7320508075688772, -1.0, 0.0], # bottom-right
        [0.0, -2.0, 0.0],      # bottom
        [-1.7320508075688772, -1.0, 0.0], # bottom-left
        [-1.7320508075688772, 1.0, 0.0],  # top-left
        [3.4641016151377544, 0.0, 0.0],   # far right (2*sqrt(3))
        [-3.4641016151377544, 0.0, 0.0],  # far left
        [1.7320508075688772, 2.918033988749895, 0.0],   # top-top-right - maximally precise
        [-1.7320508075688772, 2.918033988749895, 0.0]   # top-top-left - maximally precise
    ], dtype=np.float64)
    configs.append(config5)
    
    # Configuration 6: Inspired by the best solution from inspiration 1
    config6 = np.array([
        [0.0, 0.0, 0.0],       # center
        [0.0, 1.9, 0.0],       # top
        [1.64, 0.95, 0.0],     # top-right
        [1.64, -0.95, 0.0],    # bottom-right
        [0.0, -1.9, 0.0],      # bottom
        [-1.64, -0.95, 0.0],   # bottom-left
        [-1.64, 0.95, 0.0],    # top-left
        [3.28, 0.0, 0.0],      # far right
        [-3.28, 0.0, 0.0],     # far left
        [1.64, 2.85, 0.0],     # top-right extended
        [-1.64, 2.85, 0.0],    # top-left extended
    ])
    configs.append(config6)
    
    best_inv_side_length = 0.0
    best_result = None
    best_config_idx = -1
    
    # Try each configuration with optimization
    for i, initial_config in enumerate(configs):
        # Check if we're running out of time
        if time.time() - start_time > 50.0:  # Leave 10 seconds for final processing
            break
            
        # Compute initial outer hexagon size
        initial_outer_side_length = compute_outer_hexagon_side_length(initial_config)
        
        # Set up optimization parameters
        # Flatten the initial configuration for optimization
        initial_guess = []
        
        # Add positions and rotations from initial config
        for j in range(n):
            initial_guess.extend([initial_config[j][0], initial_config[j][1], initial_config[j][2]])
        
        # Add outer side length
        initial_guess.append(initial_outer_side_length)
        
        # Define bounds for optimization
        bounds = []
        
        # Bounds for positions (x, y) - more generous bounds for exploration
        pos_bounds = [(-10, 10)] * (2 * n)  # x and y coordinates for each hexagon
        bounds.extend(pos_bounds)
        
        # Bounds for rotations (0 to 360 degrees)
        rot_bounds = [(0, 360)] * n
        bounds.extend(rot_bounds)
        
        # Bounds for outer hexagon side length (reasonable range)
        bounds.append((1.0, 15.0))  # outer side length
        
        try:
            # First try differential evolution for global search with enhanced parameters
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,   # More iterations for better search
                popsize=15,   # Larger population for better exploration
                mutation=(0.9, 1),  # Higher mutation for exploration
                recombination=0.95,  # High recombination for diversity
                seed=42+i,
                disp=False,
                tol=1e-8
            )
            
            # If DE was successful and gave better result, try local refinement
            if de_result.success and -de_result.fun > best_inv_side_length:
                # Try local optimization with L-BFGS-B for fine-tuning
                try:
                    local_result = minimize(
                        objective_function,
                        de_result.x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-14},
                        tol=1e-14
                    )
                    
                    if local_result.success and -local_result.fun > best_inv_side_length:
                        best_inv_side_length = -local_result.fun
                        best_result = local_result
                        best_config_idx = i
                    elif de_result.success and -de_result.fun > best_inv_side_length:
                        best_inv_side_length = -de_result.fun
                        best_result = de_result
                        best_config_idx = i
                        
                except Exception:
                    if de_result.success and -de_result.fun > best_inv_side_length:
                        best_inv_side_length = -de_result.fun
                        best_result = de_result
                        best_config_idx = i
                        
            elif de_result.success and -de_result.fun > best_inv_side_length:
                best_inv_side_length = -de_result.fun
                best_result = de_result
                best_config_idx = i
                
        except Exception:
            continue
    
    # If we found a good result, use it
    if best_result is not None and best_result.success:
        params = best_result.x
        inner_params = params[:-1]
        outer_side_length = params[-1]
        inner_hex_data = inner_params.reshape(-1, 3)
        outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
        return inner_hex_data, outer_hex_data, outer_side_length
    
    # Fallback to best configuration from our list if optimization fails
    if best_config_idx >= 0:
        inner_hex_data = configs[best_config_idx]
        outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
        outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
        return inner_hex_data, outer_hex_data, outer_side_length
    
    # Last resort: use the first configuration
    inner_hex_data = configs[0]
    outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
    outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
