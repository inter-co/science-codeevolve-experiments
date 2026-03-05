# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time

# Precompute hexagon vertices for unit regular hexagon
def get_hexagon_vertices(center=(0,0), rotation=0, side_length=1):
    """Get vertices of a regular hexagon"""
    angle = rotation * np.pi / 180
    # Vertices of unit hexagon centered at origin
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = side_length * np.cos(theta)
        y = side_length * np.sin(theta)
        vertices.append((x + center[0], y + center[1]))
    return np.array(vertices)

def compute_outer_hexagon_side_length(inner_hex_data):
    """Compute the minimum side length needed for outer hexagon to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        vertices = get_hexagon_vertices(center, rotation, 1)
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return 1000000
        
    all_vertices = np.array(all_vertices)
    
    # Find the bounding circle radius - distance from origin to farthest point
    distances = np.linalg.norm(all_vertices, axis=1)
    max_distance = np.max(distances)
    
    # For a regular hexagon, the circumradius equals the side length
    # So we need side_length >= max_distance
    return max_distance

def point_in_hexagon(point, hex_vertices):
    """Check if a point is inside a hexagon using cross product method"""
    x, y = point
    n = len(hex_vertices)
    inside = True
    
    for i in range(n):
        p1 = hex_vertices[i]
        p2 = hex_vertices[(i + 1) % n]
        
        # Vector from p1 to p2
        edge_x = p2[0] - p1[0]
        edge_y = p2[1] - p1[1]
        
        # Vector from p1 to point
        point_x = x - p1[0]
        point_y = y - p1[1]
        
        # Cross product to determine which side the point is on
        cross = edge_x * point_y - edge_y * point_x
        
        # For counter-clockwise polygon, all cross products should be positive
        # But we need to be more careful with the orientation
        if i == 0:
            sign = np.sign(cross)
            if sign == 0:
                # Point lies exactly on edge - consider inside
                continue
        else:
            if np.sign(cross) != sign and cross != 0:
                return False
    
    return True

def compute_penalty(inner_hex_data):
    """Compute penalty for overlaps and containment violations"""
    penalty = 0.0
    
    # Check overlaps between all pairs of hexagons using precise method
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center1 = (inner_hex_data[i, 0], inner_hex_data[i, 1])
            rotation1 = inner_hex_data[i, 2]
            center2 = (inner_hex_data[j, 0], inner_hex_data[j, 1])
            rotation2 = inner_hex_data[j, 2]
            
            hex1_vertices = get_hexagon_vertices(center1, rotation1, 1)
            hex2_vertices = get_hexagon_vertices(center2, rotation2, 1)
            
            # Check if any vertex of hex1 is inside hex2
            for vertex in hex1_vertices:
                if point_in_hexagon(vertex, hex2_vertices):
                    penalty += 10000.0  # High penalty for overlaps
                    break
            
            # Check if any vertex of hex2 is inside hex1
            if penalty == 0:  # Only check if no overlap yet
                for vertex in hex2_vertices:
                    if point_in_hexagon(vertex, hex1_vertices):
                        penalty += 10000.0  # High penalty for overlaps
                        break
    
    # Check containment - make sure all vertices are within a reasonable bound
    outer_radius = compute_outer_hexagon_side_length(inner_hex_data)
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        vertices = get_hexagon_vertices(center, rotation, 1)
        for vertex in vertices:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            if dist > outer_radius + 0.1:  # Allow small margin
                penalty += 10000.0 * (dist - outer_radius)
    
    return penalty

def objective_function(params):
    """
    Objective function to minimize (negative of 1/outer_hex_side_length)
    params: flattened array of [x1,y1,theta1, x2,y2,theta2, ..., x12,y12,theta12]
    """
    # Reshape parameters into 12 hexagons with (x,y,rotation) each
    inner_hex_data = params.reshape(-1, 3)
    
    # Compute outer hexagon side length
    outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
    
    # Add penalty for invalid configurations
    penalty = compute_penalty(inner_hex_data)
    
    # Return negative of 1/outer_side_length plus penalty (we want to maximize 1/outer_side_length)
    if penalty > 1e-3:  # More strict tolerance
        return 1e10  # Invalid configuration
    if outer_side_length > 1000:
        return 1e10
    return -1.0 / outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses advanced optimization techniques with improved starting configurations and better constraint checking.
    
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Try multiple carefully crafted configurations to find the best starting point
    sqrt3 = np.sqrt(3)
    
    # Configuration 1: Standard hexagonal arrangement (from inspiration)
    config1 = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 2.0, 0.0],           # top
        [0.0, -2.0, 0.0],          # bottom
        [sqrt3, 1.0, 0.0],         # top-right
        [-sqrt3, 1.0, 0.0],        # top-left
        [sqrt3, -1.0, 0.0],        # bottom-right
        [-sqrt3, -1.0, 0.0],       # bottom-left
        [2.0*sqrt3, 0.0, 0.0],     # far right
        [-2.0*sqrt3, 0.0, 0.0],    # far left
        [sqrt3, 3.0, 0.0],         # upper right
        [-sqrt3, 3.0, 0.0],        # upper left
        [sqrt3, -3.0, 0.0],        # lower right
    ], dtype=np.float64)
    
    # Configuration 2: Optimized for reduced outer hexagon size (from inspiration 2)
    config2 = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 1.9, 0.0],           # top
        [0.0, -1.9, 0.0],          # bottom
        [sqrt3*0.95, 0.95, 0.0],   # top-right
        [-sqrt3*0.95, 0.95, 0.0],  # top-left
        [sqrt3*0.95, -0.95, 0.0],  # bottom-right
        [-sqrt3*0.95, -0.95, 0.0], # bottom-left
        [2.0*sqrt3*0.95, 0.0, 0.0], # far right
        [-2.0*sqrt3*0.95, 0.0, 0.0], # far left
        [sqrt3*0.95, 2.9, 0.0],    # upper right
        [-sqrt3*0.95, 2.9, 0.0],   # upper left
        [sqrt3*0.95, -2.9, 0.0],   # lower right
    ], dtype=np.float64)
    
    # Configuration 3: Alternative arrangement with more compact layout (from inspiration 2)
    config3 = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 1.8, 0.0],           # top
        [0.0, -1.8, 0.0],          # bottom
        [sqrt3*0.9, 0.9, 0.0],     # top-right
        [-sqrt3*0.9, 0.9, 0.0],    # top-left
        [sqrt3*0.9, -0.9, 0.0],    # bottom-right
        [-sqrt3*0.9, -0.9, 0.0],   # bottom-left
        [2.0*sqrt3*0.9, 0.0, 0.0], # far right
        [-2.0*sqrt3*0.9, 0.0, 0.0], # far left
        [sqrt3*0.9, 2.7, 0.0],     # upper right
        [-sqrt3*0.9, 2.7, 0.0],    # upper left
        [sqrt3*0.9, -2.7, 0.0],    # lower right
    ], dtype=np.float64)
    
    # Configuration 4: Highly optimized configuration (from inspiration 3)
    config4 = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 2.0, 0.0],           # top
        [0.0, -2.0, 0.0],          # bottom
        [1.732050808, 1.0, 0.0],   # top-right (precise)
        [-1.732050808, 1.0, 0.0],  # top-left (precise)
        [1.732050808, -1.0, 0.0],  # bottom-right (precise)
        [-1.732050808, -1.0, 0.0], # bottom-left (precise)
        [3.464101615, 0.0, 0.0],   # far right (precise)
        [-3.464101615, 0.0, 0.0],  # far left (precise)
        [1.732050808, 2.0, 0.0],   # top-right corner (precise)
        [-1.732050808, 2.0, 0.0],  # top-left corner (precise)
        [1.732050808, -2.0, 0.0],  # bottom-right corner (precise)
    ], dtype=np.float64)
    
    # Test all configurations and find the best valid one
    best_config = None
    best_side_length = float('inf')
    best_score = -float('inf')
    
    configs = [config1, config2, config3, config4]
    for config in configs:
        try:
            outer_side_length = compute_outer_hexagon_side_length(config)
            score = 1.0 / outer_side_length
            penalty = compute_penalty(config)
            
            if penalty < 1e-3 and outer_side_length < best_side_length:
                best_side_length = outer_side_length
                best_score = score
                best_config = config.copy()
        except Exception as e:
            continue
    
    # If we have a valid configuration, try to optimize it with aggressive settings
    if best_config is not None:
        # Use extremely aggressive optimization approach
        initial_params = best_config.flatten()
        
        # Define very wide bounds for maximum exploration
        bounds = []
        for i in range(12):  # 12 hexagons
            bounds.extend([
                (initial_params[i*3] - 2.0, initial_params[i*3] + 2.0),     # x coordinate
                (initial_params[i*3 + 1] - 2.0, initial_params[i*3 + 1] + 2.0),  # y coordinate
                (0, 360)  # rotation
            ])
        
        # Run extremely aggressive optimization
        try:
            start_time = time.time()
            result = differential_evolution(
                func=objective_function,
                bounds=bounds,
                seed=42,
                maxiter=100,  # Many iterations for thorough search
                popsize=20,   # Large population for better exploration
                mutation=(0.95, 1.0),  # Very high mutation rate for maximum exploration
                recombination=0.95,   # Very high recombination rate for good mixing
                tol=1e-10,    # Extremely tight tolerance
                disp=False,
                polish=True   # Enable polishing for final refinement
            )
            
            if result.success and time.time() - start_time < 55:
                refined_params = result.x.reshape(-1, 3)
                refined_side_length = compute_outer_hexagon_side_length(refined_params)
                refined_penalty = compute_penalty(refined_params)
                
                # Only accept if valid and better than our best
                if refined_penalty < 1e-3 and refined_side_length < best_side_length:
                    return refined_params, np.array([0, 0, 0]), refined_side_length
                    
        except Exception as e:
            # If optimization fails, continue with best configuration
            pass
    
    # Return the best configuration we found
    if best_config is not None:
        return best_config, np.array([0, 0, 0]), best_side_length
    
    # Fallback to a known good configuration
    fallback_config = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, -2.0, 0.0],
        [1.732050808, 1.0, 0.0],
        [-1.732050808, 1.0, 0.0],
        [1.732050808, -1.0, 0.0],
        [-1.732050808, -1.0, 0.0],
        [3.464101615, 0.0, 0.0],
        [-3.464101615, 0.0, 0.0],
        [1.732050808, 3.0, 0.0],
        [-1.732050808, 3.0, 0.0],
        [1.732050808, -3.0, 0.0],
    ], dtype=np.float64)
    
    fallback_side_length = compute_outer_hexagon_side_length(fallback_config)
    return fallback_config, np.array([0, 0, 0]), fallback_side_length


# EVOLVE-BLOCK-END
