# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def create_regular_hexagon_vertices(center=(0,0), radius=1, rotation=0):
    """Create vertices of a regular hexagon"""
    vertices = []
    for i in range(6):
        angle = rotation + i * math.pi / 3
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def hexagon_contains_point(hex_center, hex_radius, point):
    """Check if a point is inside a hexagon using distance to center"""
    dist_to_center = math.sqrt((point[0] - hex_center[0])**2 + (point[1] - hex_center[1])**2)
    return dist_to_center <= hex_radius

def hexagon_intersects_hexagon(hex1_center, hex1_radius, hex2_center, hex2_radius):
    """Simple check if two hexagons might intersect"""
    dist_centers = math.sqrt((hex1_center[0] - hex2_center[0])**2 + (hex1_center[1] - hex2_center[1])**2)
    return dist_centers <= 2 * hex1_radius

def get_hexagon_vertices(hex_center, hex_radius, rotation):
    """Get vertices of a hexagon with given parameters"""
    vertices = []
    for i in range(6):
        angle = rotation + i * math.pi / 3
        x = hex_center[0] + hex_radius * math.cos(angle)
        y = hex_center[1] + hex_radius * math.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_hexagon(point, hex_vertices):
    """Check if point is inside hexagon using ray casting"""
    x, y = point
    n = len(hex_vertices)
    inside = False
    
    p1x, p1y = hex_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = hex_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def compute_outer_hexagon_radius(inner_hex_data, outer_center=(0,0)):
    """Compute minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = math.radians(inner_hex_data[i][2])
        vertices = get_hexagon_vertices(center, 1, rotation)
        
        # Check distance from center to each vertex
        for vertex in vertices:
            dist = math.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # Add small buffer to ensure containment
    return max_dist * 1.01

def check_overlap_and_containment(inner_hex_data, outer_radius):
    """Check if inner hexagons overlap or are outside outer hexagon"""
    n = len(inner_hex_data)
    
    # Check containment - all vertices of each hexagon must be inside outer hexagon
    outer_vertices = get_hexagon_vertices((0,0), outer_radius, 0)
    
    for i in range(n):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = math.radians(inner_hex_data[i][2])
        vertices = get_hexagon_vertices(center, 1, rotation)
        
        # Check if all vertices are inside outer hexagon
        for vertex in vertices:
            if not point_in_hexagon(vertex, outer_vertices):
                return False, "Outside bounds"
        
        # Check overlap with other hexagons
        for j in range(i+1, n):
            center2 = (inner_hex_data[j][0], inner_hex_data[j][1])
            rotation2 = math.radians(inner_hex_data[j][2])
            vertices2 = get_hexagon_vertices(center2, 1, rotation2)
            
            # Quick bounding box check first
            min_x1, max_x1 = min(v[0] for v in vertices), max(v[0] for v in vertices)
            min_y1, max_y1 = min(v[1] for v in vertices), max(v[1] for v in vertices)
            min_x2, max_x2 = min(v[0] for v in vertices2), max(v[0] for v in vertices2)
            min_y2, max_y2 = min(v[1] for v in vertices2), max(v[1] for v in vertices2)
            
            if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                continue  # No overlap in bounding boxes
            
            # More precise overlap check
            # For now we'll use a simplified approach - if centers too close, there's likely overlap
            dist_centers = math.sqrt((center[0] - center2[0])**2 + (center[1] - center2[1])**2)
            if dist_centers < 2:  # Hexagons are close enough to potentially overlap
                # Check actual vertex intersections (simplified)
                for v1 in vertices:
                    for v2 in vertices2:
                        if math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2) < 0.1:
                            return False, "Overlap detected"
    
    return True, "Valid"

def objective_function(params):
    """Objective function to minimize (negative of 1/outer_radius)"""
    # params: [x1, y1, theta1, x2, y2, theta2, ..., x11, y11, theta11, outer_radius]
    n = 11
    inner_params = params[:3*n]
    outer_radius = params[3*n]
    
    inner_hex_data = np.array([
        [inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]] 
        for i in range(n)
    ])
    
    valid, message = check_overlap_and_containment(inner_hex_data, outer_radius)
    
    if not valid:
        # Penalize invalid configurations heavily
        return 1e10
    
    # Return negative of 1/outer_radius to minimize
    return -1.0 / outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    
    # Start with a better initial configuration based on known hexagonal packings
    # Using a central hexagon surrounded by layers
    initial_positions = [
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [0, -2, 0],     # bottom  
        [1.732, 1, 0],  # top-right
        [-1.732, 1, 0], # top-left
        [1.732, -1, 0], # bottom-right
        [-1.732, -1, 0],# bottom-left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [1.732, 3, 0],  # upper-right
        [-1.732, 3, 0], # upper-left
    ]
    
    # Convert to numpy array
    inner_hex_data = np.array(initial_positions)
    
    # Estimate initial outer radius
    outer_radius = compute_outer_hexagon_radius(inner_hex_data)
    
    # Flatten parameters for optimization
    params = []
    for i in range(n):
        params.extend([inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2]])
    params.append(outer_radius)
    
    # Use optimization to improve the configuration
    # Define bounds for optimization
    bounds = []
    # Positions: x, y in range [-10, 10]
    for _ in range(n):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    # Outer radius: positive value
    bounds.append((1.0, 20.0))
    
    try:
        # Optimize using scipy minimize
        result = minimize(objective_function, params, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 1000, 'ftol': 1e-8})
        
        if result.success:
            final_params = result.x
            # Extract optimized parameters
            inner_hex_data_opt = np.array([
                [final_params[3*i], final_params[3*i+1], final_params[3*i+2]] 
                for i in range(n)
            ])
            outer_radius_opt = final_params[3*n]
            
            # Verify the final configuration
            valid, message = check_overlap_and_containment(inner_hex_data_opt, outer_radius_opt)
            if valid:
                inner_hex_data = inner_hex_data_opt
                outer_radius = outer_radius_opt
            else:
                # Fall back to initial configuration if optimization failed
                pass
    except Exception as e:
        # If optimization fails, keep the initial configuration
        pass
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END
