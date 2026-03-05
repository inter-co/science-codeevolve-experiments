# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def hexagon_vertices(center_x, center_y, side_length=1, rotation_degrees=0):
    """Generate vertices of a regular hexagon."""
    angle = math.radians(rotation_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Vertices of unit hexagon centered at origin
    hex_points = []
    for i in range(6):
        theta = math.pi/3 * i
        x = math.cos(theta)
        y = math.sin(theta)
        hex_points.append((x, y))
    
    # Rotate and translate
    rotated_points = []
    for x, y in hex_points:
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        rotated_points.append((center_x + x_rot, center_y + y_rot))
    
    return np.array(rotated_points)

def check_containment(hex_vertices, outer_center_x, outer_center_y, outer_side_length):
    """Check if all vertices of hexagon are inside outer hexagon."""
    # Create outer hexagon vertices
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_side_length, 0)
    
    # Check if each vertex of inner hex is inside outer hex
    for vertex in hex_vertices:
        x, y = vertex
        # Use point-in-polygon test
        # For a regular hexagon, we can check distance from center
        dx = x - outer_center_x
        dy = y - outer_center_y
        distance = math.sqrt(dx*dx + dy*dy)
        # Max distance from center for containment in hexagon with side length s is s
        if distance >= outer_side_length:
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using separating axis theorem."""
    # Get all edges of both hexagons
    edges1 = []
    edges2 = []
    
    for i in range(6):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i+1)%6]
        edge = p2 - p1
        edges1.append(edge)
        
        p1 = hex2_vertices[i]
        p2 = hex2_vertices[(i+1)%6]
        edge = p2 - p1
        edges2.append(edge)
    
    # Check separation axes (normals to edges)
    all_edges = edges1 + edges2
    for edge in all_edges:
        # Normal to edge
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Project both polygons onto normal
        proj1 = [np.dot(vertex, normal) for vertex in hex1_vertices]
        proj2 = [np.dot(vertex, normal) for vertex in hex2_vertices]
        
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        
        # Check if projections overlap
        if max1 < min2 or max2 < min1:
            return False  # No overlap
    
    return True  # Overlap exists

def calculate_outer_hex_radius(inner_positions, side_length=1):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons."""
    max_distance = 0
    
    # For each inner hexagon, get all vertices and find maximum distance from center
    for x, y, _ in inner_positions:
        vertices = hexagon_vertices(x, y, side_length, 0)
        for vertex in vertices:
            dist = math.sqrt((vertex[0])**2 + (vertex[1])**2)
            max_distance = max(max_distance, dist)
    
    # Add buffer to ensure containment
    # For a hexagon with side length r, the distance from center to vertex is r
    # So we need radius = max_distance + 1 (buffer for unit hexagon)
    return max_distance + 1.0

def evaluate_arrangement(positions, outer_center=(0, 0)):
    """Evaluate if arrangement is valid and return penalty if invalid."""
    n = len(positions)
    
    # Calculate outer hexagon side length needed
    outer_side_length = calculate_outer_hex_radius(positions)
    
    # Check all pairwise overlaps
    valid = True
    for i in range(n):
        hex1_vertices = hexagon_vertices(positions[i][0], positions[i][1], 1, positions[i][2])
        
        # Check containment
        if not check_containment(hex1_vertices, outer_center[0], outer_center[1], outer_side_length):
            valid = False
            break
            
        for j in range(i+1, n):
            hex2_vertices = hexagon_vertices(positions[j][0], positions[j][1], 1, positions[j][2])
            
            if check_overlap(hex1_vertices, hex2_vertices):
                valid = False
                break
                
        if not valid:
            break
    
    if valid:
        return -1.0 / outer_side_length  # Negative because we minimize
    else:
        # Return large penalty for invalid configurations
        return -1000.0 / outer_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a more sophisticated approach than simple grid placement.
    """
    # Initial guess - try a pattern that might be more efficient
    # Based on known optimal arrangements for hexagonal packing
    initial_positions = [
        [0, 0, 0],           # center
        [0, 2.0, 0],         # top
        [0, -2.0, 0],        # bottom  
        [1.732, 1.0, 0],     # top-right (sqrt(3) = 1.732)
        [-1.732, 1.0, 0],    # top-left
        [1.732, -1.0, 0],    # bottom-right
        [-1.732, -1.0, 0],   # bottom-left
        [3.464, 2.0, 0],     # far top-right
        [-3.464, 2.0, 0],    # far top-left
        [3.464, -2.0, 0],    # far bottom-right
        [-3.464, -2.0, 0],   # far bottom-left
    ]
    
    # Convert to numpy array
    inner_hex_data = np.array(initial_positions)
    
    # Calculate outer hexagon size
    outer_side_length = calculate_outer_hex_radius(initial_positions)
    
    # Refine using optimization
    def objective(params):
        # Reshape params back into positions
        positions = []
        for i in range(11):
            positions.append([params[i*3], params[i*3+1], params[i*3+2]])
        return evaluate_arrangement(positions)
    
    # Flatten initial positions for optimization
    initial_flat = []
    for pos in initial_positions:
        initial_flat.extend(pos)
    
    # Optimize using scipy minimize
    try:
        result = minimize(objective, initial_flat, method='L-BFGS-B', 
                         bounds=[(-10, 10), (-10, 10), (0, 360)] * 11,
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        # Extract optimized positions
        optimized_positions = []
        for i in range(11):
            optimized_positions.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
            
        # Recalculate final outer hexagon size
        outer_side_length = calculate_outer_hex_radius(optimized_positions)
        
        # Convert back to desired format
        inner_hex_data = np.array(optimized_positions)
        outer_hex_data = np.array([0, 0, 0])  # centered
        
        return inner_hex_data, outer_hex_data, outer_side_length
        
    except Exception as e:
        # If optimization fails, return the initial attempt
        outer_hex_data = np.array([0, 0, 0])  # centered
        return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
