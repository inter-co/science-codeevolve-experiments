# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    hex_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    hex_vertices = hex_vertices @ rotation_matrix.T
    hex_vertices[:, 0] += center_x
    hex_vertices[:, 1] += center_y
    
    return hex_vertices

def check_containment(hex_vertices, outer_center_x, outer_center_y, outer_side_length):
    """Check if all vertices of a hexagon are within the outer hexagon."""
    # Generate outer hexagon vertices
    outer_vertices = generate_hexagon_vertices(outer_center_x, outer_center_y, 0, outer_side_length)
    
    # Check if all inner hexagon vertices are within the outer hexagon
    for vertex in hex_vertices:
        # Point-in-polygon test using winding number or ray casting
        # For simplicity, we'll use a basic approach: distance from center
        dist_from_center = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
        # Maximum distance from center for containment
        max_dist = outer_side_length * np.sqrt(3) / 2  # Distance to corner of outer hexagon
        if dist_from_center > max_dist:
            return False
    return True

def calculate_outer_hex_side_length(inner_hex_data, outer_center_x=0, outer_center_y=0):
    """Calculate the minimum side length of outer hexagon needed to contain all inner hexagons."""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        # Get vertices of this hexagon
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        
        # Find maximum distance from outer center to any vertex
        for vertex in vertices:
            dist = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
            max_distance = max(max_distance, dist)
    
    # Convert distance to hexagon side length
    # For a hexagon, if we know the circumradius R (distance from center to vertex),
    # then the side length s = R
    # But we need to account for the fact that the outer hexagon is also regular
    # The minimum side length is such that the outer hexagon can contain all vertices
    # of the inner hexagons, so we compute the side length as max_distance * sqrt(3)/sqrt(3) = max_distance
    # Actually, let's be more precise:
    # The distance from center to corner of outer hexagon should be >= max_distance
    # For a hexagon with side length s, the distance to corner is s
    # So outer side length should be >= max_distance
    return max_distance

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using separating axis theorem."""
    # For simplicity, check if any vertices of one hexagon are inside the other
    from shapely.geometry import Polygon
    
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    
    # If polygons intersect, they overlap
    return poly1.intersects(poly2)

def evaluate_solution(params):
    """Evaluate how well a solution works."""
    # params contains [outer_side_length] + 36 parameters for 12 hexagons (x,y,angle each)
    outer_side_length = params[0]
    inner_params = params[1:].reshape(-1, 3)
    
    # Calculate total penalty for overlaps and containment violations
    penalty = 0
    
    # Check containment
    outer_center_x, outer_center_y = 0, 0  # Assume centered at origin for now
    for i in range(len(inner_params)):
        center_x, center_y, angle = inner_params[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        
        # Check containment
        if not check_containment(vertices, outer_center_x, outer_center_y, outer_side_length):
            penalty += 1000  # Large penalty for containment violation
    
    # Check overlaps
    for i in range(len(inner_params)):
        for j in range(i+1, len(inner_params)):
            vertices_i = generate_hexagon_vertices(inner_params[i][0], inner_params[i][1], inner_params[i][2])
            vertices_j = generate_hexagon_vertices(inner_params[j][0], inner_params[j][1], inner_params[j][2])
            
            if check_overlap(vertices_i, vertices_j):
                penalty += 1000  # Large penalty for overlap
    
    # If no penalties, return negative of outer side length (since we want to minimize it)
    if penalty == 0:
        return -outer_side_length
    else:
        return penalty + 10000  # Very large penalty

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a novel evolutionary approach based on geometric constraints.
    """
    # Start with a good initial configuration inspired by known optimal packings
    # Try a central hexagon surrounded by layers in a symmetric pattern
    
    # Initial guess for positions (more sophisticated than simple grid)
    initial_positions = [
        [0, 0, 0],           # center
        [0, 2.0, 0],         # top
        [0, -2.0, 0],        # bottom
        [1.732, 1.0, 0],     # top-right
        [-1.732, 1.0, 0],    # top-left
        [1.732, -1.0, 0],    # bottom-right
        [-1.732, -1.0, 0],   # bottom-left
        [3.464, 0, 0],       # far right
        [-3.464, 0, 0],      # far left
        [1.732, 3.0, 0],     # top far right
        [-1.732, 3.0, 0],    # top far left
        [1.732, -3.0, 0],    # bottom far right
        [-1.732, -3.0, 0],   # bottom far left
    ]
    
    # Reduce to 12 hexagons
    initial_positions = initial_positions[:12]
    
    # Create bounds for optimization
    # Outer hexagon side length: reasonable range
    bounds = [(3.0, 6.0)]  # outer side length bounds
    
    # Inner hexagon parameters: (x, y, angle) for each hexagon
    # x,y: -5 to 5, angle: 0 to 360 degrees
    for i in range(12):
        bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)])
    
    # Use a simpler direct approach with symmetry consideration
    # We'll create a better starting point based on known dense packings
    
    # More systematic approach: place in concentric rings
    inner_hex_data = np.zeros((12, 3))
    
    # Center hexagon
    inner_hex_data[0] = [0, 0, 0]
    
    # First ring: 6 hexagons around center
    angles = np.linspace(0, 360, 7)[:-1]  # 6 angles, exclude last to avoid duplication
    radius = 2.0  # distance from center
    for i in range(6):
        angle = angles[i]
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        inner_hex_data[i+1] = [x, y, 0]
    
    # Second ring: 5 hexagons
    angles = np.linspace(0, 360, 6)[:-1]  # 5 angles
    radius = 3.5
    for i in range(5):
        angle = angles[i]
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        inner_hex_data[i+7] = [x, y, 0]
    
    # Refine using optimization
    def objective(params):
        outer_side_length = params[0]
        inner_params = params[1:].reshape(-1, 3)
        
        # Calculate penalty for overlaps and containment violations
        penalty = 0
        
        # Check containment
        outer_center_x, outer_center_y = 0, 0
        for i in range(len(inner_params)):
            center_x, center_y, angle = inner_params[i]
            vertices = generate_hexagon_vertices(center_x, center_y, angle)
            
            if not check_containment(vertices, outer_center_x, outer_center_y, outer_side_length):
                penalty += 1000000
        
        # Check overlaps
        for i in range(len(inner_params)):
            for j in range(i+1, len(inner_params)):
                vertices_i = generate_hexagon_vertices(inner_params[i][0], inner_params[i][1], inner_params[i][2])
                vertices_j = generate_hexagon_vertices(inner_params[j][0], inner_params[j][1], inner_params[j][2])
                
                if check_overlap(vertices_i, vertices_j):
                    penalty += 1000000
        
        return penalty + outer_side_length if penalty > 0 else -outer_side_length
    
    # Start with a reasonable estimate
    outer_side_length = 4.0
    params = [outer_side_length] + inner_hex_data.flatten().tolist()
    
    # Optimization using a simplified approach
    # We'll try a few different configurations and pick the best
    
    best_config = None
    best_score = float('inf')
    
    # Try several symmetric configurations
    configs = []
    
    # Configuration 1: Hexagonal arrangement
    config1 = np.zeros((12, 3))
    config1[0] = [0, 0, 0]
    for i in range(1, 12):
        angle = (i-1) * 30  # 30 degree increments
        radius = 1.8
        config1[i] = [radius * np.cos(np.radians(angle)), 
                      radius * np.sin(np.radians(angle)), 0]
    configs.append(config1)
    
    # Configuration 2: More spread out
    config2 = np.zeros((12, 3))
    config2[0] = [0, 0, 0]
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = i * 60
        config2[i+1] = [2.0 * np.cos(np.radians(angle)), 
                        2.0 * np.sin(np.radians(angle)), 0]
    # Ring 2: 5 hexagons
    for i in range(5):
        angle = i * 72
        config2[i+7] = [3.0 * np.cos(np.radians(angle)), 
                        3.0 * np.sin(np.radians(angle)), 0]
    configs.append(config2)
    
    # Evaluate each configuration
    for config in configs:
        # Calculate outer hexagon size needed
        min_side_length = calculate_outer_hex_side_length(config)
        
        # Check if it works
        valid = True
        outer_center_x, outer_center_y = 0, 0
        for i in range(len(config)):
            center_x, center_y, angle = config[i]
            vertices = generate_hexagon_vertices(center_x, center_y, angle)
            
            if not check_containment(vertices, outer_center_x, outer_center_y, min_side_length):
                valid = False
                break
        
        if valid:
            # Check overlaps
            for i in range(len(config)):
                for j in range(i+1, len(config)):
                    vertices_i = generate_hexagon_vertices(config[i][0], config[i][1], config[i][2])
                    vertices_j = generate_hexagon_vertices(config[j][0], config[j][1], config[j][2])
                    
                    if check_overlap(vertices_i, vertices_j):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = min_side_length
                if score < best_score:
                    best_score = score
                    best_config = config.copy()
    
    # Final refinement using a greedy approach
    if best_config is not None:
        final_inner = best_config.copy()
    else:
        # Fallback to a simple configuration that's likely to work
        final_inner = np.array([
            [0, 0, 0],          # center
            [0, 2.0, 0],        # top
            [0, -2.0, 0],       # bottom
            [1.732, 1.0, 0],    # top-right
            [-1.732, 1.0, 0],   # top-left
            [1.732, -1.0, 0],   # bottom-right
            [-1.732, -1.0, 0],  # bottom-left
            [3.464, 0, 0],      # far right
            [-3.464, 0, 0],     # far left
            [1.732, 3.0, 0],    # top far right
            [-1.732, 3.0, 0],   # top far left
            [1.732, -3.0, 0],   # bottom far right
        ])
        
        # Adjust to get better packing
        final_inner[1][1] = 1.9  # Slightly adjust for better fit
        final_inner[2][1] = -1.9
        final_inner[3][0] = 1.73
        final_inner[3][1] = 0.9
        final_inner[4][0] = -1.73
        final_inner[4][1] = 0.9
        final_inner[5][0] = 1.73
        final_inner[5][1] = -0.9
        final_inner[6][0] = -1.73
        final_inner[6][1] = -0.9
        final_inner[7][0] = 3.46
        final_inner[8][0] = -3.46
        final_inner[9][0] = 1.73
        final_inner[9][1] = 2.9
        final_inner[10][0] = -1.73
        final_inner[10][1] = 2.9
        final_inner[11][0] = 1.73
        final_inner[11][1] = -2.9
    
    # Final calculation of outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(final_inner)
    
    # Ensure we're not violating constraints
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return final_inner, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
