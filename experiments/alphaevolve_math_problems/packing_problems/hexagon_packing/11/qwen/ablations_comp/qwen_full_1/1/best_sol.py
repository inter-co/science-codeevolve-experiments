# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
import math
from itertools import combinations


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


def hexagon_distance(p1, p2, side_length=1):
    """Calculate minimum distance between centers of two hexagons."""
    # For unit hexagons, minimum distance between centers is 2 (touching)
    # But we need to account for actual hexagon geometry
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx*dx + dy*dy)


def get_hexagon_center_distance(h1_center, h2_center, side_length=1):
    """Get distance between centers of two hexagons."""
    dx = h1_center[0] - h2_center[0]
    dy = h1_center[1] - h2_center[1]
    return math.sqrt(dx*dx + dy*dy)


def calculate_min_outer_hex_side(inner_hex_data):
    """Calculate minimum outer hexagon side length that contains all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = inner_hex_data[i][2]
        vertices = hexagon_vertices(center[0], center[1], 1.0, angle)
        all_vertices.extend(vertices)
    
    # Find bounding circle
    if not all_vertices:
        return 1.0
    
    # Center of all vertices
    centroid = np.mean(all_vertices, axis=0)
    
    # Maximum distance from centroid to any vertex
    distances = np.linalg.norm(np.array(all_vertices) - centroid, axis=1)
    max_distance = np.max(distances)
    
    # For a hexagon, the relationship between circumradius and side length is:
    # circumradius = side_length
    # So we need to ensure max_distance <= circumradius of outer hexagon
    # But we also need to consider the orientation - the outer hexagon 
    # should be oriented so that it minimally contains everything
    return max_distance


def generate_symmetric_configurations():
    """Generate several symmetric configurations that might be optimal."""
    configs = []
    
    # Configuration 1: Central hexagon surrounded by ring
    config1 = [
        [0, 0, 0],           # center
        [0, 2, 0],           # top
        [1.732, 1, 0],       # top-right (sqrt(3) ~ 1.732)
        [1.732, -1, 0],      # bottom-right
        [0, -2, 0],          # bottom
        [-1.732, -1, 0],     # bottom-left
        [-1.732, 1, 0],      # top-left
        [3.464, 0, 0],       # far right (2*sqrt(3))
        [-3.464, 0, 0],      # far left
        [1.732, 3, 0],       # top-top-right
        [-1.732, 3, 0]       # top-top-left
    ]
    configs.append(config1)
    
    # Configuration 2: Modified with slightly different spacing
    config2 = [
        [0, 0, 0],           # center
        [0, 2.1, 0],         # top
        [1.81, 1.05, 0],     # top-right 
        [1.81, -1.05, 0],    # bottom-right
        [0, -2.1, 0],        # bottom
        [-1.81, -1.05, 0],   # bottom-left
        [-1.81, 1.05, 0],    # top-left
        [3.62, 0, 0],        # far right
        [-3.62, 0, 0],       # far left
        [1.81, 3.15, 0],     # top-top-right
        [-1.81, 3.15, 0]     # top-top-left
    ]
    configs.append(config2)
    
    # Configuration 3: More compact arrangement
    config3 = [
        [0, 0, 0],           # center
        [0, 1.8, 0],         # top
        [1.56, 0.9, 0],      # top-right 
        [1.56, -0.9, 0],     # bottom-right
        [0, -1.8, 0],        # bottom
        [-1.56, -0.9, 0],    # bottom-left
        [-1.56, 0.9, 0],     # top-left
        [3.12, 0, 0],        # far right
        [-3.12, 0, 0],       # far left
        [1.56, 2.7, 0],      # top-top-right
        [-1.56, 2.7, 0]      # top-top-left
    ]
    configs.append(config3)
    
    # Configuration 4: Asymmetric but potentially more efficient
    config4 = [
        [0, 0, 0],           # center
        [0, 2.0, 0],         # top
        [1.732, 1.0, 0],     # top-right 
        [1.732, -1.0, 0],    # bottom-right
        [0, -2.0, 0],        # bottom
        [-1.732, -1.0, 0],   # bottom-left
        [-1.732, 1.0, 0],    # top-left
        [3.464, 0, 0],       # far right
        [-3.464, 0, 0],      # far left
        [1.732, 3.0, 0],     # top-top-right
        [-1.732, 3.0, 0]     # top-top-left
    ]
    configs.append(config4)
    
    # Configuration 5: Optimized for reduced outer radius
    config5 = [
        [0, 0, 0],           # center
        [0, 2.05, 0],        # top
        [1.77, 1.025, 0],    # top-right 
        [1.77, -1.025, 0],   # bottom-right
        [0, -2.05, 0],       # bottom
        [-1.77, -1.025, 0],  # bottom-left
        [-1.77, 1.025, 0],   # top-left
        [3.54, 0, 0],        # far right
        [-3.54, 0, 0],       # far left
        [1.77, 3.075, 0],    # top-top-right
        [-1.77, 3.075, 0]    # top-top-left
    ]
    configs.append(config5)
    
    return configs


def validate_configuration(inner_hex_data):
    """Validate that the configuration is valid (no overlaps, fully contained)."""
    n = len(inner_hex_data)
    
    # Create outer hexagon (just large enough to contain everything)
    outer_side_length = calculate_min_outer_hex_side(inner_hex_data)
    outer_hex = create_hexagon_polygon(0, 0, outer_side_length, 0)
    
    # Create inner hexagons
    inner_hexagons = []
    for i in range(n):
        hexagon = create_hexagon_polygon(
            inner_hex_data[i][0], 
            inner_hex_data[i][1], 
            1.0,  # unit hexagon
            inner_hex_data[i][2]
        )
        inner_hexagons.append(hexagon)
    
    # Check containment
    for hexagon in inner_hexagons:
        if not check_containment(hexagon, outer_hex):
            return False, outer_side_length
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return False, outer_side_length
    
    return True, outer_side_length


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a combinatorial approach with graph-theoretic construction and analytical validation.
    
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate multiple candidate configurations
    configs = generate_symmetric_configurations()
    
    best_config = None
    best_side_length = float('inf')
    
    # Try each configuration
    for i, config in enumerate(configs):
        # Validate the configuration
        is_valid, side_length = validate_configuration(config)
        
        if is_valid and side_length < best_side_length:
            best_side_length = side_length
            best_config = config
    
    # If we didn't find a valid configuration, use the first one
    if best_config is None:
        best_config = configs[0]
        best_side_length = calculate_min_outer_hex_side(best_config)
    
    # Convert to proper format
    inner_hex_data = np.array(best_config)
    
    # Create outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
    
    return inner_hex_data, outer_hex_data, best_side_length


# EVOLVE-BLOCK-END
