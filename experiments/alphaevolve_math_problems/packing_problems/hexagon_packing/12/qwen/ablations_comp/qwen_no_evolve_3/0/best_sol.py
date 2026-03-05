# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from shapely.ops import unary_union

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = 1.0  # distance from center to vertex for unit hexagon
HEXAGON_VERTICES = 6

def get_hexagon_vertices(center_x, center_y, angle_degrees, radius=UNIT_HEX_RADIUS):
    """Get vertices of a regular hexagon given center, angle, and radius"""
    angle_rad = np.radians(angle_degrees)
    vertices = []
    for i in range(HEXAGON_VERTICES):
        theta = angle_rad + i * (2 * np.pi / HEXAGON_VERTICES)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def create_outer_hexagon(side_length, center=(0, 0), angle=0):
    """Create outer hexagon vertices"""
    return get_hexagon_vertices(center[0], center[1], angle, side_length)

def check_containment(inner_hex_vertices, outer_hex_vertices):
    """Check if all inner hexagon vertices are contained within outer hexagon"""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in inner_hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hex_side_length(inner_hex_data, outer_hex_center=(0, 0), outer_hex_angle=0):
    """Calculate minimum outer hexagon side length that contains all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_inner_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(center_x, center_y, angle)
        all_inner_vertices.extend(vertices)
    
    # Convert to numpy array
    all_vertices = np.array(all_inner_vertices)
    
    # Find bounding circle radius
    centroid = np.mean(all_vertices, axis=0)
    distances = np.linalg.norm(all_vertices - centroid, axis=1)
    max_distance = np.max(distances)
    
    # For hexagon, side length = max_distance
    return max_distance

def evaluate_packing(inner_hex_data, outer_hex_center=(0, 0), outer_hex_angle=0):
    """Evaluate a packing configuration"""
    # Calculate required outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data, outer_hex_center, outer_hex_angle)
    
    # Check if all hexagons fit and don't overlap
    valid = True
    total_area = 0
    
    # Create outer hexagon polygon
    outer_vertices = create_outer_hexagon(outer_side_length, outer_hex_center, outer_hex_angle)
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment and overlap
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        inner_vertices = get_hexagon_vertices(center_x, center_y, angle)
        
        # Check containment
        if not check_containment(inner_vertices, outer_vertices):
            valid = False
            break
            
        # Check overlap with others
        for j in range(i+1, len(inner_hex_data)):
            center_x2, center_y2, angle2 = inner_hex_data[j]
            inner_vertices2 = get_hexagon_vertices(center_x2, center_y2, angle2)
            
            if check_overlap(inner_vertices, inner_vertices2):
                valid = False
                break
                
        if not valid:
            break
    
    if valid:
        return 1.0 / outer_side_length  # Return inverse side length
    else:
        return 0.0  # Invalid configuration

def generate_symmetric_configurations():
    """Generate symmetric initial configurations"""
    configs = []
    
    # Configuration 1: 3 layers - center, ring, outer ring
    config1 = np.array([
        [0, 0, 0],           # center
        [0, 2, 0],           # top
        [1.732, 1, 0],       # top-right
        [1.732, -1, 0],      # bottom-right
        [0, -2, 0],          # bottom
        [-1.732, -1, 0],     # bottom-left
        [-1.732, 1, 0],      # top-left
        [3.464, 0, 0],       # far right
        [1.732, 3, 0],       # top far right
        [-1.732, 3, 0],      # top far left
        [-3.464, 0, 0],      # far left
        [-1.732, -3, 0],     # bottom far left
    ])
    configs.append(config1)
    
    # Configuration 2: Hexagonal close packing pattern
    config2 = np.array([
        [0, 0, 0],           # center
        [2, 0, 0],           # right
        [1, 1.732, 0],       # upper right
        [-1, 1.732, 0],      # upper left
        [-2, 0, 0],          # left
        [-1, -1.732, 0],     # lower left
        [1, -1.732, 0],      # lower right
        [3, 1.732, 0],       # upper right far
        [3, -1.732, 0],      # lower right far
        [-3, 1.732, 0],      # upper left far
        [-3, -1.732, 0],     # lower left far
        [0, -3.464, 0],      # bottom far
    ])
    configs.append(config2)
    
    return configs

def optimize_hexagon_packing():
    """Use evolutionary optimization to find better packing"""
    best_score = 0.0
    best_config = None
    best_outer_length = float('inf')
    
    # Try different initial configurations
    initial_configs = generate_symmetric_configurations()
    
    # Use a more sophisticated optimization approach
    # Define bounds for optimization: [x, y, angle] for each of 12 hexagons + outer hex parameters
    bounds = []
    # For inner hexagons: x, y, angle each with reasonable ranges
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
    
    # For outer hexagon: center and angle
    bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    
    def objective_function(params):
        # Parse parameters
        inner_params = params[:36].reshape(12, 3)
        outer_center_x, outer_center_y, outer_angle = params[36:]
        
        # Create configuration
        config = np.copy(inner_params)
        config[:, 0] += outer_center_x  # Adjust inner positions relative to outer center
        config[:, 1] += outer_center_y
        
        # Evaluate
        score = evaluate_packing(config, (outer_center_x, outer_center_y), outer_angle)
        return -score  # Negative because we want to maximize
    
    # Try several random starting points
    for _ in range(10):
        # Random initialization
        start_params = []
        for _ in range(12):
            start_params.extend([np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(0, 360)])
        start_params.extend([0, 0, 0])  # Outer hex center and angle
        
        try:
            # Simple local search around good initial configs
            for initial_config in initial_configs:
                # Perturb initial config slightly
                perturbed = initial_config.copy().astype(float)
                perturbed[:, 0] += np.random.normal(0, 0.5, 12)
                perturbed[:, 1] += np.random.normal(0, 0.5, 12)
                perturbed[:, 2] += np.random.normal(0, 30, 12)
                
                # Evaluate this configuration
                score = evaluate_packing(perturbed, (0, 0), 0)
                if score > best_score:
                    best_score = score
                    best_config = perturbed.copy()
                    best_outer_length = 1.0 / score
        except Exception as e:
            continue
    
    # If we still haven't found something good, try a simpler approach
    if best_config is None:
        # Start with a known good symmetric pattern
        best_config = np.array([
            [0, 0, 0],           # center
            [0, 2, 0],           # top
            [1.732, 1, 0],       # top-right
            [1.732, -1, 0],      # bottom-right
            [0, -2, 0],          # bottom
            [-1.732, -1, 0],     # bottom-left
            [-1.732, 1, 0],      # top-left
            [3.464, 0, 0],       # far right
            [1.732, 3, 0],       # top far right
            [-1.732, 3, 0],      # top far left
            [-3.464, 0, 0],      # far left
            [-1.732, -3, 0],     # bottom far left
        ])
        best_score = evaluate_packing(best_config, (0, 0), 0)
        best_outer_length = 1.0 / best_score if best_score > 0 else float('inf')
    
    return best_config, (0, 0, 0), best_outer_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a symmetry-aware evolutionary approach for better results.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use optimization approach
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    
    # Ensure the final result is properly formatted
    # The outer hexagon is centered at origin with some angle
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
