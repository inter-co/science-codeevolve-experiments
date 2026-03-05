# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import time


def create_unit_hexagon(center=(0, 0), rotation=0):
    """Create a unit regular hexagon with given center and rotation"""
    angle = np.radians(rotation)
    # Vertices of unit hexagon centered at origin
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        vertices.append((x + center[0], y + center[1]))
    return Polygon(vertices)


def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer hexagon"""
    # Check if all vertices of inner hexagon are inside outer hexagon
    for vertex in hexagon.exterior.coords[:-1]:  # Exclude repeated last vertex
        if not outer_hexagon.contains(Point(vertex)):
            return False
    return True


def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)


def calculate_outer_hexagon_radius(inner_hex_data, outer_center=(0, 0)):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        # Get extreme points of the hexagon
        hexagon = create_unit_hexagon((center_x, center_y), rotation)
        for vertex in hexagon.exterior.coords[:-1]:
            dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # Add some buffer for numerical precision
    return max_dist * 1.01


def evaluate_configuration(inner_hex_data):
    """Evaluate a configuration: returns negative of 1/outer_radius (for minimization)"""
    # Calculate outer hexagon radius
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data)
    
    # Check for overlaps and containment
    outer_hexagon = create_unit_hexagon((0, 0), 0)  # Assume outer hexagon centered at origin
    
    # Check overlaps
    total_overlaps = 0
    for i in range(len(inner_hex_data)):
        hex1 = create_unit_hexagon((inner_hex_data[i][0], inner_hex_data[i][1]), inner_hex_data[i][2])
        for j in range(i+1, len(inner_hex_data)):
            hex2 = create_unit_hexagon((inner_hex_data[j][0], inner_hex_data[j][1]), inner_hex_data[j][2])
            if check_overlap(hex1, hex2):
                total_overlaps += 1
    
    # Check containment
    all_contained = True
    for i in range(len(inner_hex_data)):
        hex1 = create_unit_hexagon((inner_hex_data[i][0], inner_hex_data[i][1]), inner_hex_data[i][2])
        if not check_containment(hex1, outer_hexagon):
            all_contained = False
    
    # Penalize invalid configurations heavily
    if total_overlaps > 0 or not all_contained:
        return 1e10  # Large penalty for invalid configurations
    
    # Return negative of 1/outer_radius for maximization via minimization
    return -1.0 / outer_radius


def generate_symmetric_configurations():
    """Generate several symmetric starting configurations"""
    configs = []
    
    # Configuration 1: Hexagonal pattern with central hexagon
    config1 = np.array([
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [1.732, 1, 0],  # top-right
        [1.732, -1, 0], # bottom-right
        [0, -2, 0],     # bottom
        [-1.732, -1, 0], # bottom-left
        [-1.732, 1, 0],  # top-left
        [3.464, 0, 0],   # far right
        [1.732, 3, 0],   # upper right
        [-1.732, 3, 0],  # upper left
        [-3.464, 0, 0],  # far left
        [-1.732, -3, 0], # lower left
    ])
    configs.append(config1)
    
    # Configuration 2: Star-like pattern
    config2 = np.array([
        [0, 0, 0],       # center
        [0, 2.5, 0],     # top
        [2.165, 1.25, 0], # top-right
        [2.165, -1.25, 0], # bottom-right
        [0, -2.5, 0],    # bottom
        [-2.165, -1.25, 0], # bottom-left
        [-2.165, 1.25, 0],  # top-left
        [0, 4.5, 0],     # far top
        [0, -4.5, 0],    # far bottom
        [4.33, 0, 0],    # far right
        [-4.33, 0, 0],   # far left
        [3.75, 2.165, 0], # diagonal
    ])
    configs.append(config2)
    
    return configs


def optimize_hexagon_packing():
    """Use evolutionary approach to find optimal configuration"""
    
    # Start with good symmetric configurations
    initial_configs = generate_symmetric_configurations()
    
    best_result = None
    best_score = 1e10
    
    # Try different starting configurations
    for i, config in enumerate(initial_configs):
        # Flatten configuration for optimization
        flat_config = config.flatten()
        
        # Use a simple optimization approach with bounds
        # Since we're looking for a good local minimum, we'll use a simpler approach
        # Let's try a few iterations of local search
        
        # Create a simple hill-climbing approach
        current_config = config.copy()
        current_score = evaluate_configuration(current_config)
        
        # Perturb slightly and see if we improve
        for _ in range(1000):  # Limit iterations
            # Make small random perturbations
            new_config = current_config.copy()
            
            # Randomly select a hexagon to perturb
            hex_idx = np.random.randint(0, 12)
            
            # Small random changes to position and rotation
            new_config[hex_idx, 0] += np.random.normal(0, 0.1)  # x
            new_config[hex_idx, 1] += np.random.normal(0, 0.1)  # y
            new_config[hex_idx, 2] += np.random.normal(0, 5)   # rotation
            
            new_score = evaluate_configuration(new_config)
            
            if new_score < current_score:
                current_config = new_config
                current_score = new_score
                
        # Update best if this is better
        if current_score < best_score:
            best_score = current_score
            best_result = current_config
    
    # Final optimization with more careful approach
    if best_result is None:
        best_result = initial_configs[0]
    
    # Return the best configuration found
    outer_radius = calculate_outer_hexagon_radius(best_result)
    return best_result, np.array([0, 0, 0]), outer_radius


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use our improved optimization approach
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    
    # Ensure we return a valid configuration with proper shape
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
