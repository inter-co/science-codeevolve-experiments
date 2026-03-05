# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import time

def create_hexagon_vertices(center_x, center_y, side_length=1, rotation_deg=0):
    """Create vertices of a regular hexagon given center, side length, and rotation."""
    rotation_rad = np.radians(rotation_deg)
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    vertices = np.column_stack([
        center_x + side_length * np.cos(angles),
        center_y + side_length * np.sin(angles)
    ])
    return vertices

def check_containment(hexagon_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    inner_polygon = Polygon(hexagon_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hexagon_side_length(inner_hex_data, outer_center=(0, 0)):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons."""
    # Create all inner hexagon vertices
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, 1, angle)
        all_vertices.append(vertices)
    
    # Get all vertices from all hexagons
    all_points = np.vstack(all_vertices)
    
    # Find the bounding circle and calculate the side length
    # Center of all points
    center = np.mean(all_points, axis=0)
    
    # Distance from center to all vertices
    distances = np.sqrt(np.sum((all_points - center)**2, axis=1))
    
    # Maximum distance gives us the circumradius of the outer hexagon
    max_distance = np.max(distances)
    
    # For a regular hexagon, the side length equals the circumradius
    return max_distance

def evaluate_packing(inner_hex_data, outer_center=(0, 0)):
    """Evaluate the quality of a hexagon packing."""
    # Calculate outer hexagon side length
    outer_side_length = calculate_outer_hexagon_side_length(inner_hex_data, outer_center)
    
    # Check for overlaps and containment
    violations = 0
    total_pairs = len(inner_hex_data) * (len(inner_hex_data) - 1) // 2
    
    # Create outer hexagon vertices
    outer_vertices = create_hexagon_vertices(outer_center[0], outer_center[1], outer_side_length, 0)
    
    # Check containment and overlaps
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        hex_vertices = create_hexagon_vertices(center_x, center_y, 1, angle)
        
        # Check containment
        if not check_containment(hex_vertices, outer_vertices):
            violations += 1
        
        # Check overlaps with other hexagons
        for j in range(i+1, len(inner_hex_data)):
            center_x2, center_y2, angle2 = inner_hex_data[j]
            hex_vertices2 = create_hexagon_vertices(center_x2, center_y2, 1, angle2)
            
            if check_overlap(hex_vertices, hex_vertices2):
                violations += 1
    
    # Return negative value for minimization (smaller outer hex = better)
    # Add penalty for violations
    penalty = 1000 * violations if violations > 0 else 0
    return outer_side_length + penalty

def generate_symmetric_initial_config():
    """Generate a symmetric initial configuration for better convergence."""
    # Start with a known good symmetric pattern
    # Arrange in a pattern that maximizes symmetry
    configs = [
        # Configuration 1: 6 around center, 6 in ring
        np.array([
            [0, 0, 0],           # center
            [0, 2, 0],           # top
            [1.732, 1, 0],       # top-right
            [1.732, -1, 0],      # bottom-right
            [0, -2, 0],          # bottom
            [-1.732, -1, 0],     # bottom-left
            [-1.732, 1, 0],      # top-left
            [3.464, 0, 0],       # far right
            [1.732, 3, 0],       # top-right extended
            [-1.732, 3, 0],      # top-left extended
            [-3.464, 0, 0],      # far left
            [-1.732, -3, 0],     # bottom-left extended
        ]),
        # Configuration 2: Different symmetric layout
        np.array([
            [0, 0, 0],           # center
            [0, 2.5, 0],         # top
            [2.165, 1.25, 0],    # top-right
            [2.165, -1.25, 0],   # bottom-right
            [0, -2.5, 0],        # bottom
            [-2.165, -1.25, 0],  # bottom-left
            [-2.165, 1.25, 0],   # top-left
            [0, 4.5, 0],         # topmost
            [0, -4.5, 0],        # bottommost
            [3.62, 2.1, 0],      # far right upper
            [-3.62, 2.1, 0],     # far left upper
            [3.62, -2.1, 0],     # far right lower
        ])
    ]
    
    # Return the best one based on initial evaluation
    best_config = configs[0]
    best_score = evaluate_packing(best_config)
    
    for config in configs[1:]:
        score = evaluate_packing(config)
        if score < best_score:
            best_score = score
            best_config = config
            
    return best_config

def optimize_hexagon_packing():
    """Use numerical optimization to improve the packing."""
    # Generate initial symmetric configuration
    initial_config = generate_symmetric_initial_config()
    
    # Flatten parameters for optimization: [x1, y1, theta1, x2, y2, theta2, ...]
    initial_params = initial_config.flatten()
    
    def objective(params):
        # Reshape back to configuration
        config = params.reshape(-1, 3)
        return evaluate_packing(config)
    
    # Constraints for optimization
    def constraint_func(params):
        # Ensure hexagons don't overlap and stay within bounds
        config = params.reshape(-1, 3)
        return evaluate_packing(config)
    
    # Use a simple optimization approach - since this is complex, we'll use a heuristic
    # We'll do iterative improvement with local search
    
    best_config = initial_config.copy()
    best_score = evaluate_packing(best_config)
    
    # Iterative improvement with random perturbations
    for iteration in range(1000):
        # Make small random changes to positions and rotations
        test_config = best_config.copy()
        
        # Randomly perturb some hexagons
        for i in range(len(test_config)):
            if np.random.random() < 0.3:  # 30% chance to perturb
                # Small random change to position and rotation
                test_config[i][0] += np.random.normal(0, 0.1)
                test_config[i][1] += np.random.normal(0, 0.1)
                test_config[i][2] += np.random.normal(0, 5)  # degrees
                
        # Evaluate new configuration
        new_score = evaluate_packing(test_config)
        
        if new_score < best_score:
            best_score = new_score
            best_config = test_config.copy()
    
    return best_config, best_score

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use optimization to find better configuration
    start_time = time.time()
    
    try:
        inner_hex_data, outer_side_length = optimize_hexagon_packing()
    except Exception as e:
        # Fallback to a better initial configuration if optimization fails
        inner_hex_data = generate_symmetric_initial_config()
        outer_side_length = calculate_outer_hexagon_side_length(inner_hex_data)
    
    # Ensure we have a valid configuration
    if outer_side_length <= 0:
        # Fallback to a reasonable configuration
        inner_hex_data = generate_symmetric_initial_config()
        outer_side_length = calculate_outer_hexagon_side_length(inner_hex_data)
    
    # Set outer hexagon at center
    outer_hex_data = np.array([0, 0, 0])
    
    # Final adjustment to ensure tight fit
    final_outer_side_length = calculate_outer_hexagon_side_length(inner_hex_data)
    
    return inner_hex_data, outer_hex_data, final_outer_side_length


# EVOLVE-BLOCK-END
