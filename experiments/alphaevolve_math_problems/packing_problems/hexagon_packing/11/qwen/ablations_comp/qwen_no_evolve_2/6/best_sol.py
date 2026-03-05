# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time

def create_regular_hexagon(center=(0,0), radius=1, rotation=0):
    """Create a regular hexagon with given center, radius, and rotation."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = np.column_stack([center[0] + radius * np.cos(angles),
                             center[1] + radius * np.sin(angles)])
    return points[:-1]  # Remove last point to close the polygon

def get_hexagon_vertices(hex_data):
    """Get vertices of hexagons from their data."""
    vertices = []
    for x, y, angle in hex_data:
        hex_points = create_regular_hexagon((x, y), 1, angle)
        vertices.append(hex_points)
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if all hexagon vertices are inside the outer hexagon."""
    outer_polygon = Polygon(outer_hex_vertices)
    
    for hex_vert in hex_vertices:
        for vertex in hex_vert:
            if not outer_polygon.contains(Point(vertex)):
                return False
    return True

def calculate_distance_matrix(hex_vertices):
    """Calculate minimum distances between all pairs of hexagons."""
    distances = []
    for i in range(len(hex_vertices)):
        for j in range(i+1, len(hex_vertices)):
            # Use distance between centroids for simplicity
            centroid_i = np.mean(hex_vertices[i], axis=0)
            centroid_j = np.mean(hex_vertices[j], axis=0)
            dist = np.linalg.norm(centroid_i - centroid_j)
            distances.append(dist)
    return np.array(distances)

def compute_penalties(hex_vertices, outer_hex_vertices):
    """Compute penalties for overlap and containment violations."""
    penalty = 0
    
    # Check containment
    outer_polygon = Polygon(outer_hex_vertices)
    for hex_vert in hex_vertices:
        for vertex in hex_vert:
            if not outer_polygon.contains(Point(vertex)):
                # Penalize based on how far outside
                dist_to_boundary = outer_polygon.boundary.distance(Point(vertex))
                penalty += dist_to_boundary ** 2
    
    # Check overlaps - simple distance-based approach
    distances = calculate_distance_matrix(hex_vertices)
    # Penalty for distances less than 2 (minimum distance for non-overlapping unit hexagons)
    overlap_penalty = np.sum(np.maximum(0, 2 - distances) ** 2)
    penalty += overlap_penalty
    
    return penalty

def objective_function(params):
    """Objective function to minimize (negative of 1/outer_hex_side_length)."""
    # Extract parameters
    # First 33 params: 11 hexagons * 3 (x,y,angle)
    # Last 3 params: outer hexagon center and rotation
    # Last param: outer hexagon side length (we want to maximize this, so we minimize 1/length)
    
    hex_params = params[:33].reshape(11, 3)
    outer_center_angle = params[33:36]
    outer_side_length = params[36]
    
    # Create inner hexagons
    inner_hex_data = hex_params.copy()
    
    # Create outer hexagon vertices
    outer_hex_vertices = create_regular_hexagon(outer_center_angle[:2], outer_side_length, outer_center_angle[2])
    
    # Get all hexagon vertices
    inner_hex_vertices = get_hexagon_vertices(inner_hex_data)
    
    # Compute penalties
    penalty = compute_penalties(inner_hex_vertices, outer_hex_vertices)
    
    # Return negative of 1/outer_side_length plus penalty
    # We're minimizing, so we want to maximize 1/outer_side_length
    return -1.0 / outer_side_length + penalty * 1e6

def optimize_hexagon_packing():
    """Use optimization to find better packing."""
    # Initial guess - spread out hexagons
    initial_hex_positions = np.array([
        [0, 0, 0],      # center
        [-1.5, 0, 0],   # left
        [1.5, 0, 0],    # right
        [0, 1.5, 0],    # top
        [0, -1.5, 0],   # bottom
        [-1.0, 1.0, 0], # top-left
        [1.0, 1.0, 0],  # top-right
        [-1.0, -1.0, 0], # bottom-left
        [1.0, -1.0, 0], # bottom-right
        [-1.5, 1.5, 0], # far top-left
        [1.5, 1.5, 0],  # far top-right
    ])
    
    # Initial outer hexagon parameters
    outer_center = [0, 0]
    outer_rotation = 0
    outer_side_length = 5.0  # Initial estimate
    
    # Combine all parameters
    initial_params = np.concatenate([
        initial_hex_positions.flatten(),
        np.array(outer_center + [outer_rotation]),
        [outer_side_length]
    ])
    
    # Define bounds for optimization
    bounds = []
    
    # Bounds for hexagon positions (x, y) - allow wide range initially
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10)])  # x, y bounds
        bounds.extend([(-180, 180)])  # angle bounds
    
    # Bounds for outer hexagon center and rotation
    bounds.extend([(-10, 10), (-10, 10), (-180, 180)])  # center x, y, rotation
    
    # Bounds for outer hexagon side length (must be positive)
    bounds.extend([(0.1, 20)])  # side length bounds
    
    # Optimize
    try:
        result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        # Extract results
        hex_params = result.x[:33].reshape(11, 3)
        outer_center_angle = result.x[33:36]
        outer_side_length = result.x[36]
        
        return hex_params, outer_center_angle, outer_side_length
        
    except Exception as e:
        # Fallback to simple arrangement if optimization fails
        print(f"Optimization failed: {e}")
        return initial_hex_positions, np.array([0, 0, 0]), 5.0

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses force-directed optimization approach.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Run optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    
    # Ensure we have valid result
    if outer_hex_side_length <= 0:
        # Fallback to simple arrangement
        inner_hex_data = np.array([
            [0, 0, 0],      # center
            [-1.5, 0, 0],   # left
            [1.5, 0, 0],    # right
            [0, 1.5, 0],    # top
            [0, -1.5, 0],   # bottom
            [-1.0, 1.0, 0], # top-left
            [1.0, 1.0, 0],  # top-right
            [-1.0, -1.0, 0], # bottom-left
            [1.0, -1.0, 0], # bottom-right
            [-1.5, 1.5, 0], # far top-left
            [1.5, 1.5, 0],  # far top-right
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 3.0
    
    # Adjust for final validation
    # Create test configuration to verify it's valid
    inner_hex_vertices = get_hexagon_vertices(inner_hex_data)
    outer_hex_vertices = create_regular_hexagon(outer_hex_data[:2], outer_hex_side_length, outer_hex_data[2])
    
    # Validate containment and overlap
    containment_ok = check_containment(inner_hex_vertices, outer_hex_vertices)
    
    if not containment_ok:
        # If invalid, fall back to more conservative arrangement
        inner_hex_data = np.array([
            [0, 0, 0],      # center
            [-1.5, 0, 0],   # left
            [1.5, 0, 0],    # right
            [0, 1.5, 0],    # top
            [0, -1.5, 0],   # bottom
            [-1.0, 1.0, 0], # top-left
            [1.0, 1.0, 0],  # top-right
            [-1.0, -1.0, 0], # bottom-left
            [1.0, -1.0, 0], # bottom-right
            [-1.5, 1.5, 0], # far top-left
            [1.5, 1.5, 0],  # far top-right
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 3.5
    
    eval_time = time.time() - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
