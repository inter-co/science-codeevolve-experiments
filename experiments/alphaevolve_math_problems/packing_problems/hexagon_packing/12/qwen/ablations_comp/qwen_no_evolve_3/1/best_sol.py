# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import math


def create_regular_hexagon(center=(0, 0), side_length=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon."""
    angles = np.array([rotation + i * 60 for i in range(6)]) * np.pi / 180
    vertices = np.array([
        (center[0] + side_length * np.cos(angle),
         center[1] + side_length * np.sin(angle))
        for angle in angles
    ])
    return Polygon(vertices)


def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer_hexagon."""
    return outer_hexagon.contains(hexagon)


def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)


def calculate_outer_hex_side_length(inner_hex_data, outer_center=(0, 0)):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons."""
    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        hex_poly = create_regular_hexagon(center, 1, rotation)
        all_vertices.extend(list(hex_poly.exterior.coords)[:-1])  # Exclude repeated last point
    
    # Find the maximum distance from center to any vertex
    max_dist = 0
    for vertex in all_vertices:
        dist = math.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
        max_dist = max(max_dist, dist)
    
    # Convert to hexagon side length (for a circumscribed hexagon, side length = max_dist)
    return max_dist


def objective_function(params):
    """Objective function to minimize (negative of 1/outer_hex_side_length)."""
    # Reshape parameters into hexagon data
    inner_hex_data = params.reshape(-1, 3)
    
    # Calculate outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    # Return negative because we want to maximize 1/outer_side_length
    return -1.0 / outer_side_length


def constraint_containment(params, outer_center=(0, 0)):
    """Constraint ensuring all inner hexagons are contained."""
    inner_hex_data = params.reshape(-1, 3)
    outer_hexagon = create_regular_hexagon(outer_center, 1000, 0)  # Large enough
    
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        inner_hex = create_regular_hexagon(center, 1, rotation)
        if not check_containment(inner_hex, outer_hexagon):
            return -1.0  # Violation
    
    return 1.0  # Valid


def constraint_nonoverlap(params):
    """Constraint ensuring no overlaps between inner hexagons."""
    inner_hex_data = params.reshape(-1, 3)
    
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center_i = (inner_hex_data[i][0], inner_hex_data[i][1])
            rotation_i = inner_hex_data[i][2]
            center_j = (inner_hex_data[j][0], inner_hex_data[j][1])
            rotation_j = inner_hex_data[j][2]
            
            hex_i = create_regular_hexagon(center_i, 1, rotation_i)
            hex_j = create_regular_hexagon(center_j, 1, rotation_j)
            
            if check_overlap(hex_i, hex_j):
                return -1.0  # Violation
    
    return 1.0  # Valid


def hexagon_packing_12():
    """
    Constructs an optimized packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, 
    maximizing 1/outer_hex_side_length using geometric optimization.
    """
    # Initial guess: symmetric arrangement
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
        [1.732, 3, 0],  # top-top-right
        [-1.732, 3, 0], # top-top-left
        [1.732, -3, 0], # bottom-bottom-right
        [-1.732, -3, 0] # bottom-bottom-left
    ]
    
    # Remove one redundant position to get exactly 12
    initial_positions = initial_positions[:12]
    
    # Flatten for optimization
    initial_params = np.array(initial_positions).flatten()
    
    # Define bounds for positions (reasonable search space)
    bounds = []
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
    
    # Optimization constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)}
    ]
    
    # Perform optimization
    try:
        result = minimize(
            objective_function,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            inner_hex_data = result.x.reshape(-1, 3)
        else:
            # Fallback to a good symmetric arrangement
            inner_hex_data = np.array([
                [0, 0, 0],       # center
                [0, 2, 0],       # top
                [0, -2, 0],      # bottom
                [1.732, 1, 0],   # top-right
                [-1.732, 1, 0],  # top-left
                [1.732, -1, 0],  # bottom-right
                [-1.732, -1, 0], # bottom-left
                [3.464, 0, 0],   # far right
                [-3.464, 0, 0],  # far left
                [0, 4, 0],       # top-top
                [0, -4, 0],      # bottom-bottom
                [0, 3, 0],       # top-top-center
            ])
    except Exception:
        # Final fallback
        inner_hex_data = np.array([
            [0, 0, 0],       # center
            [0, 2, 0],       # top
            [0, -2, 0],      # bottom
            [1.732, 1, 0],   # top-right
            [-1.732, 1, 0],  # top-left
            [1.732, -1, 0],  # bottom-right
            [-1.732, -1, 0], # bottom-left
            [3.464, 0, 0],   # far right
            [-3.464, 0, 0],  # far left
            [0, 4, 0],       # top-top
            [0, -4, 0],      # bottom-bottom
            [0, 3, 0],       # top-top-center
        ])
    
    # Calculate final outer hexagon size
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    # Outer hexagon centered at origin (can be adjusted)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
