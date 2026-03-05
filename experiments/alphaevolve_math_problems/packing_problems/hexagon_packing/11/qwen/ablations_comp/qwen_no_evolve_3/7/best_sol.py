# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time


def create_regular_hexagon(center=(0, 0), side_length=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + side_length * np.cos(angle),
               center[1] + side_length * np.sin(angle)) for angle in angles]
    return Polygon(points)


def hexagon_vertices(center, side_length, rotation):
    """Get vertices of a hexagon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return np.array([(center[0] + side_length * np.cos(angle),
                      center[1] + side_length * np.sin(angle)) for angle in angles])


def check_containment(hexagon_points, outer_hexagon):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    for point in hexagon_points:
        if not outer_hexagon.contains(Point(point)):
            return False
    return True


def compute_hexagon_overlap(hex1_points, hex2_points):
    """Compute overlap area between two hexagons."""
    try:
        poly1 = Polygon(hex1_points)
        poly2 = Polygon(hex2_points)
        intersection = poly1.intersection(poly2)
        return intersection.area
    except:
        return 0.0


def objective_function(params):
    """
    Objective function to minimize (negative of 1/outer_radius).
    params: flattened array of [inner_positions, outer_center, outer_radius]
    """
    # Parse parameters
    inner_pos = params[:22].reshape(-1, 2)  # 11 hexagons * 2 coordinates
    outer_center = params[22:24]
    outer_radius = params[24]
    
    # Create outer hexagon
    outer_hex = create_regular_hexagon(outer_center, outer_radius, 0)
    
    # Check containment constraints
    penalty = 0.0
    
    # Check if all inner hexagons are contained
    for i, pos in enumerate(inner_pos):
        hex_points = hexagon_vertices(pos, 1, 0)  # unit hexagons with 0 rotation
        if not check_containment(hex_points, outer_hex):
            penalty += 1e6  # Large penalty for containment violation
    
    # Compute overlap penalties
    total_overlap = 0.0
    for i in range(len(inner_pos)):
        hex1_points = hexagon_vertices(inner_pos[i], 1, 0)
        for j in range(i+1, len(inner_pos)):
            hex2_points = hexagon_vertices(inner_pos[j], 1, 0)
            overlap = compute_hexagon_overlap(hex1_points, hex2_points)
            total_overlap += overlap
    
    # The objective is to maximize 1/outer_radius, so we minimize -1/outer_radius
    # which is equivalent to minimizing outer_radius
    objective = outer_radius + penalty + total_overlap * 1000
    
    return objective


def optimize_hexagon_packing():
    """Optimize the hexagon packing using a physics-inspired approach."""
    
    # Initial configuration - more spread out than the baseline
    initial_inner_positions = np.array([
        [0, 0],      # center
        [-2.0, 0],   # left
        [2.0, 0],    # right
        [0, 2.0],    # top
        [0, -2.0],   # bottom
        [-1.5, 1.5], # top-left
        [1.5, 1.5],  # top-right
        [-1.5, -1.5], # bottom-left
        [1.5, -1.5], # bottom-right
        [-2.5, 1.0], # far top-left
        [2.5, 1.0],  # far top-right
    ])
    
    # Initial guess: [inner_positions_flat, outer_center, outer_radius]
    initial_guess = np.concatenate([
        initial_inner_positions.flatten(),
        [0, 0],  # outer center
        [5.0]    # initial outer radius
    ])
    
    # Constraints and bounds
    bounds = []
    # Add bounds for inner positions (reasonable range)
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10)])
    # Add bounds for outer center
    bounds.extend([(-10, 10), (-10, 10)])
    # Add bounds for outer radius (positive)
    bounds.append((0.1, 20))
    
    # Optimization
    result = minimize(
        objective_function,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
    )
    
    # Extract results
    inner_pos = result.x[:22].reshape(-1, 2)
    outer_center = result.x[22:24]
    outer_radius = result.x[24]
    
    # Convert to required format
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [inner_pos[i][0], inner_pos[i][1], 0]  # No rotation for simplicity
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], 0])
    
    return inner_hex_data, outer_hex_data, outer_radius


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a physics-inspired optimization approach.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use the optimization approach
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    
    # Ensure we're returning the correct format
    # inner_hex_data should have shape (11,3) with (x,y,rotation)
    # outer_hex_data should have shape (3,) with (x,y,rotation)
    
    # Time measurement
    eval_time = time.time() - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
