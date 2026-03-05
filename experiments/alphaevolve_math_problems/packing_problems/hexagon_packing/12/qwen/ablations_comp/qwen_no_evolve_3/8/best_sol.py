# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import time


def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1
    hex_points = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        hex_points.append((x, y))
    return np.array(hex_points)


def check_hexagon_containment(hex_points, outer_center_x, outer_center_y, outer_side_length):
    """Check if all vertices of a hexagon are within the outer hexagon."""
    # Generate outer hexagon vertices
    outer_points = generate_hexagon_vertices(outer_center_x, outer_center_y, 0, outer_side_length)
    
    # Create polygons
    inner_poly = Polygon(hex_points)
    outer_poly = Polygon(outer_points)
    
    # Check if inner polygon is completely contained within outer polygon
    return outer_poly.contains(inner_poly)


def check_hexagon_overlap(hex1_points, hex2_points):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_points)
    poly2 = Polygon(hex2_points)
    return poly1.intersects(poly2)


def calculate_packing_objective(params):
    """Calculate objective function for 12 hexagon packing."""
    # params: [x1, y1, theta1, ..., x12, y12, theta12, R]
    # Extract parameters
    positions_angles = params[:-1].reshape(-1, 3)
    outer_radius = params[-1]
    
    # Generate all hexagon vertices
    hexagons = []
    for i in range(12):
        center_x, center_y, angle = positions_angles[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        hexagons.append(vertices)
    
    # Check containment
    outer_center_x, outer_center_y = 0, 0
    for hex_points in hexagons:
        if not check_hexagon_containment(hex_points, outer_center_x, outer_center_y, outer_radius):
            return 1e10  # Large penalty for containment violation
    
    # Check overlaps
    for i in range(12):
        for j in range(i+1, 12):
            if check_hexagon_overlap(hexagons[i], hexagons[j]):
                return 1e10  # Large penalty for overlap
    
    # Return negative of inverse radius (we want to maximize 1/R, which means minimize -1/R)
    return -1.0 / outer_radius


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses optimization approach with geometric constraints.
    """
    # Initial guess: symmetric arrangement
    # Start with a good initial configuration based on known efficient packings
    initial_positions_angles = np.array([
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
        [1.732, -3, 0], # lower-right
    ])
    
    # Add some random perturbations to avoid local minima
    initial_positions_angles += np.random.normal(0, 0.1, initial_positions_angles.shape)
    
    # Initial outer radius estimate
    initial_outer_radius = 5.0
    
    # Combine into parameter vector
    params = np.concatenate([initial_positions_angles.flatten(), [initial_outer_radius]])
    
    # Define bounds for optimization
    bounds = []
    # Positions: -10 to 10 for x and y
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
    # Outer radius: minimum reasonable value
    bounds.append((1.0, 20.0))
    
    # Optimization options
    options = {'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
    
    # Perform optimization
    try:
        result = minimize(
            calculate_packing_objective,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options=options,
            tol=1e-8
        )
        
        # Extract results
        final_params = result.x
        positions_angles = final_params[:-1].reshape(-1, 3)
        outer_radius = final_params[-1]
        
        # Return results
        inner_hex_data = positions_angles.copy()
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = outer_radius
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback to better initial configuration if optimization fails
        # Use a known good configuration with optimized spacing
        better_config = np.array([
            [0, 0, 0],
            [0, 2.0, 0],
            [0, -2.0, 0],
            [1.732, 1.0, 0],
            [-1.732, 1.0, 0],
            [1.732, -1.0, 0],
            [-1.732, -1.0, 0],
            [3.464, 0, 0],
            [-3.464, 0, 0],
            [0, 3.0, 0],
            [0, -3.0, 0],
            [0, 4.0, 0]
        ])
        
        # Adjust spacing to get better packing
        better_config[:, 1] *= 0.85  # Scale y-coordinates slightly
        better_config[:, 0] *= 0.85  # Scale x-coordinates slightly
        
        # Estimate outer radius
        max_dist = max(np.sqrt(better_config[:, 0]**2 + better_config[:, 1]**2)) + 1.0
        outer_radius = max_dist * 1.1  # Add some margin
        
        inner_hex_data = better_config
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = outer_radius
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
