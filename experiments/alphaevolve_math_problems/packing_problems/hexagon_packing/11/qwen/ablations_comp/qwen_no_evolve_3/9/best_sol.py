# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import math


def hexagon_vertices(center_x, center_y, side_length, rotation_degrees):
    """Generate vertices of a regular hexagon."""
    angle_rad = math.radians(rotation_degrees)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices


def hexagon_polygon(center_x, center_y, side_length, rotation_degrees):
    """Create a Shapely polygon for a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, side_length, rotation_degrees)
    return Polygon(vertices)


def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hexagon_poly)


def calculate_overlap_penalty(hex1_poly, hex2_poly):
    """Calculate overlap penalty between two hexagons."""
    if hex1_poly.intersects(hex2_poly):
        intersection = hex1_poly.intersection(hex2_poly)
        return intersection.area
    return 0.0


def evaluate_configuration(inner_positions, outer_radius):
    """Evaluate configuration and return penalty."""
    # Create outer hexagon
    outer_hex = hexagon_polygon(0, 0, outer_radius, 0)
    
    # Calculate total penalty
    total_penalty = 0.0
    
    # Check containment and overlap penalties
    hexagons = []
    for i, pos in enumerate(inner_positions):
        center_x, center_y, angle = pos
        hex_poly = hexagon_polygon(center_x, center_y, 1.0, angle)
        
        # Check containment
        if not check_containment(hex_poly, outer_hex):
            total_penalty += 1000.0  # Large penalty for containment violation
            
        hexagons.append(hex_poly)
        
        # Check overlaps with other hexagons
        for j in range(i):
            overlap = calculate_overlap_penalty(hexagons[i], hexagons[j])
            total_penalty += overlap * 1000.0  # Penalty for overlaps
    
    return total_penalty


def objective_function(params):
    """Objective function to minimize (negative of 1/outer_radius)."""
    # Extract parameters
    outer_radius = params[-1]  # Last parameter is outer radius
    inner_params = params[:-1].reshape(-1, 3)  # First N*3 parameters are positions/angles
    
    # Evaluate configuration
    penalty = evaluate_configuration(inner_params, outer_radius)
    
    # Return negative of 1/outer_radius plus penalty
    return -(1.0 / outer_radius) + penalty


def generate_initial_guess():
    """Generate initial configuration using a more sophisticated layout."""
    # Start with a hexagonal pattern
    positions = []
    
    # Center hexagon
    positions.append([0.0, 0.0, 0.0])
    
    # Surrounding hexagons in 2 layers
    layer1_radius = 2.0  # Distance from center
    layer2_radius = 3.5  # Distance from center
    
    # Layer 1: 6 hexagons around center
    for i in range(6):
        angle = i * 60  # 60 degrees apart
        rad_angle = math.radians(angle)
        x = layer1_radius * math.cos(rad_angle)
        y = layer1_radius * math.sin(rad_angle)
        positions.append([x, y, 0.0])
    
    # Layer 2: 6 hexagons in outer ring
    for i in range(6):
        angle = i * 60 + 30  # Offset by 30 degrees
        rad_angle = math.radians(angle)
        x = layer2_radius * math.cos(rad_angle)
        y = layer2_radius * math.sin(rad_angle)
        positions.append([x, y, 0.0])
    
    # Add one more hexagon to fill the pattern
    positions.append([0.0, 0.0, 0.0])  # Placeholder
    
    # Trim to exactly 11 positions
    positions = positions[:11]
    
    # Initialize with reasonable outer radius
    outer_radius = 6.0
    
    # Flatten parameters
    flat_params = np.array(positions).flatten()
    flat_params = np.append(flat_params, outer_radius)
    
    return flat_params


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses optimization-based approach with physics-inspired constraints.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate initial guess
    x0 = generate_initial_guess()
    
    # Define bounds for optimization
    # Positions: -10 to 10 for x and y
    # Angles: 0 to 360 degrees
    # Outer radius: 2 to 20 (reasonable range)
    bounds = []
    for i in range(11):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle for each hexagon
    bounds.append((2, 20))  # outer radius
    
    # Optimization settings
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Perform optimization
        result = minimize(objective_function, x0, method='L-BFGS-B', bounds=bounds, options=options)
        
        # Extract results
        best_params = result.x
        outer_radius = best_params[-1]
        inner_positions = best_params[:-1].reshape(-1, 3)
        
        # Create final arrays
        inner_hex_data = np.array(inner_positions)
        outer_hex_data = np.array([0, 0, 0])  # Centered at origin
        
    except Exception as e:
        # Fallback to original configuration if optimization fails
        print(f"Optimization failed: {e}")
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_radius = 8.0
    
    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END
