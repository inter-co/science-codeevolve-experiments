# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import math


def generate_hexagon_vertices(center_x, center_y, side_length, angle_degrees):
    """Generate vertices of a regular hexagon given center, side length, and rotation."""
    angle_rad = math.radians(angle_degrees)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices


def hexagon_to_polygon(center_x, center_y, side_length, angle_degrees):
    """Convert hexagon parameters to Shapely Polygon object."""
    vertices = generate_hexagon_vertices(center_x, center_y, side_length, angle_degrees)
    return Polygon(vertices)


def check_containment(inner_hex_poly, outer_hex_poly):
    """Check if inner hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(inner_hex_poly)


def calculate_penetration_area(hex1_poly, hex2_poly):
    """Calculate the area of overlap between two hexagons."""
    try:
        intersection = hex1_poly.intersection(hex2_poly)
        return intersection.area
    except:
        return 0.0


def compute_objective(params):
    """Compute objective function for optimization."""
    # Extract parameters
    # params = [x1, y1, angle1, ..., x11, y11, angle11, R]
    n = 11
    hex_params = params[:3*n]  # First 33 params: 11 hexagons with (x,y,angle)
    outer_radius = params[3*n]  # Last param: outer hexagon radius
    
    # Create outer hexagon
    outer_hex = hexagon_to_polygon(0, 0, outer_radius, 0)
    
    # Calculate total penetration penalty
    total_penalty = 0.0
    
    # Check all pairwise intersections and containment
    for i in range(n):
        x1, y1, angle1 = hex_params[3*i:3*i+3]
        hex1 = hexagon_to_polygon(x1, y1, 1.0, angle1)
        
        # Check containment
        if not check_containment(hex1, outer_hex):
            total_penalty += 1000.0  # Large penalty for containment violation
            
        # Check intersections with other hexagons
        for j in range(i+1, n):
            x2, y2, angle2 = hex_params[3*j:3*j+3]
            hex2 = hexagon_to_polygon(x2, y2, 1.0, angle2)
            
            penetration = calculate_penetration_area(hex1, hex2)
            total_penalty += penetration * 1000.0  # Penalty for overlaps
    
    # Objective is to minimize the penalty, so we return negative of 1/outer_radius
    # This means maximizing 1/outer_radius (minimizing outer_radius)
    if total_penalty > 0.0:
        return 1000.0 + total_penalty  # Large penalty for constraint violations
    else:
        return -1.0 / outer_radius  # We want to maximize 1/R, so minimize -1/R


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a force-directed optimization approach with geometric constraints.
    """
    n = 11
    # Initial guess based on a more strategic arrangement
    initial_guess = np.array([
        # Center hexagon
        0.0, 0.0, 0.0,
        # Surrounding hexagons in a ring pattern
        2.0, 0.0, 0.0,
        -2.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, -2.0, 0.0,
        1.0, 1.732, 0.0,
        -1.0, 1.732, 0.0,
        1.0, -1.732, 0.0,
        -1.0, -1.732, 0.0,
        3.0, 0.0, 0.0,
        -3.0, 0.0, 0.0,
        # Outer hexagon radius (initially large)
        6.0
    ])
    
    # Define bounds for optimization
    bounds = []
    # Bounds for hexagon positions (-10, 10) for x and y coordinates
    for _ in range(n):
        bounds.extend([(-10.0, 10.0), (-10.0, 10.0), (0.0, 360.0)])
    # Bounds for outer radius (should be reasonable)
    bounds.append((1.0, 20.0))
    
    # Optimize using L-BFGS-B method
    try:
        result = minimize(
            compute_objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        # Extract optimized parameters
        hex_params = result.x[:-1]
        outer_radius = result.x[-1]
        
        # Format output
        inner_hex_data = np.zeros((n, 3))
        for i in range(n):
            inner_hex_data[i] = hex_params[3*i:3*i+3]
        
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        return inner_hex_data, outer_hex_data, outer_radius
        
    except Exception as e:
        # Fallback to the original approach if optimization fails
        print(f"Optimization failed: {e}")
        # Return the original configuration
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ])

        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = 8  # large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
