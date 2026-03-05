# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import math
from typing import Tuple, List


def create_unit_hexagon(center=(0, 0), rotation=0):
    """Create a unit regular hexagon centered at center with given rotation."""
    angle = rotation * math.pi / 180
    radius = 1  # unit hexagon side length
    points = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        points.append((x, y))
    return Polygon(points)


def check_hexagon_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer_hexagon."""
    return outer_hexagon.contains(hexagon)


def calculate_hexagon_distance(hex1, hex2):
    """Calculate minimum distance between two hexagons."""
    return hex1.distance(hex2)


def compute_outer_hexagon_radius(inner_hex_data, outer_center=(0, 0)):
    """Compute the minimum radius needed for outer hexagon to contain all inner hexagons."""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        # Get all vertices of the inner hexagon
        hexagon = create_unit_hexagon((x, y), angle)
        # Find the maximum distance from center to any vertex
        for point in list(hexagon.exterior.coords):
            dist = math.sqrt((point[0] - outer_center[0])**2 + (point[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    return max_dist


def evaluate_configuration(inner_hex_data, outer_center=(0, 0)):
    """Evaluate a configuration: returns negative of 1/outer_radius to use with minimization."""
    # Create outer hexagon based on the current configuration
    outer_radius = compute_outer_hexagon_radius(inner_hex_data, outer_center)
    
    # Check for overlaps
    total_penalty = 0
    n = len(inner_hex_data)
    
    for i in range(n):
        hex1 = create_unit_hexagon((inner_hex_data[i][0], inner_hex_data[i][1]), inner_hex_data[i][2])
        for j in range(i+1, n):
            hex2 = create_unit_hexagon((inner_hex_data[j][0], inner_hex_data[j][1]), inner_hex_data[j][2])
            if hex1.intersects(hex2):
                # Large penalty for overlaps
                overlap_area = hex1.intersection(hex2).area
                total_penalty += overlap_area * 10000
    
    # Check containment
    outer_hexagon = create_unit_hexagon(outer_center, 0)
    for i in range(n):
        hex1 = create_unit_hexagon((inner_hex_data[i][0], inner_hex_data[i][1]), inner_hex_data[i][2])
        if not check_hexagon_containment(hex1, outer_hexagon):
            total_penalty += 10000
    
    # Return negative of 1/outer_radius plus penalties
    if outer_radius > 0:
        objective_value = -1.0 / outer_radius + total_penalty
    else:
        objective_value = -10000 + total_penalty
    
    return objective_value


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining geometric intuition with optimization.
    """
    # Initial configuration based on a more structured approach
    # Try to arrange in a pattern similar to a hexagonal close packing
    initial_positions = [
        (0, 0, 0),      # center
        (0, 2, 0),      # top
        (0, -2, 0),     # bottom  
        (1.732, 1, 0),  # top-right
        (-1.732, 1, 0), # top-left
        (1.732, -1, 0), # bottom-right
        (-1.732, -1, 0),# bottom-left
        (3.464, 0, 0),  # far right
        (-3.464, 0, 0), # far left
        (1.732, 3, 0),  # top-top-right
        (-1.732, 3, 0), # top-top-left
    ]
    
    # Convert to numpy array
    inner_hex_data = np.array(initial_positions)
    
    # Optimize using scipy minimize
    def objective(params):
        # Reshape params back into hexagon data
        hex_data = params.reshape(-1, 3)
        return evaluate_configuration(hex_data)
    
    # Flatten initial parameters
    initial_params = inner_hex_data.flatten()
    
    # Optimization bounds (positions within reasonable range, angles 0-360)
    bounds = [(None, None), (None, None), (0, 360)] * 11
    
    # Perform optimization
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        # Extract optimized solution
        if result.success:
            optimized_hex_data = result.x.reshape(-1, 3)
        else:
            # Fallback to initial if optimization fails
            optimized_hex_data = inner_hex_data
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        optimized_hex_data = inner_hex_data
    
    # Calculate final outer hexagon size
    outer_radius = compute_outer_hexagon_radius(optimized_hex_data)
    outer_side_length = outer_radius  # For regular hexagon, radius equals side length
    
    # Final adjustment to ensure we have proper orientation and positioning
    # Use a more conservative estimate for outer hexagon
    outer_side_length = compute_outer_hexagon_radius(optimized_hex_data) * 1.1  # Add some margin
    
    # Ensure the outer hexagon is centered at origin
    outer_hex_data = np.array([0, 0, 0])
    
    return optimized_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
