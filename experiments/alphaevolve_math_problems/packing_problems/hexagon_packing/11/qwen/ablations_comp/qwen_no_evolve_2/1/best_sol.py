# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import math

def create_regular_hexagon(center=(0,0), side_length=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon"""
    angle_offset = rotation * math.pi / 180
    points = []
    for i in range(6):
        angle = angle_offset + i * math.pi / 3
        x = center[0] + side_length * math.cos(angle)
        y = center[1] + side_length * math.sin(angle)
        points.append((x, y))
    return Polygon(points)

def hexagon_vertices(center, side_length, rotation):
    """Get vertices of a hexagon"""
    angle_offset = rotation * math.pi / 180
    vertices = []
    for i in range(6):
        angle = angle_offset + i * math.pi / 3
        x = center[0] + side_length * math.cos(angle)
        y = center[1] + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained in outer hexagon"""
    return outer_hex_poly.contains(hexagon_poly) or outer_hex_poly.intersects(hexagon_poly)

def calculate_outer_hex_side_length(inner_hex_data, outer_hex_center=(0,0), outer_hex_rotation=0):
    """Calculate minimum outer hexagon side length that contains all inner hexagons"""
    # Create all inner hexagons
    inner_hexagons = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        hex_poly = create_regular_hexagon(center, 1, rotation)
        inner_hexagons.append(hex_poly)
    
    # Find bounding box of all inner hexagons
    all_points = []
    for hex_poly in inner_hexagons:
        for point in hex_poly.exterior.coords:
            all_points.append(point)
    
    # Calculate min bounding circle radius and convert to hexagon side length
    if not all_points:
        return 1000
    
    # Get the maximum distance from center to any vertex
    max_dist = 0
    for point in all_points:
        dist = math.sqrt((point[0] - outer_hex_center[0])**2 + (point[1] - outer_hex_center[1])**2)
        max_dist = max(max_dist, dist)
    
    # Convert to hexagon side length (for a circumscribed hexagon)
    # In a regular hexagon, the circumradius equals the side length
    return max_dist * 2 / math.sqrt(3)  # Approximate conversion

def evaluate_packing(inner_hex_data, outer_hex_center=(0,0), outer_hex_rotation=0):
    """Evaluate if the packing is valid and return the outer hexagon side length"""
    # Create all inner hexagons
    inner_hexagons = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        hex_poly = create_regular_hexagon(center, 1, rotation)
        inner_hexagons.append(hex_poly)
    
    # Check for overlaps
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            if inner_hexagons[i].intersects(inner_hexagons[j]):
                return float('inf')  # Invalid packing due to overlap
    
    # Calculate outer hexagon side length
    outer_radius = 0
    for hex_poly in inner_hexagons:
        for point in hex_poly.exterior.coords:
            dist = math.sqrt((point[0] - outer_hex_center[0])**2 + (point[1] - outer_hex_center[1])**2)
            outer_radius = max(outer_radius, dist)
    
    # Convert to hexagon side length
    # For a regular hexagon, side length = circumradius * sqrt(3)/3 * 2
    outer_side_length = outer_radius * 2 / math.sqrt(3)
    return outer_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a global optimization approach to find a better configuration than the initial grid.
    """
    
    # Define bounds for optimization
    # Format: [x0, y0, angle0, x1, y1, angle1, ..., x10, y10, angle10]
    bounds = []
    # Center hexagon: x, y, angle
    bounds.extend([(-3, 3), (-3, 3), (0, 360)])
    # Other hexagons: x, y, angle (more restrictive bounds for better convergence)
    for _ in range(10):
        bounds.extend([(-5, 5), (-5, 5), (0, 360)])
    
    def objective(x):
        # Convert flat array to 11 hexagon data
        inner_hex_data = []
        for i in range(11):
            idx = i * 3
            inner_hex_data.append([x[idx], x[idx+1], x[idx+2]])
        
        # Evaluate the packing
        side_length = evaluate_packing(np.array(inner_hex_data))
        if math.isinf(side_length):
            return 1000  # Penalty for invalid configurations
        return side_length
    
    # Use differential evolution for global optimization
    result = differential_evolution(objective, bounds, seed=42, maxiter=100, popsize=15)
    
    # Extract best solution
    best_x = result.x
    inner_hex_data = []
    for i in range(11):
        idx = i * 3
        inner_hex_data.append([best_x[idx], best_x[idx+1], best_x[idx+2]])
    
    inner_hex_data = np.array(inner_hex_data)
    
    # Calculate final outer hexagon side length
    outer_hex_side_length = evaluate_packing(inner_hex_data)
    
    # Set outer hexagon at center
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
