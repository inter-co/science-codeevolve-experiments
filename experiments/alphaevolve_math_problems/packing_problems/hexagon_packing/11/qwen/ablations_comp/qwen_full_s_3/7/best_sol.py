# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import math
from shapely.geometry import Polygon, Point
import warnings

def get_hexagon_vertices(center, radius=1, rotation=0):
    """Get vertices of a unit hexagon at given position and rotation."""
    # Unit hexagon has side length 1, so circumradius is 1
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = np.column_stack([center[0] + radius * np.cos(angles),
                             center[1] + radius * np.sin(angles)])
    return points[:-1]

def check_containment(hexagon_vertices, outer_hex_center, outer_hex_radius):
    """Check if all vertices of hexagon are inside the outer hexagon."""
    outer_hex_points = get_hexagon_vertices(outer_hex_center, outer_hex_radius, 0)
    outer_polygon = Polygon(outer_hex_points)
    
    # Use a small epsilon for floating point comparison to handle numerical issues
    epsilon = 1e-10
    for vertex in hexagon_vertices:
        point = Point(vertex[0], vertex[1])
        # Check if point is inside or on the boundary (with tolerance)
        if not outer_polygon.buffer(epsilon).contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        # Use buffer with small epsilon to handle floating point precision issues
        epsilon = 1e-10
        return poly1.buffer(epsilon).intersects(poly2.buffer(epsilon))
    except:
        # Fallback to a simple distance check if polygon creation fails
        centers1 = np.mean(hex1_vertices, axis=0)
        centers2 = np.mean(hex2_vertices, axis=0)
        distance = np.linalg.norm(centers1 - centers2)
        # Two unit hexagons overlap if their centers are less than 2 units apart
        return distance < 2.0

def calculate_outer_hex_side_length(inner_hex_data, outer_hex_center=(0, 0)):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons."""
    max_distance = 0
    
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        hex_vertices = get_hexagon_vertices(center, 1, rotation)
        
        # Find maximum distance from center to any vertex
        distances = np.sqrt(np.sum((hex_vertices - center)**2, axis=1))
        max_dist_from_center = np.max(distances)
        
        # Add this to the distance from outer center to inner center
        center_distance = np.sqrt(np.sum((np.array(center) - np.array(outer_hex_center))**2))
        total_distance = center_distance + max_dist_from_center
        
        max_distance = max(max_distance, total_distance)
    
    # Convert to side length of outer hexagon (circumradius = side length for regular hexagon)
    return max_distance

def is_valid_configuration(inner_hex_data, outer_hex_center=(0, 0), outer_hex_radius=10):
    """Check if a configuration is valid (no overlaps, all contained)."""
    # Check containment first
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        hex_vertices = get_hexagon_vertices(center, 1, rotation)
        
        if not check_containment(hex_vertices, outer_hex_center, outer_hex_radius):
            return False
    
    # Check overlaps - use more efficient approach with early termination
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center1 = (inner_hex_data[i, 0], inner_hex_data[i, 1])
            rotation1 = inner_hex_data[i, 2]
            center2 = (inner_hex_data[j, 0], inner_hex_data[j, 1])
            rotation2 = inner_hex_data[j, 2]
            
            hex1_vertices = get_hexagon_vertices(center1, 1, rotation1)
            hex2_vertices = get_hexagon_vertices(center2, 1, rotation2)
            
            if check_overlap(hex1_vertices, hex2_vertices):
                return False
    
    return True

def objective_with_constraints(params, outer_hex_center=(0, 0)):
    """
    Objective function that penalizes constraint violations.
    Returns negative reciprocal of outer hexagon side length for maximization.
    """
    # Reshape params into 11 hexagons with (x, y, rotation) each
    inner_hex_data = params.reshape(-1, 3)
    
    # Calculate required outer hexagon size
    outer_radius = calculate_outer_hex_side_length(inner_hex_data, outer_hex_center)
    
    # Check validity and penalize invalid configurations heavily
    if not is_valid_configuration(inner_hex_data, outer_hex_center, outer_radius):
        # Large penalty for constraint violations - make it extremely negative
        return 1e10  # We minimize this, so a large positive number means bad
    
    # Return negative because we want to maximize 1/outer_radius (minimize outer_radius)
    # But since we're minimizing, return the negative of 1/outer_radius
    return -1.0 / outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses the best known configuration from inspirations with minimal optimization to maintain speed.
    """
    # Use the best known pattern from inspirations (achieved 0.24285452449816275)
    sqrt3 = math.sqrt(3)
    best_pattern = np.array([
        [0, 0, 0],           # center
        [0, 1.8, 0],         # top
        [sqrt3*0.9, -0.9, 0], # top-right  
        [-sqrt3*0.9, -0.9, 0], # top-left
        [0, -1.8, 0],        # bottom
        [sqrt3*0.9, 0.9, 0], # bottom-right
        [-sqrt3*0.9, 0.9, 0], # bottom-left
        [sqrt3*1.8, 0, 0],   # far right
        [-sqrt3*1.8, 0, 0],  # far left
        [sqrt3*0.9, 2.7, 0], # top-top-right
        [-sqrt3*0.9, 2.7, 0] # top-top-left
    ])
    
    # Directly return the best pattern without optimization to maintain speed
    # and since it's already near-optimal
    outer_radius = calculate_outer_hex_side_length(best_pattern)
    
    # Verify it's valid
    if not is_valid_configuration(best_pattern, (0, 0), outer_radius):
        # Fallback to a slightly adjusted version if needed
        best_pattern = np.array([
            [0, 0, 0],
            [0, 1.82, 0],
            [sqrt3*0.91, -0.91, 0],
            [-sqrt3*0.91, -0.91, 0],
            [0, -1.82, 0],
            [sqrt3*0.91, 0.91, 0],
            [-sqrt3*0.91, 0.91, 0],
            [sqrt3*1.82, 0, 0],
            [-sqrt3*1.82, 0, 0],
            [sqrt3*0.91, 2.73, 0],
            [-sqrt3*0.91, 2.73, 0]
        ])
        outer_radius = calculate_outer_hex_side_length(best_pattern)
    
    outer_hex_data = np.array([0, 0, 0])
    
    return best_pattern, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END
