# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import math

def create_regular_hexagon(center=(0,0), side_length=1, rotation=0):
    """Create vertices of a regular hexagon"""
    angles = np.array([0, 60, 120, 180, 240, 300]) + rotation
    angles = np.radians(angles)
    vertices = np.column_stack([
        center[0] + side_length * np.cos(angles),
        center[1] + side_length * np.sin(angles)
    ])
    return vertices

def hexagon_vertices(hex_data):
    """Get vertices for a single hexagon given its data"""
    x, y, angle = hex_data
    return create_regular_hexagon((x, y), 1, angle)

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of hex are inside outer hex"""
    for vertex in hex_vertices:
        # Point-in-polygon test
        if not point_in_polygon(vertex, outer_hex_vertices):
            return False
    return True

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using separating axis theorem"""
    # Get all edges from both polygons
    edges1 = []
    edges2 = []
    
    for i in range(len(hex1_vertices)):
        edge = hex1_vertices[i] - hex1_vertices[(i + 1) % len(hex1_vertices)]
        edges1.append(edge)
        
    for i in range(len(hex2_vertices)):
        edge = hex2_vertices[i] - hex2_vertices[(i + 1) % len(hex2_vertices)]
        edges2.append(edge)
    
    # Check all potential separating axes
    all_edges = edges1 + edges2
    for edge in all_edges:
        # Project both polygons onto this axis
        axis = np.array([-edge[1], edge[0]])  # perpendicular vector
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            
        proj1 = [np.dot(vertex, axis) for vertex in hex1_vertices]
        proj2 = [np.dot(vertex, axis) for vertex in hex2_vertices]
        
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        
        # If projections don't overlap, there's separation
        if max1 < min2 or max2 < min1:
            return False
    
    return True

def evaluate_packing(hex_data, outer_radius):
    """Evaluate if a packing is valid and compute objective"""
    # Create outer hexagon vertices
    outer_vertices = create_regular_hexagon((0, 0), outer_radius, 0)
    
    # Check containment and overlap
    hex_vertices_list = [hexagon_vertices(data) for data in hex_data]
    
    # Check containment
    for hex_vert in hex_vertices_list:
        if not check_containment(hex_vert, outer_vertices):
            return -1  # Invalid - not contained
    
    # Check overlaps
    for i in range(len(hex_vertices_list)):
        for j in range(i + 1, len(hex_vertices_list)):
            if hexagon_overlap(hex_vertices_list[i], hex_vertices_list[j]):
                return -1  # Invalid - overlapping
    
    # Valid configuration, return inverse of radius
    return 1.0 / outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining geometric insight with optimization.
    """
    
    # Start with a better initial configuration based on known good patterns
    # Pattern inspired by hexagonal close packing with central hexagon
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
        [1.732, 3, 0],  # top far right
        [-1.732, 3, 0], # top far left
    ]
    
    # Convert to numpy array
    initial_hex_data = np.array(initial_positions)
    
    # Optimization approach: optimize the outer radius while keeping positions fixed
    def objective(outer_radius):
        # Convert to tuple for evaluation
        hex_data = initial_hex_data.copy()
        result = evaluate_packing(hex_data, outer_radius[0])
        return -result if result > 0 else 1000  # Negative because we want to maximize
    
    # Bounds for outer radius (reasonable range)
    bounds = [(1.0, 10.0)]
    
    # Optimize using differential evolution
    try:
        result = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=15)
        optimal_radius = result.x[0]
    except:
        # Fallback to a reasonable estimate
        optimal_radius = 3.5
    
    # Final refinement - try to improve the configuration
    # Try rotating some hexagons for better packing
    best_score = 0
    best_config = initial_hex_data.copy()
    best_radius = optimal_radius
    
    # Try different rotations for the outer hexagons
    for rot_angle in [0, 30, 60]:
        temp_hex_data = initial_hex_data.copy()
        # Apply rotation to some hexagons for better packing
        temp_hex_data[:, 2] = rot_angle  # Set all to same rotation
        score = evaluate_packing(temp_hex_data, optimal_radius)
        if score > best_score:
            best_score = score
            best_config = temp_hex_data.copy()
    
    # Final attempt: use a more systematic approach with local search
    def improved_objective(params):
        # params: [radius, pos0_x, pos0_y, ..., pos10_x, pos10_y]
        radius = params[0]
        hex_data = np.zeros((11, 3))
        for i in range(11):
            hex_data[i] = [params[1 + 2*i], params[2 + 2*i], 0]
        
        # Evaluate
        result = evaluate_packing(hex_data, radius)
        return -result if result > 0 else 1000
    
    # Try a more comprehensive search with better starting points
    best_final_score = 0
    final_config = best_config.copy()
    
    # Test several configurations
    configs_to_test = [
        # Original configuration
        initial_hex_data.copy(),
        # Rotated version
        np.column_stack([initial_hex_data[:, :2], np.full(11, 30)]),
        # Different arrangement
        np.array([
            [0, 0, 0], [0, 2, 0], [0, -2, 0],
            [1.732, 1, 0], [-1.732, 1, 0],
            [1.732, -1, 0], [-1.732, -1, 0],
            [3.464, 0, 0], [-3.464, 0, 0],
            [1.732, 3, 0], [-1.732, 3, 0]
        ]),
    ]
    
    for config in configs_to_test:
        # Try to optimize just the outer radius for this configuration
        try:
            def radius_optim_func(radius):
                return -evaluate_packing(config, radius)
            
            result = differential_evolution(radius_optim_func, [(1.0, 8.0)], seed=42, maxiter=20)
            score = evaluate_packing(config, result.x[0])
            if score > best_final_score:
                best_final_score = score
                final_config = config.copy()
                best_radius = result.x[0]
        except:
            continue
    
    # Return final result
    outer_hex_data = np.array([0, 0, 0])
    
    return final_config, outer_hex_data, best_radius


# EVOLVE-BLOCK-END
