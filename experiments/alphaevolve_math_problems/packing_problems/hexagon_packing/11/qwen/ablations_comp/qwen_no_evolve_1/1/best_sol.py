# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time

def create_hexagon_vertices(center, side_length, rotation_degrees):
    """Create vertices of a regular hexagon given center, side length, and rotation."""
    angle_rad = np.radians(rotation_degrees)
    # Vertices of a regular hexagon with side length 1, centered at origin
    base_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = base_vertices @ rotation_matrix.T
    return rotated_vertices * side_length + np.array(center)

def check_containment(hexagon_vertices, outer_hex_vertices):
    """Check if hexagon is fully contained within outer hexagon."""
    inner_polygon = Polygon(hexagon_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def evaluate_configuration(config, outer_radius):
    """Evaluate a configuration: returns penalty if invalid, or negative area if valid."""
    # Parse config into positions and rotations
    positions = config[:22].reshape(-1, 2)  # 11 hexagons * 2 coordinates
    rotations = config[22:]  # 11 rotations
    
    # Create hexagon vertices
    hexagons = []
    for i in range(11):
        pos = positions[i]
        rot = rotations[i]
        vertices = create_hexagon_vertices(pos, 1.0, rot)
        hexagons.append(vertices)
    
    # Check containment
    outer_hex_vertices = create_hexagon_vertices([0, 0], outer_radius, 0)
    
    # Check all pairwise overlaps
    total_penalty = 0
    
    # Check containment
    for hex_vertices in hexagons:
        if not check_containment(hex_vertices, outer_hex_vertices):
            total_penalty += 1000000  # Large penalty for containment violation
    
    # Check overlaps
    for i in range(11):
        for j in range(i+1, 11):
            if check_overlap(hexagons[i], hexagons[j]):
                total_penalty += 1000000  # Large penalty for overlap
    
    # If valid, return negative area (we want to minimize outer radius)
    if total_penalty == 0:
        # Calculate actual area of outer hexagon (smaller is better)
        area = 3 * np.sqrt(3) * outer_radius * outer_radius / 2
        return -area
    else:
        return total_penalty

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining geometric constraints with optimization.
    """
    # Start with a better initial configuration based on known good arrangements
    # This uses a more efficient arrangement than the grid-based one
    initial_positions = np.array([
        [0, 0],      # center
        [0, 2],      # top
        [0, -2],     # bottom  
        [1.732, 1],  # top right
        [-1.732, 1], # top left
        [1.732, -1], # bottom right
        [-1.732, -1],# bottom left
        [3.464, 0],  # far right
        [-3.464, 0], # far left
        [1.732, 3],  # top far right
        [-1.732, 3]  # top far left
    ])
    
    # Initial rotations (all 0 for simplicity)
    initial_rotations = np.zeros(11)
    
    # Combine into single configuration vector
    initial_config = np.concatenate([initial_positions.flatten(), initial_rotations])
    
    # Use optimization to improve the configuration
    # We'll try to minimize the outer hexagon radius
    best_radius = 5.0  # Initial guess
    best_config = initial_config.copy()
    
    # Try several optimization approaches
    for attempt in range(3):
        # Define bounds for optimization
        bounds = []
        # Positions: x and y coordinates
        for i in range(22):
            bounds.append((-10, 10))  # Reasonable bounds for positions
        # Rotations: 0-360 degrees
        for i in range(11):
            bounds.append((0, 360))
        
        # Simple local optimization
        try:
            # Use a coarse optimization first
            result = minimize(
                lambda x: evaluate_configuration(x, best_radius),
                initial_config,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            
            if result.success:
                # Test the result
                penalty = evaluate_configuration(result.x, best_radius)
                if penalty < 0:  # Valid configuration
                    # Update best if improved
                    if penalty < evaluate_configuration(best_config, best_radius):
                        best_config = result.x.copy()
        except:
            pass
            
        # Refine with a better starting point
        # Adjust positions slightly for better packing
        adjusted_positions = initial_positions.copy()
        adjusted_positions[0] = [0, 0]  # Center stays fixed
        adjusted_positions[1] = [0, 1.8]  # Top
        adjusted_positions[2] = [0, -1.8]  # Bottom
        adjusted_positions[3] = [1.5, 0.8]  # Top right
        adjusted_positions[4] = [-1.5, 0.8]  # Top left
        adjusted_positions[5] = [1.5, -0.8]  # Bottom right
        adjusted_positions[6] = [-1.5, -0.8]  # Bottom left
        adjusted_positions[7] = [3, 0]  # Far right
        adjusted_positions[8] = [-3, 0]  # Far left
        adjusted_positions[9] = [1.5, 2.5]  # Top far right
        adjusted_positions[10] = [-1.5, 2.5]  # Top far left
        
        # Update config with adjusted positions
        initial_config[:22] = adjusted_positions.flatten()
        
        # Try again with better initial guess
        try:
            result = minimize(
                lambda x: evaluate_configuration(x, best_radius),
                initial_config,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if result.success:
                penalty = evaluate_configuration(result.x, best_radius)
                if penalty < 0 and penalty < evaluate_configuration(best_config, best_radius):
                    best_config = result.x.copy()
        except:
            pass
    
    # Extract final configuration
    final_positions = best_config[:22].reshape(-1, 2)
    final_rotations = best_config[22:]
    
    # Determine optimal outer radius by finding the minimum that works
    # Binary search approach
    min_radius = 3.0
    max_radius = 6.0
    
    # Find the smallest valid radius
    test_radius = 4.0
    while max_radius - min_radius > 0.01:
        penalty = evaluate_configuration(best_config, test_radius)
        if penalty < 0:  # Valid configuration
            max_radius = test_radius
            test_radius = (min_radius + test_radius) / 2
        else:
            min_radius = test_radius
            test_radius = (test_radius + max_radius) / 2
    
    # Final refinement with the best found radius
    final_radius = max_radius
    
    # Create the final data structures
    inner_hex_data = np.column_stack([final_positions, final_rotations])
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    outer_hex_side_length = final_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
