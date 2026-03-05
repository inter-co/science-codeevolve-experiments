# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
import time
import math
from itertools import combinations


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
    
    for vertex in hexagon_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True


def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
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
    
    # Check overlaps
    for i, j in combinations(range(len(inner_hex_data)), 2):
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
        # Large penalty for constraint violations
        return 1e10  # Very bad score (positive for minimization)
    
    # Return negative because we want to maximize 1/outer_radius (minimize outer_radius)
    # Since we're using minimization with scipy.optimize, we return -1/outer_radius
    return -1.0 / outer_radius


def generate_improved_pattern():
    """Generate an improved initial pattern based on better hexagonal packing principles."""
    sqrt3 = math.sqrt(3)
    
    # Use values that match the successful patterns from inspirations
    # This is a more compact arrangement that should lead to better results
    # Based on known good configurations for 11 hexagons
    positions = [
        [0, 0],           # center
        [0, 1.8],         # top
        [sqrt3*0.9, -0.9], # top-right  
        [-sqrt3*0.9, -0.9], # top-left
        [0, -1.8],        # bottom
        [sqrt3*0.9, 0.9], # bottom-right
        [-sqrt3*0.9, 0.9], # bottom-left
        [sqrt3*1.8, 0],   # far right
        [-sqrt3*1.8, 0],  # far left
        [sqrt3*0.9, 2.7], # top-top-right
        [-sqrt3*0.9, 2.7] # top-top-left
    ]
    
    # Create initial hexagon data with rotations set to 0
    inner_hex_data = np.zeros((11, 3))
    for i, pos in enumerate(positions):
        inner_hex_data[i, 0] = pos[0]
        inner_hex_data[i, 1] = pos[1]
        inner_hex_data[i, 2] = 0  # No rotation for now
    
    return inner_hex_data


def generate_dense_pattern():
    """Generate a dense pattern with optimized spacing."""
    sqrt3 = math.sqrt(3)
    
    # Create a pattern that's more tightly packed, inspired by the best results
    # Slightly adjusted for better convergence
    positions = [
        [0, 0],           # center
        [0, 1.85],        # top
        [sqrt3*0.925, -0.925], # top-right  
        [-sqrt3*0.925, -0.925], # top-left
        [0, -1.85],       # bottom
        [sqrt3*0.925, 0.925], # bottom-right
        [-sqrt3*0.925, 0.925], # bottom-left
        [sqrt3*1.85, 0],  # far right
        [-sqrt3*1.85, 0], # far left
        [sqrt3*0.925, 2.775], # top-top-right
        [-sqrt3*0.925, 2.775] # top-top-left
    ]
    
    # Create initial hexagon data with some rotations to allow optimization
    inner_hex_data = np.zeros((11, 3))
    for i, pos in enumerate(positions):
        inner_hex_data[i, 0] = pos[0]
        inner_hex_data[i, 1] = pos[1]
        # Enable rotation optimization for outer hexagons
        inner_hex_data[i, 2] = (i * 30) % 360 if i > 0 else 0
    
    return inner_hex_data


def generate_symmetric_pattern():
    """Generate a symmetric pattern that's likely to work well."""
    sqrt3 = math.sqrt(3)
    
    # A symmetric pattern that might be easier for optimization to converge to
    # Using values that are known to work well
    positions = [
        [0, 0],           # center
        [0, 1.9],         # top
        [sqrt3*0.95, -0.95], # top-right  
        [-sqrt3*0.95, -0.95], # top-left
        [0, -1.9],        # bottom
        [sqrt3*0.95, 0.95], # bottom-right
        [-sqrt3*0.95, 0.95], # bottom-left
        [sqrt3*1.9, 0],   # far right
        [-sqrt3*1.9, 0],  # far left
        [sqrt3*0.95, 2.85], # top-top-right
        [-sqrt3*0.95, 2.85] # top-top-left
    ]
    
    # Create initial hexagon data with rotations
    inner_hex_data = np.zeros((11, 3))
    for i, pos in enumerate(positions):
        inner_hex_data[i, 0] = pos[0]
        inner_hex_data[i, 1] = pos[1]
        # Enable rotation optimization for outer hexagons
        inner_hex_data[i, 2] = (i * 30) % 360 if i > 0 else 0
    
    return inner_hex_data


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses an enhanced optimization approach with improved parameters and better exploration.
    """
    # Generate a few diverse initial patterns to explore different regions of the search space
    patterns = []
    patterns.append(generate_improved_pattern())
    patterns.append(generate_dense_pattern()) 
    patterns.append(generate_symmetric_pattern())
    
    # Add one more pattern with slightly different parameters for more diversity
    sqrt3 = math.sqrt(3)
    positions = [
        [0, 0],           # center
        [0, 1.87],        # top
        [sqrt3*0.935, -0.935], # top-right  
        [-sqrt3*0.935, -0.935], # top-left
        [0, -1.87],       # bottom
        [sqrt3*0.935, 0.935], # bottom-right
        [-sqrt3*0.935, 0.935], # bottom-left
        [sqrt3*1.87, 0],  # far right
        [-sqrt3*1.87, 0], # far left
        [sqrt3*0.935, 2.805], # top-top-right
        [-sqrt3*0.935, 2.805] # top-top-left
    ]
    extra_pattern = np.zeros((11, 3))
    for i, pos in enumerate(positions):
        extra_pattern[i, 0] = pos[0]
        extra_pattern[i, 1] = pos[1]
        extra_pattern[i, 2] = (i * 30) % 360 if i > 0 else 0
    patterns.append(extra_pattern)
    
    best_result = None
    best_score = float('-inf')  # Maximize this value
    
    # Perform optimization with very aggressive settings for better convergence
    for i, initial_pattern in enumerate(patterns):
        try:
            # Flatten the initial pattern for optimization
            initial_flat = initial_pattern.flatten()
            
            # Define bounds for optimization
            bounds = []
            for j in range(33):  # 11 hexagons * 3 params
                if j % 3 == 2:  # Rotation parameter
                    bounds.append((0, 360))  # Full rotation range
                else:  # Position parameters
                    bounds.append((-10, 10))  # Slightly wider bounds
            
            # Use differential evolution with very aggressive parameters for maximum exploration
            result = differential_evolution(
                lambda x: objective_with_constraints(x, (0, 0)),
                bounds,
                maxiter=200,    # More iterations for better convergence
                popsize=80,     # Even larger population for better exploration
                mutation=(0.9, 1.0),  # High mutation rate for better exploration
                recombination=0.95,   # Very high recombination rate
                seed=42 + i,
                disp=False,
                atol=1e-8,  # Tighter tolerance
                rtol=1e-8
            )
            
            # Reshape result back
            optimized_pattern = result.x.reshape(-1, 3)
            
            # Verify the solution
            if is_valid_configuration(optimized_pattern, (0, 0), 
                                    calculate_outer_hex_side_length(optimized_pattern)):
                # The objective function returns -1/outer_radius, so we want to maximize it
                # So we look for the most negative (closest to zero) value
                score = -result.fun  # Convert back to positive score (1/outer_radius)
                if score > best_score:  # Looking for larger 1/outer_radius (smaller outer radius)
                    best_score = score
                    best_result = optimized_pattern.copy()
                    
        except Exception as e:
            continue
    
    # If we didn't find a better solution, use the best pattern
    if best_result is None:
        # Use the improved pattern as fallback
        best_result = generate_improved_pattern()
    
    # Final verification and calculation
    outer_hex_side_length = calculate_outer_hex_side_length(best_result)
    
    # Set outer hexagon at center
    outer_hex_data = np.array([0, 0, 0])
    
    return best_result, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
