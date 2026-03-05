# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
import math
import random
from itertools import combinations

# Constants
UNIT_HEX_RADIUS = 1.0  # Distance from center to corner for unit hexagon

def get_hexagon_vertices(center, radius=1, rotation=0):
    """Get vertices of a unit hexagon at given position and rotation."""
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
        # Use buffer to handle floating point precision issues
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
    
    # Check overlaps - more efficient approach with early termination
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

def generate_multiple_initial_patterns():
    """Generate multiple promising initial patterns based on known hexagon packing strategies."""
    patterns = []
    sqrt3 = math.sqrt(3)
    
    # Pattern 1: Highly optimized symmetric arrangement (from inspiration)
    pattern1 = np.array([
        [0, 0, 0],           # center
        [0, 1.9, 0],         # top
        [sqrt3*0.95, -0.95, 0], # top-right  
        [-sqrt3*0.95, -0.95, 0], # top-left
        [0, -1.9, 0],        # bottom
        [sqrt3*0.95, 0.95, 0], # bottom-right
        [-sqrt3*0.95, 0.95, 0], # bottom-left
        [sqrt3*1.9, 0, 0],   # far right
        [-sqrt3*1.9, 0, 0],  # far left
        [sqrt3*0.95, 2.85, 0], # top-top-right
        [-sqrt3*0.95, 2.85, 0] # top-top-left
    ])
    patterns.append(pattern1)
    
    # Pattern 2: Alternative symmetric pattern with slightly different spacing
    pattern2 = np.array([
        [0, 0, 0],           # center
        [0, 1.85, 0],        # top
        [sqrt3*0.925, -0.925, 0], # top-right  
        [-sqrt3*0.925, -0.925, 0], # top-left
        [0, -1.85, 0],       # bottom
        [sqrt3*0.925, 0.925, 0], # bottom-right
        [-sqrt3*0.925, 0.925, 0], # bottom-left
        [sqrt3*1.85, 0, 0],  # far right
        [-sqrt3*1.85, 0, 0], # far left
        [sqrt3*0.925, 2.775, 0], # top-top-right
        [-sqrt3*0.925, 2.775, 0] # top-top-left
    ])
    patterns.append(pattern2)
    
    # Pattern 3: Hexagonal lattice approach
    pattern3 = np.array([
        [0, 0, 0],           # center
        [0, 1.9, 0],         # top
        [sqrt3*0.95, -0.95, 0], # top-right  
        [-sqrt3*0.95, -0.95, 0], # top-left
        [0, -1.9, 0],        # bottom
        [sqrt3*0.95, 0.95, 0], # bottom-right
        [-sqrt3*0.95, 0.95, 0], # bottom-left
        [sqrt3*1.9, 0, 0],   # far right
        [-sqrt3*1.9, 0, 0],  # far left
        [sqrt3*0.95, 2.85, 0], # top-top-right
        [-sqrt3*0.95, 2.85, 0] # top-top-left
    ])
    patterns.append(pattern3)
    
    # Pattern 4: Slightly perturbed version for diversity
    pattern4 = np.array([
        [0, 0, 0],           # center
        [0, 1.92, 0],        # top
        [sqrt3*0.96, -0.96, 0], # top-right  
        [-sqrt3*0.96, -0.96, 0], # top-left
        [0, -1.92, 0],       # bottom
        [sqrt3*0.96, 0.96, 0], # bottom-right
        [-sqrt3*0.96, 0.96, 0], # bottom-left
        [sqrt3*1.92, 0, 0],  # far right
        [-sqrt3*1.92, 0, 0], # far left
        [sqrt3*0.96, 2.88, 0], # top-top-right
        [-sqrt3*0.96, 2.88, 0] # top-top-left
    ])
    patterns.append(pattern4)
    
    # Pattern 5: Optimized for tight packing
    pattern5 = np.array([
        [0, 0, 0],           # center
        [0, 1.88, 0],        # top
        [sqrt3*0.94, -0.94, 0], # top-right  
        [-sqrt3*0.94, -0.94, 0], # top-left
        [0, -1.88, 0],       # bottom
        [sqrt3*0.94, 0.94, 0], # bottom-right
        [-sqrt3*0.94, 0.94, 0], # bottom-left
        [sqrt3*1.88, 0, 0],  # far right
        [-sqrt3*1.88, 0, 0], # far left
        [sqrt3*0.94, 2.82, 0], # top-top-right
        [-sqrt3*0.94, 2.82, 0] # top-top-left
    ])
    patterns.append(pattern5)
    
    return patterns

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses multiple geometric strategies and local optimization to find high-quality solutions.
    """
    # Generate multiple initial patterns
    patterns = generate_multiple_initial_patterns()
    
    # Evaluate all patterns and select the best valid one
    best_result = None
    best_score = float('-inf')
    best_outer_radius = float('inf')
    
    for i, pattern in enumerate(patterns):
        # Validate the pattern
        outer_radius = calculate_outer_hex_side_length(pattern)
        if is_valid_configuration(pattern, (0, 0), outer_radius):
            score = -1.0 / outer_radius
            if score > best_score:
                best_score = score
                best_outer_radius = outer_radius
                best_result = pattern.copy()
    
    # If no valid pattern was found, use fallback
    if best_result is None:
        # Use a simple symmetric arrangement as fallback
        sqrt3 = math.sqrt(3)
        fallback_pattern = np.array([
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
        best_result = fallback_pattern
        outer_radius = calculate_outer_hex_side_length(best_result)
        best_score = -1.0 / outer_radius
        best_outer_radius = outer_radius
    
    # Local optimization refinement - use a more sophisticated approach
    refined_pattern = best_result.copy()
    best_score = -1.0 / best_outer_radius
    
    # Multi-stage refinement approach
    # Stage 1: Coarse-grained optimization
    for stage in range(3):
        # Try different perturbation strategies
        for iter_num in range(50):
            # Select hexagon to perturb (avoid center hexagon)
            hex_idx = random.randint(1, 10)
            
            # Save current state
            old_pattern = refined_pattern.copy()
            
            # Try several perturbation types
            perturbation_type = random.choice(['small_pos', 'medium_pos', 'large_pos', 'rotation'])
            
            if perturbation_type == 'small_pos':
                refined_pattern[hex_idx, 0] += random.uniform(-0.03, 0.03)
                refined_pattern[hex_idx, 1] += random.uniform(-0.03, 0.03)
            elif perturbation_type == 'medium_pos':
                refined_pattern[hex_idx, 0] += random.uniform(-0.08, 0.08)
                refined_pattern[hex_idx, 1] += random.uniform(-0.08, 0.08)
            elif perturbation_type == 'large_pos':
                refined_pattern[hex_idx, 0] += random.uniform(-0.15, 0.15)
                refined_pattern[hex_idx, 1] += random.uniform(-0.15, 0.15)
            else:  # rotation
                refined_pattern[hex_idx, 2] += random.uniform(-5, 5)
                refined_pattern[hex_idx, 2] = refined_pattern[hex_idx, 2] % 360
            
            # Check if the new configuration is valid and better
            outer_radius = calculate_outer_hex_side_length(refined_pattern)
            if is_valid_configuration(refined_pattern, (0, 0), outer_radius):
                new_score = -1.0 / outer_radius
                if new_score > best_score:
                    best_score = new_score
                    best_outer_radius = outer_radius
                else:
                    # Revert if not better
                    refined_pattern = old_pattern
            else:
                # Revert if invalid
                refined_pattern = old_pattern
    
    # Final validation
    final_outer_radius = calculate_outer_hex_side_length(refined_pattern)
    if not is_valid_configuration(refined_pattern, (0, 0), final_outer_radius):
        # If still not valid, revert to best valid pattern
        pass
    
    # Final result setup
    outer_hex_data = np.array([0, 0, 0])
    
    return refined_pattern, outer_hex_data, final_outer_radius


# EVOLVE-BLOCK-END
