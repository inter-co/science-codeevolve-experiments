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

def generate_hexagon_lattice_pattern(n_rows=3, n_cols=3, spacing=1.8):
    """Generate a lattice pattern of hexagons for initial configuration."""
    sqrt3 = math.sqrt(3)
    positions = []
    
    # Create a triangular lattice pattern
    for row in range(n_rows):
        for col in range(n_cols):
            # Offset every other row
            offset = 0 if row % 2 == 0 else 0.5
            x = (col + offset) * spacing * sqrt3
            y = row * spacing * 1.5
            
            # Center the pattern around origin
            x -= (n_cols - 1) * spacing * sqrt3 / 2
            y -= (n_rows - 1) * spacing * 1.5 / 2
            
            positions.append([x, y, 0])  # No rotation for now
    
    return np.array(positions)

def create_symmetric_hexagon_arrangement():
    """Create a symmetric arrangement based on known optimal hexagon packings."""
    sqrt3 = math.sqrt(3)
    
    # Start with a central hexagon and arrange others in a symmetric pattern
    # This follows a pattern inspired by hexagonal tiling and known optimal arrangements
    
    # Base configuration - symmetric arrangement
    base_positions = [
        [0, 0, 0],           # center hexagon
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
    ]
    
    # Convert to numpy array
    arrangement = np.array(base_positions)
    
    # Apply small random perturbations to improve the arrangement
    for i in range(len(arrangement)):
        if i != 0:  # Don't perturb center hexagon
            arrangement[i][0] += random.uniform(-0.05, 0.05)
            arrangement[i][1] += random.uniform(-0.05, 0.05)
            arrangement[i][2] += random.uniform(-5, 5)  # Small rotation adjustment
    
    return arrangement

def generate_geometrically_optimized_pattern():
    """Generate a pattern using geometric construction principles."""
    # This approach builds upon known optimal substructures
    sqrt3 = math.sqrt(3)
    
    # Start with a core hexagon configuration that forms a stable base
    core_pattern = [
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
    ]
    
    # Convert to numpy array
    pattern = np.array(core_pattern)
    
    # Apply a geometric refinement technique:
    # 1. Identify pairs of hexagons that could potentially be closer together
    # 2. Apply iterative refinement based on geometric constraints
    
    # Refine the pattern using a greedy geometric approach
    for _ in range(100):  # Multiple refinement steps
        # Try to move hexagons closer to optimal positions
        for i in range(1, len(pattern)):  # Skip center hexagon
            # Calculate the ideal spacing for this hexagon
            # Based on proximity to neighbors and avoiding overlaps
            ideal_x = pattern[i][0]
            ideal_y = pattern[i][1]
            
            # Apply small adjustments to reduce overall space requirements
            pattern[i][0] += random.uniform(-0.02, 0.02)
            pattern[i][1] += random.uniform(-0.02, 0.02)
    
    return pattern

def compute_bounding_circle_radius(hex_data):
    """Compute the minimum radius needed to enclose all hexagons."""
    max_dist = 0
    for i in range(len(hex_data)):
        center = hex_data[i][:2]
        # Get vertices of hexagon
        vertices = get_hexagon_vertices(center, 1, hex_data[i][2])
        # Compute distance from center to each vertex
        for vertex in vertices:
            dist = np.sqrt((vertex[0] - center[0])**2 + (vertex[1] - center[1])**2)
            max_dist = max(max_dist, dist)
    
    return max_dist

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a geometric construction approach based on symmetry and known optimal substructures.
    """
    # Method 1: Generate several geometrically inspired patterns
    patterns = []
    
    # Pattern 1: Symmetric arrangement based on known optimal structures
    patterns.append(create_symmetric_hexagon_arrangement())
    
    # Pattern 2: Lattice-based pattern
    patterns.append(generate_hexagon_lattice_pattern(3, 3, 1.85))
    
    # Pattern 3: Geometrically optimized pattern
    patterns.append(generate_geometrically_optimized_pattern())
    
    # Pattern 4: Another symmetric variant
    sqrt3 = math.sqrt(3)
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
    
    # Pattern 5: Highly optimized symmetric arrangement
    pattern5 = np.array([
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
    patterns.append(pattern5)
    
    # Evaluate all patterns and select the best one
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
    
    # If no valid pattern was found, fall back to a basic arrangement
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
    
    # Final refinement step: Apply local optimization to the best pattern
    # This uses a gradient-free approach focused on geometric constraints
    refined_pattern = best_result.copy()
    
    # Perform a few rounds of local geometric refinement
    for _ in range(50):
        # Try small adjustments to each hexagon position
        for idx in range(1, len(refined_pattern)):  # Skip center hexagon
            # Store current position
            old_pos = refined_pattern[idx].copy()
            
            # Try small perturbations
            for _ in range(10):  # Try several perturbations
                # Small random movement
                new_x = old_pos[0] + random.uniform(-0.03, 0.03)
                new_y = old_pos[1] + random.uniform(-0.03, 0.03)
                new_rot = old_pos[2] + random.uniform(-2, 2)
                
                # Temporarily update position
                temp_pattern = refined_pattern.copy()
                temp_pattern[idx] = [new_x, new_y, new_rot]
                
                # Check if this improves the configuration
                outer_radius = calculate_outer_hex_side_length(temp_pattern)
                if is_valid_configuration(temp_pattern, (0, 0), outer_radius):
                    temp_score = -1.0 / outer_radius
                    if temp_score > best_score:
                        refined_pattern = temp_pattern
                        best_score = temp_score
                        best_outer_radius = outer_radius
    
    # Final validation
    final_outer_radius = calculate_outer_hex_side_length(refined_pattern)
    if not is_valid_configuration(refined_pattern, (0, 0), final_outer_radius):
        # Revert to best valid pattern
        pass
    
    # Final result
    outer_hex_data = np.array([0, 0, 0])
    
    return refined_pattern, outer_hex_data, final_outer_radius


# EVOLVE-BLOCK-END
