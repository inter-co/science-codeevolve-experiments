# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from itertools import combinations
import time

def create_unit_hexagon(center=(0,0), rotation=0):
    """Create a unit regular hexagon centered at center with given rotation."""
    angle = rotation * np.pi / 180
    # Vertices of unit hexagon centered at origin
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        vertices.append((x + center[0], y + center[1]))
    return Polygon(vertices)

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hexagon.contains(hexagon)

def check_disjoint(hex1, hex2):
    """Check if two hexagons are disjoint (no overlap)."""
    return not hex1.intersects(hex2)

def evaluate_packing(inner_positions, inner_rotations, outer_center=(0,0), outer_rotation=0):
    """Evaluate a packing configuration and return penalty if constraints violated."""
    # Create outer hexagon (will be optimized to minimal size)
    # For now, we'll use a placeholder - actual optimization will adjust this
    
    # Create inner hexagons
    inner_hexagons = []
    for pos, rot in zip(inner_positions, inner_rotations):
        hexagon = create_unit_hexagon(pos, rot)
        inner_hexagons.append(hexagon)
    
    # Check for overlaps
    penalties = 0
    for i, j in combinations(range(len(inner_hexagons)), 2):
        if not check_disjoint(inner_hexagons[i], inner_hexagons[j]):
            penalties += 1000  # Large penalty for overlaps
    
    # Check containment
    # We need to define outer hexagon - for now, we'll compute the bounding box
    if len(inner_hexagons) > 0:
        # Get all vertices of all inner hexagons
        all_vertices = []
        for hexagon in inner_hexagons:
            all_vertices.extend(list(hexagon.exterior.coords))
        
        # Find bounding box
        min_x = min(v[0] for v in all_vertices)
        max_x = max(v[0] for v in all_vertices)
        min_y = min(v[1] for v in all_vertices)
        max_y = max(v[1] for v in all_vertices)
        
        # Calculate approximate outer hexagon radius needed
        # This is a rough estimate - more precise would require proper hexagon fitting
        outer_radius = max(abs(max_x - min_x), abs(max_y - min_y)) / 2
        
        # Return negative penalty (since we want to minimize) plus the area of the outer hexagon
        # Area of outer hexagon = 3*sqrt(3)/2 * R^2 where R is distance from center to corner
        # But we're optimizing for the side length, which is equal to R for regular hexagon
        outer_side_length = outer_radius * 2 / np.sqrt(3)  # Approximate conversion
        return penalties + outer_side_length * 100  # Penalty for overlaps + size term
    
    return 10000  # Very high penalty if no hexagons

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a symmetry-aware optimization approach to find near-optimal arrangements.
    """
    
    # Initial configuration based on known good symmetric patterns
    # Using 12 hexagons arranged in a pattern inspired by close packing
    initial_positions = [
        (0, 0),           # Center
        (0, 2),           # Top
        (0, -2),          # Bottom
        (1.732, 1),       # Top right
        (-1.732, 1),      # Top left
        (1.732, -1),      # Bottom right
        (-1.732, -1),     # Bottom left
        (3.464, 0),       # Far right
        (-3.464, 0),      # Far left
        (1.732, 3),       # Top far right
        (-1.732, 3),      # Top far left
        (1.732, -3),      # Bottom far right
        (-1.732, -3),     # Bottom far left
    ]
    
    # Initial rotations (all 0 degrees for simplicity in starting point)
    initial_rotations = [0] * 12
    
    # Optimize using scipy minimize
    # We'll parameterize by positions and rotations, then optimize
    def objective(x):
        # x contains [pos_x0, pos_y0, rot0, pos_x1, pos_y1, rot1, ...]
        positions = [(x[3*i], x[3*i+1]) for i in range(12)]
        rotations = [x[3*i+2] for i in range(12)]
        
        # Calculate the effective outer hexagon size
        # First, let's get the actual minimum bounding hexagon
        inner_hexagons = []
        for pos, rot in zip(positions, rotations):
            hexagon = create_unit_hexagon(pos, rot)
            inner_hexagons.append(hexagon)
            
        # Compute the convex hull of all vertices to estimate outer bounds
        all_vertices = []
        for hexagon in inner_hexagons:
            all_vertices.extend(list(hexagon.exterior.coords))
        
        if len(all_vertices) < 12:  # Not enough points
            return 10000
            
        # Get extreme points
        min_x = min(v[0] for v in all_vertices)
        max_x = max(v[0] for v in all_vertices)
        min_y = min(v[1] for v in all_vertices)
        max_y = max(v[1] for v in all_vertices)
        
        # Estimate outer hexagon side length based on bounding box
        width = max_x - min_x
        height = max_y - min_y
        # For a regular hexagon, the relationship between width/height and side length
        # is complex, so we'll use a conservative estimate
        estimated_side = max(width, height) / 2
        
        # Check overlaps
        penalty = 0
        for i, j in combinations(range(len(inner_hexagons)), 2):
            if inner_hexagons[i].intersects(inner_hexagons[j]):
                penalty += 1000000
                
        return estimated_side + penalty
    
    # Flatten initial parameters
    initial_params = []
    for i in range(12):
        initial_params.extend([initial_positions[i][0], initial_positions[i][1], initial_rotations[i]])
    
    # Optimization bounds (positions in reasonable range, rotations 0-360)
    bounds = []
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    
    # Use scipy optimization
    try:
        # Run optimization with L-BFGS-B method
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        # Extract results
        final_positions = [(result.x[3*i], result.x[3*i+1]) for i in range(12)]
        final_rotations = [result.x[3*i+2] for i in range(12)]
        outer_side_length = result.fun
        
        # Ensure rotations are within 0-360 range
        final_rotations = [r % 360 for r in final_rotations]
        
    except Exception as e:
        # Fallback to simple symmetric arrangement
        print(f"Optimization failed: {e}")
        final_positions = initial_positions
        final_rotations = initial_rotations
        outer_side_length = 4.0  # Conservative estimate
    
    # Convert to the required format
    inner_hex_data = np.array([
        [final_positions[i][0], final_positions[i][1], final_rotations[i]] 
        for i in range(12)
    ])
    
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    # Refine the outer hexagon side length calculation
    # Create the hexagons and find the actual minimum enclosing hexagon
    try:
        inner_hexagons = []
        for pos, rot in zip(final_positions, final_rotations):
            hexagon = create_unit_hexagon(pos, rot)
            inner_hexagons.append(hexagon)
            
        # Get all vertices
        all_vertices = []
        for hexagon in inner_hexagons:
            all_vertices.extend(list(hexagon.exterior.coords))
        
        # Find bounding circle
        if len(all_vertices) >= 2:
            # Center of mass
            cx = sum(v[0] for v in all_vertices) / len(all_vertices)
            cy = sum(v[1] for v in all_vertices) / len(all_vertices)
            
            # Maximum distance from center
            max_dist = max(np.sqrt((v[0]-cx)**2 + (v[1]-cy)**2) for v in all_vertices)
            
            # For a regular hexagon, if we know the circumradius (max_dist),
            # the side length is also max_dist
            outer_side_length = max_dist
            
    except:
        pass  # Keep previous estimate
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
