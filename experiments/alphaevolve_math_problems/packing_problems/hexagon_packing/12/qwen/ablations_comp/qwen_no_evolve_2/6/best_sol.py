# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
import math

def generate_hexagon_vertices(center_x, center_y, side_length=1, rotation_deg=0):
    """Generate vertices of a regular hexagon given center, side length, and rotation."""
    rotation_rad = math.radians(rotation_deg)
    angle_step = 2 * math.pi / 6
    vertices = []
    for i in range(6):
        angle = angle_step * i + rotation_rad
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def check_hexagon_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if a hexagon is fully contained within the outer hexagon."""
    inner_poly = Polygon(hexagon_vertices)
    outer_poly = Polygon(outer_hexagon_vertices)
    
    # Check if inner polygon is completely inside outer polygon
    return outer_poly.contains(inner_poly)

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    
    # Check if polygons intersect
    return poly1.intersects(poly2)

def compute_outer_hexagon_radius(inner_hex_data, side_length=1):
    """Compute the minimum radius needed for outer hexagon to contain all inner hexagons."""
    max_distance = 0
    
    # Get all vertices of all inner hexagons
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, side_length, rotation)
        
        # Find maximum distance from origin to any vertex
        for vx, vy in vertices:
            distance = math.sqrt(vx*vx + vy*vy)
            max_distance = max(max_distance, distance)
    
    # Add buffer to ensure full containment (accounting for hexagon size)
    # For a unit hexagon, the circumradius is 1, so we need to account for the outer hexagon's circumradius
    # The outer hexagon needs to have radius >= max_distance + 1 (to fully contain the unit hexagon)
    return max_distance + 1.0

def evaluate_packing(inner_hex_data, side_length=1):
    """Evaluate if a packing configuration is valid and return performance metrics."""
    n = len(inner_hex_data)
    
    # Generate all hexagon vertices
    hexagon_vertices_list = []
    for i in range(n):
        center_x, center_y, rotation = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, side_length, rotation)
        hexagon_vertices_list.append(vertices)
    
    # Check for overlaps
    for i in range(n):
        for j in range(i+1, n):
            if check_hexagon_overlap(hexagon_vertices_list[i], hexagon_vertices_list[j]):
                return False, float('inf'), 0
    
    # Compute outer hexagon radius
    outer_radius = compute_outer_hexagon_radius(inner_hex_data, side_length)
    
    # Create outer hexagon vertices (circumradius = outer_radius)
    outer_vertices = generate_hexagon_vertices(0, 0, outer_radius, 0)
    
    # Check containment
    for i in range(n):
        if not check_hexagon_containment(hexagon_vertices_list[i], outer_vertices):
            return False, float('inf'), 0
    
    # Return inverse of outer radius (objective function)
    return True, 1.0 / outer_radius, outer_radius

def generate_symmetric_arrangement():
    """Generate a symmetric arrangement of 12 hexagons based on known optimal patterns."""
    # This is inspired by the known optimal solution for 12 hexagons
    # Based on research, an optimal arrangement uses a pattern with rotational symmetry
    
    # Create a configuration that's more optimized than the initial grid
    arrangement = []
    
    # Central hexagon
    arrangement.append([0, 0, 0])
    
    # Layer 1: 6 hexagons around the center
    angles = [0, 60, 120, 180, 240, 300]
    for angle in angles:
        rad = 1.732  # approximately sqrt(3) which is the distance between centers in optimal packing
        x = rad * math.cos(math.radians(angle))
        y = rad * math.sin(math.radians(angle))
        arrangement.append([x, y, 0])
    
    # Layer 2: 5 hexagons forming a ring
    # These are positioned to maximize packing efficiency
    angles2 = [30, 90, 150, 210, 270]
    for angle in angles2:
        rad = 3.464  # approximately 2*sqrt(3)
        x = rad * math.cos(math.radians(angle))
        y = rad * math.sin(math.radians(angle))
        arrangement.append([x, y, 0])
    
    # One more hexagon in the center of the outer layer
    arrangement.append([0, 3.464, 0])
    
    # Adjust to ensure exactly 12 hexagons
    return np.array(arrangement[:12])

def optimize_hexagon_packing():
    """Use a hybrid optimization approach to find better packing."""
    # Start with a symmetric arrangement
    best_inner_hex_data = generate_symmetric_arrangement()
    best_inv_radius, _, _ = evaluate_packing(best_inner_hex_data)
    
    # Try some variations to improve the solution
    for iteration in range(1000):  # Limited iterations for performance
        # Create small perturbations
        new_data = best_inner_hex_data.copy()
        
        # Randomly select one hexagon to perturb
        idx = np.random.randint(0, len(new_data))
        
        # Apply small random perturbations
        new_data[idx][0] += np.random.uniform(-0.1, 0.1)
        new_data[idx][1] += np.random.uniform(-0.1, 0.1)
        
        # Check if this improves the solution
        valid, inv_radius, _ = evaluate_packing(new_data)
        if valid and inv_radius > best_inv_radius:
            best_inner_hex_data = new_data
            best_inv_radius = inv_radius
            
            # Early stopping if we're approaching the target
            if best_inv_radius > 0.2537:
                break
    
    # Final refinement
    final_data = best_inner_hex_data.copy()
    for _ in range(100):
        # Try to refine positions
        for i in range(len(final_data)):
            old_pos = final_data[i].copy()
            
            # Small random movement
            final_data[i][0] += np.random.uniform(-0.05, 0.05)
            final_data[i][1] += np.random.uniform(-0.05, 0.05)
            
            valid, inv_radius, _ = evaluate_packing(final_data)
            if not valid:
                final_data[i] = old_pos  # Revert if invalid
    
    return final_data

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use the optimization approach to get a better arrangement
    inner_hex_data = optimize_hexagon_packing()
    
    # Evaluate the final result
    valid, inv_radius, outer_radius = evaluate_packing(inner_hex_data)
    
    # Ensure we have a valid configuration
    if not valid:
        # Fallback to original configuration if optimization fails
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
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])
        valid, inv_radius, outer_radius = evaluate_packing(inner_hex_data)
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END
