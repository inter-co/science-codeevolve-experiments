# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math


def hexagon_vertices(center_x, center_y, side_length=1, angle_degrees=0):
    """Generate vertices of a regular hexagon."""
    angle_rad = math.radians(angle_degrees)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append([x, y])
    return np.array(vertices)


def hexagon_contains_point(hex_center_x, hex_center_y, hex_side_length, point_x, point_y):
    """Check if a point is inside a hexagon using distance to edges."""
    # Hexagon vertices
    vertices = hexagon_vertices(hex_center_x, hex_center_y, hex_side_length)
    
    # Check if point is inside using winding number or ray casting
    # For simplicity, we'll check distance to center and compare to radius
    center_dist = math.sqrt((point_x - hex_center_x)**2 + (point_y - hex_center_y)**2)
    # Radius of inscribed circle for regular hexagon is sqrt(3)/2 * side_length
    inscribed_radius = (math.sqrt(3) / 2) * hex_side_length
    
    # More precise check: point is inside if it's closer than the inscribed radius
    # But we need to also verify it's not on the wrong side of any edge
    # Using a simpler approach: if distance to center < inscribed_radius, 
    # and all edge distances are positive, then inside
    return center_dist <= inscribed_radius


def hexagon_overlap(h1_center_x, h1_center_y, h1_angle, h2_center_x, h2_center_y, h2_angle):
    """Check if two hexagons overlap using vertex inclusion."""
    # Get vertices for both hexagons
    v1 = hexagon_vertices(h1_center_x, h1_center_y, 1, h1_angle)
    v2 = hexagon_vertices(h2_center_x, h2_center_y, 1, h2_angle)
    
    # Check if any vertex of hexagon 1 is inside hexagon 2
    for vx, vy in v1:
        # Check if point is inside hexagon 2
        if hexagon_contains_point(h2_center_x, h2_center_y, 1, vx, vy):
            return True
    
    # Check if any vertex of hexagon 2 is inside hexagon 1  
    for vx, vy in v2:
        # Check if point is inside hexagon 1
        if hexagon_contains_point(h1_center_x, h1_center_y, 1, vx, vy):
            return True
            
    return False


def compute_outer_hexagon_bound(inner_hex_data, outer_hex_side_length_guess):
    """Compute the minimum bounding hexagon that contains all inner hexagons."""
    # For this simplified approach, we'll just compute the max distance from center
    # and use that to estimate the outer hexagon size
    max_dist = 0
    for i in range(len(inner_hex_data)):
        x, y, _ = inner_hex_data[i]
        dist = math.sqrt(x*x + y*y)
        max_dist = max(max_dist, dist + 1)  # +1 for hexagon radius
        
    # Estimate outer hexagon side length based on the maximum distance
    # For a regular hexagon, if we know the circumradius, we can compute side length
    # Circumradius = side_length for regular hexagon
    # So we need to scale our guess to fit all hexagons
    return max_dist * 1.1  # Add some margin


def evaluate_arrangement(inner_hex_data, outer_side_length):
    """Evaluate if arrangement is valid and return penalty if invalid."""
    # Check all pairwise overlaps
    penalty = 0
    n = len(inner_hex_data)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, a1 = inner_hex_data[i]
            x2, y2, a2 = inner_hex_data[j]
            
            if hexagon_overlap(x1, y1, a1, x2, y2, a2):
                penalty += 1000  # Strong penalty for overlaps
                
    # Check containment - all vertices of all inner hexagons must be inside outer hexagon
    outer_vertices = hexagon_vertices(0, 0, outer_side_length)
    
    for i in range(n):
        x, y, a = inner_hex_data[i]
        vertices = hexagon_vertices(x, y, 1, a)
        
        for vx, vy in vertices:
            # Simple containment check: point should be within distance
            dist_from_origin = math.sqrt(vx*vx + vy*vy)
            if dist_from_origin > outer_side_length:
                penalty += 1000  # Penalty for containment violation
    
    return penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initial configuration - more sophisticated than simple grid
    # Try a honeycomb-like arrangement around a central hexagon
    initial_positions = [
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [0, -2, 0],     # bottom  
        [1.732, 1, 0],  # top-right (sqrt(3) = 1.732)
        [-1.732, 1, 0], # top-left
        [1.732, -1, 0], # bottom-right
        [-1.732, -1, 0],# bottom-left
        [3.464, 0, 0],  # far right (2*sqrt(3))
        [-3.464, 0, 0], # far left
        [1.732, 3, 0],  # top-top-right
        [-1.732, 3, 0], # top-top-left
    ]
    
    # Convert to numpy array
    initial_hex_data = np.array(initial_positions)
    
    # Optimization approach: use scipy to minimize outer hexagon size
    def objective(outer_size):
        # Evaluate this configuration with given outer size
        penalty = evaluate_arrangement(initial_hex_data, outer_size)
        # Return negative of 1/outer_size since we want to maximize 1/outer_size
        # But we're minimizing so we return penalty + 1/outer_size
        return penalty + 1.0 / outer_size if outer_size > 0 else 1000000
    
    # Try to find better arrangement using optimization
    # Start with the initial guess
    initial_outer_size = 5.0  # reasonable starting point
    
    # Use a simple optimization approach
    try:
        # We'll do a coarse search first
        best_size = initial_outer_size
        best_penalty = evaluate_arrangement(initial_hex_data, best_size)
        
        # Try several values to find good starting point
        test_sizes = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        for size in test_sizes:
            penalty = evaluate_arrangement(initial_hex_data, size)
            if penalty < best_penalty:
                best_penalty = penalty
                best_size = size
                
        # Now refine with a simple gradient-free search
        step = 0.1
        current_size = best_size
        for _ in range(50):  # iterations
            # Try smaller size
            new_size = current_size - step
            if new_size > 0:
                penalty = evaluate_arrangement(initial_hex_data, new_size)
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_size = new_size
                    current_size = new_size
                else:
                    # Try larger size
                    new_size = current_size + step
                    penalty = evaluate_arrangement(initial_hex_data, new_size)
                    if penalty < best_penalty:
                        best_penalty = penalty
                        best_size = new_size
                        current_size = new_size
            else:
                break
                
        # If we found a better solution, return it
        if best_penalty < 1000:
            final_size = best_size
        else:
            final_size = initial_outer_size
            
    except Exception:
        final_size = initial_outer_size
    
    # Final result
    inner_hex_data = initial_hex_data.copy()
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = final_size
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
