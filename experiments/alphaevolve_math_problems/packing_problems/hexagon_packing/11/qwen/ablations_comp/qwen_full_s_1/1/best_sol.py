# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import time
import random
from scipy.optimize import differential_evolution

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = center_x + side_length * np.cos(theta)
        y = center_y + side_length * np.sin(theta)
        vertices.append((x, y))
    return vertices

def check_hexagon_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer hexagon"""
    try:
        hex_poly = Polygon(hexagon)
        outer_poly = Polygon(outer_hexagon)
        return outer_poly.contains(hex_poly)
    except:
        return False

def check_hexagon_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    try:
        poly1 = Polygon(hex1)
        poly2 = Polygon(hex2)
        return poly1.intersects(poly2)
    except:
        return True

def evaluate_packing(inner_hex_data, outer_hex_side_length):
    """Evaluate if a packing is valid and return penalty if invalid"""
    # Create outer hexagon
    outer_center = (0, 0)
    outer_hex = hexagon_vertices(outer_center[0], outer_center[1], 0, outer_hex_side_length)
    
    # Check if all inner hexagons are contained and non-overlapping
    total_penalty = 0
    
    inner_hexagons = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle_deg = inner_hex_data[i]
        vertices = hexagon_vertices(center_x, center_y, angle_deg)
        inner_hexagons.append(vertices)
        
        # Check containment
        if not check_hexagon_containment(vertices, outer_hex):
            total_penalty += 1000000  # Large penalty for containment violation
            
        # Check overlaps with other hexagons
        for j in range(i):
            if check_hexagon_overlap(vertices, inner_hexagons[j]):
                total_penalty += 1000000  # Large penalty for overlap
    
    return total_penalty

def binary_search_min_outer_side(inner_hex_data, max_iterations=150):
    """
    Perform binary search to find the minimum outer hexagon side length
    """
    # Start with reasonable bounds based on known geometry
    low = 1.0  # Tightened lower bound
    high = 15.0  # Increased upper bound to ensure we find valid solution
    best_valid_size = high
    
    # Binary search with moderate tolerance for better performance
    for _ in range(max_iterations):
        mid = (low + high) / 2
        penalty = evaluate_packing(inner_hex_data, mid)
        if penalty == 0:
            best_valid_size = mid
            high = mid
        else:
            low = mid
    
    return best_valid_size

def local_refinement(current_solution, max_iter=50):
    """Apply local refinement to improve solution quality"""
    best_solution = current_solution.copy()
    best_size = binary_search_min_outer_side(best_solution)
    
    for iteration in range(max_iter):
        # Make small random adjustments to positions
        refined_solution = best_solution.copy()
        
        # Randomly select hexagons to perturb
        hex_indices = random.sample(range(11), min(3, len(range(11))))
        
        for idx in hex_indices:
            # Small random perturbation
            refined_solution[idx, 0] += random.uniform(-0.02, 0.02)
            refined_solution[idx, 1] += random.uniform(-0.02, 0.02)
        
        # Check if this improves the solution
        new_size = binary_search_min_outer_side(refined_solution)
        penalty = evaluate_packing(refined_solution, new_size)
        
        if penalty == 0 and new_size < best_size:
            best_size = new_size
            best_solution = refined_solution.copy()
    
    return best_solution, best_size

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining mathematical programming with local refinement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Enhanced set of configurations based on best inspirations
    configurations = [
        # Configuration from Inspiration 1 - optimized mathematical approach
        np.array([
            [0.0, 0.0, 0.0],         # center
            [0.0, 2.0, 0.0],         # top
            [1.732, 1.0, 0.0],       # top-right (sqrt(3) = 1.732)
            [1.732, -1.0, 0.0],      # bottom-right
            [0.0, -2.0, 0.0],        # bottom
            [-1.732, -1.0, 0.0],     # bottom-left
            [-1.732, 1.0, 0.0],      # top-left
            [3.464, 0.0, 0.0],       # far right
            [-3.464, 0.0, 0.0],      # far left
            [1.732, 3.0, 0.0],       # upper triangle
            [-1.732, 3.0, 0.0],      # upper triangle
        ]),
        
        # Configuration from Inspiration 2 - highly optimized spacing
        np.array([
            [0.0, 0.0, 0.0],         # center
            [0.0, 1.98, 0.0],        # top
            [1.72, 0.99, 0.0],       # top-right
            [1.72, -0.99, 0.0],      # bottom-right
            [0.0, -1.98, 0.0],       # bottom
            [-1.72, -0.99, 0.0],     # bottom-left
            [-1.72, 0.99, 0.0],      # top-left
            [3.44, 0.0, 0.0],        # far right
            [-3.44, 0.0, 0.0],       # far left
            [1.72, 2.97, 0.0],       # upper triangle
            [-1.72, 2.97, 0.0],      # upper triangle
        ]),
        
        # Configuration from Inspiration 3 - compact arrangement
        np.array([
            [0.0, 0.0, 0.0],         # center
            [0.0, 1.8, 0.0],         # top
            [1.559, 0.9, 0.0],       # top-right
            [1.559, -0.9, 0.0],      # bottom-right
            [0.0, -1.8, 0.0],        # bottom
            [-1.559, -0.9, 0.0],     # bottom-left
            [-1.559, 0.9, 0.0],      # top-left
            [3.118, 0.0, 0.0],       # far right
            [-3.118, 0.0, 0.0],      # far left
            [1.559, 2.7, 0.0],       # upper triangle
            [-1.559, 2.7, 0.0],      # upper triangle
        ]),
        
        # Configuration from Inspiration 2 - alternative symmetric pattern
        np.array([
            [0.0, 0.0, 0.0],         # center
            [0.0, 2.1, 0.0],         # top
            [1.732, 1.05, 0.0],      # top-right
            [1.732, -1.05, 0.0],     # bottom-right
            [0.0, -2.1, 0.0],        # bottom
            [-1.732, -1.05, 0.0],    # bottom-left
            [-1.732, 1.05, 0.0],     # top-left
            [3.464, 0.0, 0.0],       # far right
            [-3.464, 0.0, 0.0],      # far left
            [1.732, 3.15, 0.0],      # upper triangle
            [-1.732, 3.15, 0.0],     # upper triangle
        ]),
        
        # Configuration from Inspiration 1 - slightly adjusted version
        np.array([
            [0.0, 0.0, 0.0],         # center
            [0.0, 2.05, 0.0],        # top
            [1.732, 1.025, 0.0],     # top-right
            [1.732, -1.025, 0.0],    # bottom-right
            [0.0, -2.05, 0.0],       # bottom
            [-1.732, -1.025, 0.0],   # bottom-left
            [-1.732, 1.025, 0.0],    # top-left
            [3.464, 0.0, 0.0],       # far right
            [-3.464, 0.0, 0.0],      # far left
            [1.732, 3.075, 0.0],     # upper triangle
            [-1.732, 3.075, 0.0],    # upper triangle
        ]),
    ]
    
    best_solution = None
    best_side_length = float('inf')
    
    # Test each configuration
    for i, config in enumerate(configurations):
        try:
            # Find the minimum outer hexagon side length for this configuration
            min_side_length = binary_search_min_outer_side(config)
            
            # Verify the solution is valid
            if evaluate_packing(config, min_side_length) == 0:
                if min_side_length < best_side_length:
                    best_side_length = min_side_length
                    best_solution = config.copy()
        except Exception as e:
            continue
    
    # Apply local refinement if we have a good solution and time permits
    if best_solution is not None and time.time() - start_time < 50:
        try:
            refined_solution, refined_size = local_refinement(best_solution)
            if refined_size < best_side_length:
                best_side_length = refined_size
                best_solution = refined_solution
        except Exception as e:
            pass
    
    # If we found a valid solution, return it
    if best_solution is not None:
        # Ensure all rotations are within [0, 360)
        best_solution[:, 2] = np.mod(best_solution[:, 2], 360)
        outer_hex_data = np.array([0, 0, 0])  # Outer hexagon centered at origin, no rotation
        return best_solution, outer_hex_data, best_side_length
    
    # Fall back to the most reliable configuration from inspirations
    final_config = configurations[0]  # Use the first (standard) configuration
    
    try:
        min_side_length = binary_search_min_outer_side(final_config)
        if evaluate_packing(final_config, min_side_length) == 0:
            final_config[:, 2] = np.mod(final_config[:, 2], 360)
            outer_hex_data = np.array([0, 0, 0])  # Outer hexagon centered at origin, no rotation
            return final_config, outer_hex_data, min_side_length
    except Exception as e:
        pass
    
    # Last resort - basic configuration
    positions = [
        (0.0, 0.0),         # center
        (0.0, 2.0),         # top
        (1.732, 1.0),       # top-right (sqrt(3) = 1.732)
        (1.732, -1.0),      # bottom-right
        (0.0, -2.0),        # bottom
        (-1.732, -1.0),     # bottom-left
        (-1.732, 1.0),      # top-left
        (3.464, 0.0),       # far right
        (-3.464, 0.0),      # far left
        (1.732, 3.0),       # upper triangle
        (-1.732, 3.0),      # upper triangle
    ]
    
    solution = np.zeros((11, 3))
    for i, (x, y) in enumerate(positions):
        solution[i] = [x, y, 0.0]
    
    # Binary search for minimal valid outer hexagon size
    min_side_length = binary_search_min_outer_side(solution)
    solution[:, 2] = np.mod(solution[:, 2], 360)
    outer_hex_data = np.array([0, 0, 0])  # Outer hexagon centered at origin, no rotation
    
    return solution, outer_hex_data, min_side_length


# EVOLVE-BLOCK-END
