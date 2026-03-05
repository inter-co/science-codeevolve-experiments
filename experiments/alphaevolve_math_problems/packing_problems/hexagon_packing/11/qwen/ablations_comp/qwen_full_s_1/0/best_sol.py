# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon
import warnings
import random
warnings.filterwarnings('ignore')

# Precomputed constants for better performance
SQRT_3 = np.sqrt(3)
SQRT_3_OVER_2 = SQRT_3 / 2.0

def get_hexagon_vertices(center, side_length, rotation=0):
    """Get vertices of a regular hexagon using optimized computation"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return [(center[0] + side_length * np.cos(angle), 
             center[1] + side_length * np.sin(angle)) 
            for angle in angles]

def check_hexagon_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer hexagon"""
    try:
        hex_poly = Polygon(hexagon)
        outer_poly = Polygon(outer_hexagon)
        # Use contains method with proper error handling
        return outer_poly.contains(hex_poly) or outer_poly.covers(hex_poly)
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
    outer_hex = get_hexagon_vertices(outer_center, outer_hex_side_length, 0)
    
    # Check if all inner hexagons are contained and non-overlapping
    total_penalty = 0
    
    inner_hexagons = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        side_length = 1.0  # unit hexagon
        hexagon = get_hexagon_vertices(center, side_length, rotation)
        inner_hexagons.append(hexagon)
        
        # Check containment
        if not check_hexagon_containment(hexagon, outer_hex):
            total_penalty += 1000000  # Large penalty for containment violation
            
        # Check overlaps with other hexagons
        for j in range(i):
            if check_hexagon_overlap(hexagon, inner_hexagons[j]):
                total_penalty += 1000000  # Large penalty for overlap
    
    return total_penalty

def binary_search_min_outer_size(inner_solution, max_size=15.0, precision=1e-6):
    """Binary search to find minimum outer hexagon size that contains all hexagons"""
    low = 1.0
    high = max_size
    best_size = max_size
    
    # Binary search with improved precision and early termination
    max_iterations = 100  # Reduced for faster execution
    for _ in range(max_iterations):
        mid = (low + high) / 2
        penalty = evaluate_packing(inner_solution, mid)
        if penalty == 0:
            best_size = mid
            high = mid
        else:
            low = mid
            
        # Early termination if we've reached desired precision
        if abs(high - low) < precision:
            break
            
    return best_size

def generate_best_initial_configurations():
    """Generate the best initial configurations from inspiration programs"""
    # Use configurations from inspirations with focus on mathematical precision and proven results
    configurations = [
        # Configuration from Inspiration 1 - high-quality benchmark
        [(0, 0), (0, 2.0), (1.7320508075688772, 1.0), (1.7320508075688772, -1.0), (0, -2.0), (-1.7320508075688772, -1.0), (-1.7320508075688772, 1.0),
         (3.4641016151377544, 0), (-3.4641016151377544, 0), (1.7320508075688772, 3.0), (-1.7320508075688772, 3.0)],
        
        # Configuration from Inspiration 2 - tight spacing
        [(0, 0), (0, 1.98), (1.72, 0.99), (1.72, -0.99), (0, -1.98), (-1.72, -0.99), (-1.72, 0.99),
         (3.44, 0), (-3.44, 0), (1.72, 2.97), (-1.72, 2.97)],
        
        # Configuration from Inspiration 3 - compact arrangement
        [(0, 0), (0, 1.8), (1.559, 0.9), (1.559, -0.9), (0, -1.8), (-1.559, -0.9), (-1.559, 0.9),
         (3.118, 0), (-3.118, 0), (1.559, 2.7), (-1.559, 2.7)],
        
        # Configuration from Inspiration 2 - alternative grid pattern
        [(0, 0), (0, 2.1), (1.732, 1.05), (1.732, -1.05), (0, -2.1), (-1.732, -1.05), (-1.732, 1.05),
         (3.464, 0), (-3.464, 0), (1.732, 3.15), (-1.732, 3.15)],
        
        # Configuration from Inspiration 1 - balanced mathematical approach
        [(0, 0), (0, 2.02), (1.732, 1.01), (1.732, -1.01), (0, -2.02), (-1.732, -1.01), (-1.732, 1.01),
         (3.464, 0), (-3.464, 0), (1.732, 3.03), (-1.732, 3.03)],
         
        # Configuration from Inspiration 3 - optimized for minimal outer hexagon
        [(0, 0), (0, 2.0), (1.732, 1.0), (1.732, -1.0), (0, -2.0), (-1.732, -1.0), (-1.732, 1.0),
         (3.464, 0), (-3.464, 0), (1.732, 3.0), (-1.732, 3.0)],
         
        # Configuration from Inspiration 2 - precise mathematical arrangement
        [(0, 0), (0, 2.17), (1.732, 1.085), (1.732, -1.085), (0, -2.17), (-1.732, -1.085), (-1.732, 1.085),
         (3.464, 0), (-3.464, 0), (1.732, 3.255), (-1.732, 3.255)],
         
        # Configuration from Inspiration 1 - very tight arrangement
        [(0, 0), (0, 1.95), (1.69, 0.975), (1.69, -0.975), (0, -1.95), (-1.69, -0.975), (-1.69, 0.975),
         (3.38, 0), (-3.38, 0), (1.69, 2.925), (-1.69, 2.925)],
    ]
    
    return configurations

def local_refinement(current_solution, max_iter=50):
    """Apply local refinement to improve solution quality"""
    best_solution = current_solution.copy()
    best_size = binary_search_min_outer_size(best_solution, 15.0, 1e-6)
    
    for iteration in range(max_iter):
        # Make small random adjustments to positions
        refined_solution = best_solution.copy()
        
        # Randomly select hexagons to perturb (focusing on outer ones for better impact)
        hex_indices = random.sample(range(1, 11), min(3, len(range(1, 11))))
        
        for idx in hex_indices:
            # Small random perturbation
            refined_solution[idx, 0] += random.uniform(-0.005, 0.005)
            refined_solution[idx, 1] += random.uniform(-0.005, 0.005)
        
        # Check if this improves the solution
        new_size = binary_search_min_outer_size(refined_solution, 15.0, 1e-6)
        penalty = evaluate_packing(refined_solution, new_size)
        
        if penalty == 0 and new_size < best_size:
            best_size = new_size
            best_solution = refined_solution.copy()
    
    return best_solution, best_size

def optimize_with_improved_strategy():
    """Use improved optimization strategy based on best inspirations"""
    # Generate configurations from inspiration programs
    configs = generate_best_initial_configurations()
    
    best_solution = None
    best_side_length = float('inf')
    start_time = time.time()
    
    # Test each configuration systematically with more thorough validation
    for i, positions in enumerate(configs):
        # Create solution with no rotation initially
        solution = np.zeros((11, 3))
        for j, (x, y) in enumerate(positions):
            solution[j] = [x, y, 0]
        
        # Binary search for minimal outer hexagon size with higher precision
        min_size = binary_search_min_outer_size(solution, 15.0, 1e-6)
        penalty = evaluate_packing(solution, min_size)
        
        # If valid solution, check if it's better
        if penalty == 0 and min_size < best_side_length:
            best_side_length = min_size
            best_solution = solution.copy()
            
        # Early exit if we have a very good solution or time limit approached
        if time.time() - start_time > 55:
            break
    
    # If no good solution found, fall back to the best known configuration
    if best_solution is None:
        # Use a reliable configuration from inspiration 1
        positions = [
            (0, 0), (0, 2.0), (1.7320508075688772, 1.0), (1.7320508075688772, -1.0), (0, -2.0), (-1.7320508075688772, -1.0), (-1.7320508075688772, 1.0),
            (3.4641016151377544, 0), (-3.4641016151377544, 0), (1.7320508075688772, 3.0), (-1.7320508075688772, 3.0)
        ]
        best_solution = np.zeros((11, 3))
        for j, (x, y) in enumerate(positions):
            best_solution[j] = [x, y, 0]
        best_side_length = binary_search_min_outer_size(best_solution, 15.0, 1e-6)
    
    # Apply local refinement to squeeze out any remaining improvement
    if best_solution is not None and time.time() - start_time < 55:
        try:
            refined_solution, refined_size = local_refinement(best_solution, 100)
            if refined_size < best_side_length:
                best_side_length = refined_size
                best_solution = refined_solution
        except Exception as e:
            pass
    
    # Additional geometric refinement: try small rotation adjustments
    if best_solution is not None and time.time() - start_time < 55:
        # Try rotating some hexagons to see if we can reduce outer hexagon size further
        best_final_solution = best_solution.copy()
        best_final_size = best_side_length
        
        # Test rotations for key hexagons
        test_rotations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        
        for _ in range(30):  # Reduced iterations for speed
            # Pick a random hexagon to rotate
            hex_idx = random.randint(0, 10)
            rotation = random.choice(test_rotations)
            
            # Try the rotation
            temp_solution = best_final_solution.copy()
            temp_solution[hex_idx, 2] = rotation
            
            # Check if this improves the packing
            size_check = binary_search_min_outer_size(temp_solution, 15.0, 1e-6)
            penalty_check = evaluate_packing(temp_solution, size_check)
            
            # If this is better, keep the change
            if penalty_check == 0 and size_check < best_final_size:
                best_final_size = size_check
                best_final_solution = temp_solution.copy()
                
            # Early exit if time is running out
            if time.time() - start_time > 55:
                break
        
        best_solution = best_final_solution
        best_side_length = best_final_size
    
    return best_solution, best_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining mathematical programming and evolutionary optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use improved optimization approach
    inner_hex_data, outer_side_length = optimize_with_improved_strategy()
    
    # Ensure all rotations are within [0, 360)
    inner_hex_data[:, 2] = np.mod(inner_hex_data[:, 2], 360)
    
    # Set outer hexagon at center with zero rotation
    outer_hex_data = np.array([0, 0, 0])
    
    eval_time = time.time() - start_time
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
