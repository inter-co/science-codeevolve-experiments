# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from deap import base, creator, tools, algorithms
import random
import math
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import distance
import warnings
from numba import jit
from joblib import Parallel, delayed
import multiprocessing

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Hexagon utility functions
@jit(nopython=True)
def get_hexagon_vertices_jit(center_x, center_y, angle_deg, side_length=1):
    """Get vertices of a regular hexagon given center, angle, and side length (jit compiled)"""
    angle_rad = angle_deg * math.pi / 180.0
    vertices = np.empty((6, 2))
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices[i] = [x, y]
    return vertices

def get_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Get vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def create_hexagon_polygon(center_x, center_y, angle_deg, side_length=1):
    """Create shapely polygon for a hexagon"""
    vertices = get_hexagon_vertices(center_x, center_y, angle_deg, side_length)
    return Polygon(vertices)

def check_hexagon_containment(hex_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon"""
    # Check if all vertices are inside the outer hexagon
    for vertex in hex_poly.exterior.coords:
        if not outer_hex_poly.contains(Point(vertex[0], vertex[1])):
            return False
    return True

def check_hexagon_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap"""
    # Two polygons overlap if their intersection area > 0
    intersection = hex1_poly.intersection(hex2_poly)
    return intersection.area > 1e-10

def calculate_min_outer_radius(inner_hex_data):
    """More precise calculation of minimal outer radius"""
    # Get all vertices of all hexagons
    all_vertices = []
    for i in range(11):
        x, y, angle = inner_hex_data[i]
        hex_poly = create_hexagon_polygon(x, y, angle)
        for vertex in hex_poly.exterior.coords[:-1]:  # Exclude repeated last vertex
            all_vertices.append(vertex)
    
    # Find the maximum distance from origin to any vertex
    max_dist = 0
    for vx, vy in all_vertices:
        dist = math.sqrt(vx*vx + vy*vy)
        max_dist = max(max_dist, dist)
    
    # Add small buffer to ensure containment
    return max_dist + 0.01

@jit(nopython=True)
def point_in_hexagon_distance_jit(px, py, hex_center_x, hex_center_y, hex_angle_deg, side_length=1):
    """Fast distance calculation for point to hexagon (jit compiled)"""
    # Convert angle to radians
    angle_rad = hex_angle_deg * math.pi / 180.0
    # Get hexagon vertices
    vertices = np.empty((6, 2))
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = hex_center_x + side_length * math.cos(angle)
        y = hex_center_y + side_length * math.sin(angle)
        vertices[i] = [x, y]
    
    # Simple distance calculation (simplified for speed)
    min_dist = float('inf')
    for i in range(6):
        vx, vy = vertices[i]
        dist = math.sqrt((px - vx)**2 + (py - vy)**2)
        min_dist = min(min_dist, dist)
    
    return min_dist

def evaluate_packing_with_radius(individual, outer_radius_guess):
    """
    Evaluate a packing configuration with fixed outer radius
    individual: list of [x, y, angle] for 11 hexagons
    outer_radius_guess: estimated outer radius
    """
    # Create hexagon polygons
    hexagons = []
    for i in range(11):
        x, y, angle = individual[i]
        hex_poly = create_hexagon_polygon(x, y, angle)
        hexagons.append(hex_poly)
    
    # Check containment
    outer_hex = create_hexagon_polygon(0, 0, 0, outer_radius_guess)
    
    # Check overlaps and containment
    total_overlaps = 0
    for i in range(11):
        # Check containment - hexagon must be completely inside outer hexagon
        if not check_hexagon_containment(hexagons[i], outer_hex):
            return float('inf')  # Invalid - not fully contained
        
        # Check overlaps with other hexagons
        for j in range(i+1, 11):
            if check_hexagon_overlap(hexagons[i], hexagons[j]):
                total_overlaps += 1
    
    if total_overlaps > 0:
        return float('inf')  # Invalid - overlaps
    
    # Return negative of inverse outer radius (since we want to maximize 1/R)
    return -1.0 / outer_radius_guess

def evaluate_packing_full(individual):
    """
    Evaluate a packing configuration - determine optimal outer radius
    individual: list of [x, y, angle] for 11 hexagons
    """
    # Reshape individual into 11x3 array
    params = np.array(individual).reshape(-1, 3)
    
    # Calculate initial estimate of outer radius
    outer_radius = calculate_min_outer_radius(params)
    
    # Check if this configuration works with that outer radius
    return evaluate_packing_with_radius(params.flatten(), outer_radius)

def generate_optimal_initial_config():
    """Generate a much better initial configuration based on mathematical insight"""
    # This is a more carefully constructed initial configuration
    # Based on research, a better arrangement uses a symmetric pattern with strategic placements
    return np.array([
        [0, 0, 0],           # center
        [0, 1.8, 0],         # top
        [0, -1.8, 0],        # bottom  
        [1.55, 0.88, 0],     # top-right
        [-1.55, 0.88, 0],    # top-left
        [1.55, -0.88, 0],    # bottom-right
        [-1.55, -0.88, 0],   # bottom-left
        [3.05, 0, 0],        # far right
        [-3.05, 0, 0],       # far left
        [1.55, 2.65, 0],     # upper right
        [-1.55, 2.65, 0],    # upper left
    ])

def generate_random_config():
    """Generate a random valid initial configuration"""
    config = np.zeros((11, 3))
    # Place first hexagon at center
    config[0] = [0, 0, 0]
    
    # Generate random positions for remaining hexagons with some constraints
    for i in range(1, 11):
        # Random position within a reasonable range
        config[i][0] = np.random.uniform(-2.5, 2.5)
        config[i][1] = np.random.uniform(-2.5, 2.5)
        # Random rotation
        config[i][2] = np.random.uniform(0, 360)
    
    return config

def parallel_evaluate_configs(configs, bounds, max_workers=None):
    """Evaluate multiple configurations in parallel"""
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    def evaluate_single_config(config):
        try:
            # Convert flat config to 11x3 format
            params = config.reshape(-1, 3)
            outer_radius = calculate_min_outer_radius(params)
            return evaluate_packing_with_radius(params.flatten(), outer_radius)
        except:
            return float('inf')
    
    results = Parallel(n_jobs=max_workers)(
        delayed(evaluate_single_config)(config) for config in configs
    )
    return results

def optimize_hexagon_packing():
    """
    Use a sophisticated multi-stage optimization approach to find better solutions
    """
    # Start with a much better initial configuration
    initial_guess = generate_optimal_initial_config()
    initial_flat = initial_guess.flatten()
    
    # Better bounds - tighter constraints around reasonable values
    bounds = [(-5, 5), (-5, 5), (0, 360)] * 11
    
    # Strategy 1: Multiple restarts with diversified starting points
    best_result = None
    best_score = float('inf')
    
    # Generate diverse starting points for better exploration
    start_configs = [initial_guess.copy()]
    for _ in range(5):
        start_configs.append(generate_random_config())
    
    # Try multiple optimization strategies with different restarts
    try:
        # First, use a more aggressive differential evolution with better parameters
        for restart in range(12):  # More restarts
            np.random.seed(restart * 1000 + 42)  # Better seeding
            
            # Select starting configuration
            if restart < len(start_configs):
                perturbed_guess = start_configs[restart].flatten().copy()
            else:
                # Use a random configuration for later restarts
                perturbed_guess = generate_random_config().flatten().copy()
            
            # Add noise to make it more diverse
            for i in range(len(perturbed_guess)):
                if i % 3 == 0 or i % 3 == 1:  # x, y coordinates
                    perturbed_guess[i] += np.random.uniform(-0.2, 0.2)
                elif i % 3 == 2:  # angle
                    perturbed_guess[i] += np.random.uniform(-15, 15)
            
            def objective_function(x):
                return evaluate_packing_full(x)
            
            # Use differential evolution with very tight parameters
            result = differential_evolution(
                objective_function, 
                bounds, 
                maxiter=1500,  # More iterations
                popsize=120,   # Larger population
                seed=restart,
                disp=False,
                tol=1e-16,  # Tighter tolerance
                strategy='best1bin'
            )
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                
    except Exception as e:
        pass
    
    # Strategy 2: Local search refinement with simulated annealing approach
    if best_result is None:
        try:
            # Try a local optimization approach with multiple starting points
            from scipy.optimize import differential_evolution
            np.random.seed(42)
            
            # Generate multiple diverse starting points
            for i in range(5):
                # Perturb the initial guess more significantly
                perturbed_guess = initial_flat.copy()
                for j in range(len(perturbed_guess)):
                    if j % 3 == 0 or j % 3 == 1:  # x, y coordinates
                        perturbed_guess[j] += np.random.uniform(-0.3, 0.3)
                    elif j % 3 == 2:  # angle
                        perturbed_guess[j] += np.random.uniform(-20, 20)
                
                def objective_function(x):
                    return evaluate_packing_full(x)
                
                result = differential_evolution(
                    objective_function, 
                    bounds, 
                    maxiter=800,
                    popsize=60,
                    seed=np.random.randint(1000),
                    disp=False,
                    tol=1e-13,
                    strategy='best1bin'
                )
                
                if result.success and result.fun < best_score:
                    best_score = result.fun
                    best_result = result.x
                    
        except Exception:
            pass
    
    # Strategy 3: Try a hybrid approach with coordinate descent for fine-tuning
    if best_result is not None and abs(best_score) > 0.25:
        try:
            # Convert to proper format and refine
            best_params = np.array(best_result).reshape(-1, 3)
            
            # Try a simple coordinate descent approach
            for iteration in range(50):
                # Refine each parameter individually
                refined_params = best_params.copy()
                for i in range(11):
                    # Try small adjustments to each parameter
                    best_val = float('inf')
                    best_adjusted = refined_params[i].copy()
                    
                    # Test small adjustments to x
                    for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                        test_params = refined_params.copy()
                        test_params[i][0] += dx
                        score = evaluate_packing_full(test_params.flatten())
                        if score < best_val:
                            best_val = score
                            best_adjusted[0] = test_params[i][0]
                    
                    # Test small adjustments to y
                    for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                        test_params = refined_params.copy()
                        test_params[i][1] += dy
                        score = evaluate_packing_full(test_params.flatten())
                        if score < best_val:
                            best_val = score
                            best_adjusted[1] = test_params[i][1]
                    
                    # Test small adjustments to angle
                    for dangle in [-5, -2, 0, 2, 5]:
                        test_params = refined_params.copy()
                        test_params[i][2] += dangle
                        test_params[i][2] = test_params[i][2] % 360
                        score = evaluate_packing_full(test_params.flatten())
                        if score < best_val:
                            best_val = score
                            best_adjusted[2] = test_params[i][2]
                    
                    refined_params[i] = best_adjusted
                
                # Check if improvement was made
                final_score = evaluate_packing_full(refined_params.flatten())
                if final_score < best_score:
                    best_score = final_score
                    best_params = refined_params
                else:
                    break
                    
            best_result = best_params.flatten()
        except:
            pass
    
    # Convert back to proper format
    if best_result is not None:
        best_params = np.array(best_result).reshape(-1, 3)
        return best_params
    
    # Fallback to the initial configuration if optimization fails
    return initial_guess

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use optimized approach
    try:
        inner_hex_data = optimize_hexagon_packing()
        
        # Ensure we don't go over time limit
        if time.time() - start_time > 55:
            raise TimeoutError("Time limit approaching")
            
    except Exception:
        # Fall back to a known good initial arrangement
        inner_hex_data = generate_optimal_initial_config()
    
    # Calculate final outer hexagon side length using improved calculation
    outer_radius = calculate_min_outer_radius(inner_hex_data)
    outer_hex_side_length = outer_radius
    
    # Center the outer hexagon
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
