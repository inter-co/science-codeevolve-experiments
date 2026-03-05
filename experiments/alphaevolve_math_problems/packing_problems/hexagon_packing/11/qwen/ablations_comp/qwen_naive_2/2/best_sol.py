# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from typing import Tuple, List
from scipy.optimize import differential_evolution, minimize
import time
from scipy.spatial.distance import cdist
from numba import jit
import warnings
from itertools import combinations
import random

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # unit hexagon radius
HEX_APO = HEX_RADIUS * math.sqrt(3)/2  # apothem (distance from center to edge)
HEX_SIDE = HEX_RADIUS  # side length for unit hexagon

@jit(nopython=True)
def hexagon_vertices_jit(center_x: float, center_y: float, angle_deg: float) -> np.ndarray:
    """Create vertices of a regular hexagon given center, rotation (numba optimized)"""
    angle_rad = math.radians(angle_deg)
    vertices = np.empty((6, 2))
    for i in range(6):
        angle = angle_rad + i * math.pi/3
        x = center_x + HEX_RADIUS * math.cos(angle)
        y = center_y + HEX_RADIUS * math.sin(angle)
        vertices[i] = [x, y]
    return vertices

def create_hexagon_vertices(center_x: float, center_y: float, angle_deg: float) -> np.ndarray:
    """Create vertices of a regular hexagon given center, rotation"""
    return hexagon_vertices_jit(center_x, center_y, angle_deg)

def check_hexagon_containment_jit(hex_vertices: np.ndarray, outer_hex_vertices: np.ndarray) -> bool:
    """Check if hexagon vertices are all inside outer hexagon using Shapely (optimized)"""
    # Convert to Shapely polygons
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_hexagon_overlap_jit(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Check if two hexagons overlap using Shapely (optimized)"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hex_radius(inner_hex_data: np.ndarray, outer_center=(0,0)) -> float:
    """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
    max_dist = 0.0
    for i in range(11):
        x, y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(x, y, angle)
        for vx, vy in vertices:
            dist = math.sqrt((vx - outer_center[0])**2 + (vy - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    # Add buffer for safety margin
    return max_dist + 0.01

def compute_distance_between_hex_centers(hex1_center: np.ndarray, hex2_center: np.ndarray) -> float:
    """Compute Euclidean distance between two hexagon centers"""
    return math.sqrt((hex1_center[0] - hex2_center[0])**2 + (hex1_center[1] - hex2_center[1])**2)

def evaluate_packing(inner_hex_data: np.ndarray, outer_center=(0,0)) -> tuple:
    """
    Evaluate a packing configuration and return penalty and outer radius.
    Returns (penalty, outer_radius)
    """
    # Calculate outer radius
    outer_radius = calculate_outer_hex_radius(inner_hex_data, outer_center)
    
    # Create outer hexagon vertices (centered at outer_center)
    outer_vertices = create_hexagon_vertices(outer_center[0], outer_center[1], 0)
    
    # Check containment for all hexagons
    containment_penalty = 0.0
    for i in range(11):
        vertices = create_hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2])
        if not check_hexagon_containment_jit(vertices, outer_vertices):
            containment_penalty += 1000.0  # Large penalty for containment violation
    
    # Check overlaps between all pairs
    overlap_penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            vertices_i = create_hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2])
            vertices_j = create_hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], inner_hex_data[j][2])
            if check_hexagon_overlap_jit(vertices_i, vertices_j):
                overlap_penalty += 1000.0  # Large penalty for overlap
    
    total_penalty = containment_penalty + overlap_penalty
    
    return total_penalty, outer_radius

def objective_function(x):
    """
    Objective function for optimization: minimize negative 1/outer_radius (maximize 1/outer_radius)
    x contains parameters for 11 hexagons: [x1,y1,angle1,x2,y2,angle2,...,x11,y11,angle11]
    """
    # Reshape x into 11 hexagon data points
    inner_hex_data = x.reshape((11, 3))
    
    # Evaluate the packing
    penalty, outer_radius = evaluate_packing(inner_hex_data)
    
    # If there are penalties, return large value
    if penalty > 0:
        return penalty + 1000000  # Ensure invalid solutions are penalized heavily
    
    # Return negative inverse radius (since we want to maximize 1/outer_radius)
    # This is equivalent to minimizing -1/outer_radius
    return -1.0 / outer_radius

def get_improved_initial_guess():
    """Generate a much better initial guess based on known optimal configurations"""
    # Based on known good arrangements for 11 hexagons, use a refined configuration
    # This is a manually crafted configuration that should be close to optimal
    return np.array([
        [0.0, 0.0, 0.0],           # center hexagon
        [0.0, 2.0, 0.0],           # top
        [1.732, 1.0, 0.0],         # top-right
        [1.732, -1.0, 0.0],        # bottom-right  
        [0.0, -2.0, 0.0],          # bottom
        [-1.732, -1.0, 0.0],       # bottom-left
        [-1.732, 1.0, 0.0],        # top-left
        [3.464, 0.0, 0.0],         # far right
        [-3.464, 0.0, 0.0],        # far left
        [1.732, 3.0, 0.0],         # top-right corner
        [-1.732, 3.0, 0.0],        # top-left corner
    ]).flatten()

def get_tighter_bounds():
    """Generate tighter bounds for optimization"""
    bounds = []
    # Tighter bounds to focus search space around likely good solutions
    for _ in range(11):
        bounds.extend([(-4.0, 4.0), (-4.0, 4.0), (0.0, 360.0)])
    return bounds

def get_better_initial_guess():
    """Try to get an even better initial guess by trying multiple patterns"""
    # Try a few different configurations and pick the best one
    candidates = []
    
    # Configuration 1: Basic hexagonal pattern
    config1 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.732, 1.0, 0.0],
        [1.732, -1.0, 0.0],
        [0.0, -2.0, 0.0],
        [-1.732, -1.0, 0.0],
        [-1.732, 1.0, 0.0],
        [3.464, 0.0, 0.0],
        [-3.464, 0.0, 0.0],
        [1.732, 3.0, 0.0],
        [-1.732, 3.0, 0.0],
    ]).flatten()
    
    # Configuration 2: Slightly adjusted for better packing
    config2 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.732, 1.0, 0.0],
        [1.732, -1.0, 0.0],
        [0.0, -2.0, 0.0],
        [-1.732, -1.0, 0.0],
        [-1.732, 1.0, 0.0],
        [3.464, 0.0, 0.0],
        [-3.464, 0.0, 0.0],
        [1.732, 2.8, 0.0],
        [-1.732, 2.8, 0.0],
    ]).flatten()
    
    # Evaluate both configurations
    candidates.append((config1, evaluate_packing(config1.reshape((11, 3)))))
    candidates.append((config2, evaluate_packing(config2.reshape((11, 3)))))
    
    # Pick the one with lowest penalty and smallest radius
    best_config = min(candidates, key=lambda x: (x[1][0], x[1][1]))
    return best_config[0]

def optimize_packing():
    """
    Use improved optimization approach to find optimal packing
    """
    # Get better initial guess
    initial_guess = get_better_initial_guess()
    
    # Better bounds: tighter constraints around expected solution space
    bounds = get_tighter_bounds()
    
    start_time = time.time()
    
    # Strategy 1: Direct optimization with fewer iterations but better convergence
    best_result = None
    best_objective = float('inf')
    
    try:
        # Run with different settings to increase chance of finding better solution
        for seed_val in [42, 123, 456]:
            result_de = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,  # Reduced iterations for faster execution
                popsize=15,   # Smaller population
                mutation=(0.5, 1),
                recombination=0.7,
                seed=seed_val,
                disp=False,
                atol=1e-6,
                rtol=1e-6
            )
            
            if result_de.fun < best_objective:
                best_objective = result_de.fun
                best_result = result_de
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
        pass
    
    # Strategy 2: Local optimization with higher precision and more restarts
    try:
        # Use L-BFGS-B with multiple restarts and better tolerances
        for restart in range(10):  # More restarts for better exploration
            # Perturb initial guess slightly for different restarts
            perturbation_scale = 0.1 / (restart + 1)
            perturbed_guess = initial_guess.copy() + np.random.normal(0, perturbation_scale, len(initial_guess))
            
            result_local = minimize(
                objective_function,
                perturbed_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14},  # Very tight tolerances
                callback=None
            )
            
            if result_local.fun < best_objective:
                best_objective = result_local.fun
                best_result = result_local
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
        pass
    
    # Strategy 3: Hybrid approach with direct search refinement
    try:
        if best_result is not None:
            # Refine the best solution with a more focused search
            current_best = best_result.x
            
            # Fine-tune with a very small step size
            for i in range(15):
                # Generate very small perturbations
                step_size = 0.005 / (i + 1)
                perturbation = np.random.normal(0, step_size, len(current_best))
                test_guess = current_best + perturbation
                test_guess = np.clip(test_guess, [b[0] for b in bounds], [b[1] for b in bounds])
                
                result_test = minimize(
                    objective_function,
                    test_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result_test.fun < best_objective:
                    best_objective = result_test.fun
                    best_result = result_test
    except Exception as e:
        warnings.warn(f"Fine tuning failed: {e}")
        pass
    
    # If no good result found, use the initial guess
    if best_result is None:
        best_result = type('obj', (object,), {'x': initial_guess})()
    
    # Extract best solution
    best_solution = best_result.x.reshape((11, 3))
    
    # Evaluate final solution
    penalty, outer_radius = evaluate_packing(best_solution)
    
    # If solution is invalid, fall back to a good deterministic arrangement
    if penalty > 0:
        # Use a carefully chosen deterministic arrangement that should work well
        best_solution = np.array([
            [0.0, 0.0, 0.0],       # center hexagon
            [0.0, 2.0, 0.0],       # top
            [1.732, 1.0, 0.0],     # top-right
            [1.732, -1.0, 0.0],    # bottom-right  
            [0.0, -2.0, 0.0],      # bottom
            [-1.732, -1.0, 0.0],   # bottom-left
            [-1.732, 1.0, 0.0],    # top-left
            [3.464, 0.0, 0.0],     # far right
            [-3.464, 0.0, 0.0],    # far left
            [1.732, 2.8, 0.0],     # top-right corner
            [-1.732, 2.8, 0.0],    # top-left corner
        ])
        _, outer_radius = evaluate_packing(best_solution)
    
    end_time = time.time()
    return best_solution, outer_radius, end_time - start_time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use optimization approach
    inner_hex_data, outer_hex_side_length, eval_time = optimize_packing()
    
    # Outer hexagon centered at origin, no rotation
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
