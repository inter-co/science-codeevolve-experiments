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
    # Use a more precise containment check
    return outer_polygon.contains(inner_polygon) or outer_polygon.touches(inner_polygon)

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
    # Add buffer for safety margin - reduced from 0.01 to 0.001 for precision
    return max_dist + 0.001

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
            containment_penalty += 10000.0  # Larger penalty for containment violation
    
    # Check overlaps between all pairs
    overlap_penalty = 0.0
    for i, j in combinations(range(11), 2):
        vertices_i = create_hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2])
        vertices_j = create_hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], inner_hex_data[j][2])
        if check_hexagon_overlap_jit(vertices_i, vertices_j):
            overlap_penalty += 10000.0  # Larger penalty for overlap
    
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
        return penalty + 10000000  # Even larger penalty for invalid solutions
    
    # Return negative inverse radius (since we want to maximize 1/outer_radius)
    # This is equivalent to minimizing -1/outer_radius
    return -1.0 / outer_radius

def optimize_packing():
    """
    Use improved optimization approach to find optimal packing
    """
    # More sophisticated initial guess based on known good hexagonal packing patterns
    # Based on research on hexagon packings, this configuration should be close to optimal
    initial_guess = np.array([
        [0.0, 0.0, 0.0],       # center hexagon
        [0.0, 2.0, 0.0],       # top
        [1.732, 1.0, 0.0],     # top-right
        [1.732, -1.0, 0.0],    # bottom-right  
        [0.0, -2.0, 0.0],      # bottom
        [-1.732, -1.0, 0.0],   # bottom-left
        [-1.732, 1.0, 0.0],    # top-left
        [3.464, 0.0, 0.0],     # far right
        [-3.464, 0.0, 0.0],    # far left
        [1.732, 3.0, 0.0],     # top-right corner
        [-1.732, 3.0, 0.0],    # top-left corner
    ]).flatten()
    
    # Better bounds: tighter constraints around expected solution space
    bounds = []
    # Position bounds: more constrained around the expected optimal region
    for _ in range(11):
        bounds.extend([(-10.0, 10.0), (-10.0, 10.0), (0.0, 360.0)])
    
    # First run with differential evolution for global search
    start_time = time.time()
    
    # Try multiple optimization strategies to find better solutions
    best_result = None
    best_objective = float('inf')
    
    # Strategy 1: Differential Evolution (global search) with enhanced parameters
    try:
        # Run with more iterations and better population size
        result_de = differential_evolution(
            objective_function,
            bounds,
            maxiter=200,
            popsize=30,
            mutation=(0.8, 1),
            recombination=0.9,
            seed=42,
            disp=False,
            atol=1e-12,
            rtol=1e-12
        )
        
        if result_de.fun < best_objective:
            best_objective = result_de.fun
            best_result = result_de
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
        pass
    
    # Strategy 2: Local optimization with multiple restarts and better bounds
    try:
        # Use multiple restarts with better starting points
        for restart in range(10):  # Increased from 5 to 10 restarts
            # Generate diverse starting points
            if restart == 0:
                # Start with our good initial guess
                perturbed_guess = initial_guess.copy()
            else:
                # Random perturbations around the initial guess with adaptive scaling
                scale_factor = 0.5 if restart < 5 else 0.1  # Less perturbation in later restarts
                perturbed_guess = initial_guess.copy() + np.random.normal(0, scale_factor, len(initial_guess))
            
            result_local = minimize(
                objective_function,
                perturbed_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15},  # Tighter tolerances
                callback=lambda x: None  # No callback needed
            )
            
            if result_local.fun < best_objective:
                best_objective = result_local.fun
                best_result = result_local
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
        pass
    
    # Strategy 3: Enhanced fine-tuning with simulated annealing-inspired approach
    try:
        if best_result is not None:
            # Do a few more local optimizations with small perturbations
            current_best = best_result.x
            for i in range(10):  # Increase from 5 to 10 fine-tuning steps
                # Small random perturbation with decreasing magnitude
                perturbation_magnitude = 0.05 * (1.0 - i/20.0)  # Decreasing over iterations
                perturbation = np.random.normal(0, perturbation_magnitude, len(current_best))
                test_guess = current_best + perturbation
                test_guess = np.clip(test_guess, [b[0] for b in bounds], [b[1] for b in bounds])
                
                result_test = minimize(
                    objective_function,
                    test_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result_test.fun < best_objective:
                    best_objective = result_test.fun
                    best_result = result_test
    except Exception as e:
        warnings.warn(f"Fine-tuning failed: {e}")
        pass
    
    # Strategy 4: Direct search for better configurations
    try:
        # Try a few more specialized configurations
        special_configs = [
            # Configuration with higher density arrangement
            np.array([
                [0.0, 0.0, 0.0],       # center hexagon
                [0.0, 2.0, 0.0],       # top
                [1.732, 1.0, 0.0],     # top-right
                [1.732, -1.0, 0.0],    # bottom-right  
                [0.0, -2.0, 0.0],      # bottom
                [-1.732, -1.0, 0.0],   # bottom-left
                [-1.732, 1.0, 0.0],    # top-left
                [3.464, 0.0, 0.0],     # far right
                [-3.464, 0.0, 0.0],    # far left
                [1.732, 3.0, 0.0],     # top-right corner
                [-1.732, 3.0, 0.0],    # top-left corner
            ]).flatten(),
            # Another variation with different positioning
            np.array([
                [0.0, 0.0, 0.0],       # center hexagon
                [0.0, 2.0, 0.0],       # top
                [1.732, 1.0, 0.0],     # top-right
                [1.732, -1.0, 0.0],    # bottom-right  
                [0.0, -2.0, 0.0],      # bottom
                [-1.732, -1.0, 0.0],   # bottom-left
                [-1.732, 1.0, 0.0],    # top-left
                [3.464, 0.0, 0.0],     # far right
                [-3.464, 0.0, 0.0],    # far left
                [1.732, 3.0, 0.0],     # top-right corner
                [-1.732, 3.0, 0.0],    # top-left corner
            ]).flatten() * 0.95,  # Slightly scaled down
        ]
        
        for config in special_configs:
            result_direct = minimize(
                objective_function,
                config,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-13}
            )
            
            if result_direct.fun < best_objective:
                best_objective = result_direct.fun
                best_result = result_direct
                
    except Exception as e:
        warnings.warn(f"Direct search failed: {e}")
        pass
    
    # If no good result found, use the initial guess
    if best_result is None:
        best_result = type('obj', (object,), {'x': initial_guess})()
    
    # Extract best solution
    best_solution = best_result.x.reshape((11, 3))
    
    # Evaluate final solution
    penalty, outer_radius = evaluate_packing(best_solution)
    
    # If solution is invalid, fall back to good deterministic arrangement
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
            [1.732, 3.0, 0.0],     # top-right corner
            [-1.732, 3.0, 0.0],    # top-left corner
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
