# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple, List
import random
import time
from itertools import combinations
import warnings
import cvxpy as cp
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Uses an advanced evolutionary approach with better initialization and optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Start timing
    start_time = time.time()
    
    # Try different rectangle dimensions to find optimal configuration
    best_result = None
    best_sum = 0
    
    # Test several width/height combinations more systematically
    # Focus on ratios that might yield better results - more concentrated around promising ranges
    ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # Add some key ratios that often work well for circle packing
    key_ratios = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    ratios.extend(key_ratios)
    
    # Also test some extreme ratios to explore more possibilities
    extreme_ratios = [0.5, 0.6, 0.7]
    ratios.extend(extreme_ratios)
    
    # Prioritize some ratios that historically work well for circle packing problems
    prioritized_ratios = [1.0, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5]
    
    for ratio in prioritized_ratios:
        width = 2 * ratio / (1 + ratio)
        height = 2 / (1 + ratio)
        
        # Initialize with a more sophisticated approach
        circles = initialize_advanced_placement(width, height, 21)
        
        # Optimize using multiple strategies with better parameter tuning
        optimized_circles = optimize_with_multiple_strategies(circles, width, height, start_time)
        
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = optimized_circles
            
        # Early stopping if we're running close to time limit
        if time.time() - start_time > 55:
            break
    
    # Try the remaining ratios for additional exploration
    for ratio in ratios:
        if ratio in prioritized_ratios:
            continue
            
        width = 2 * ratio / (1 + ratio)
        height = 2 / (1 + ratio)
        
        # Initialize with a more sophisticated approach
        circles = initialize_advanced_placement(width, height, 21)
        
        # Optimize using multiple strategies with better parameter tuning
        optimized_circles = optimize_with_multiple_strategies(circles, width, height, start_time)
        
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = optimized_circles
            
        # Early stopping if we're running close to time limit
        if time.time() - start_time > 55:
            break
    
    # If no good solution found, fallback to a robust initialization
    if best_result is None:
        width, height = 1.0, 1.0
        best_result = initialize_advanced_placement(width, height, 21)
    
    return best_result


def initialize_advanced_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circle positions using advanced systematic approach."""
    circles = np.zeros((n, 3))
    
    # For 21 circles, we'll use a hexagonal lattice arrangement with adjustments
    # Try to place in approximately 5 rows and 4 columns (but adjust for 21 circles)
    
    # Calculate ideal grid dimensions
    cols = max(1, int(np.sqrt(n)))
    rows = math.ceil(n / cols)
    
    # Make sure we have enough space
    while cols * rows < n:
        cols += 1
    
    # Calculate spacing
    cell_width = width / cols if cols > 0 else width
    cell_height = height / rows if rows > 0 else height
    
    # Use hexagonal packing pattern for better density
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Hexagonal offset
            x_offset = (j + 0.5) * cell_width
            if i % 2 == 1:
                x_offset += cell_width / 2
                
            y_offset = (i + 0.5) * cell_height
            
            # Ensure positions are within bounds with margin
            x = max(0.01, min(width - 0.01, x_offset))
            y = max(0.01, min(height - 0.01, y_offset))
            
            # Calculate initial radius based on proximity to edges and center
            # Distance to nearest edge
            dist_to_edge = min(x, width - x, y, height - y)
            
            # Distance to center
            center_x = width / 2
            center_y = height / 2
            dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt((width/2)**2 + (height/2)**2)
            
            # Radius based on distance from center (larger in center, smaller at edges)
            radius_factor = max(0.3, 1.0 - dist_to_center/max_dist)
            
            # Ensure radius is limited by proximity to edges
            max_radius = min(dist_to_edge, cell_width/3, cell_height/3)
            radius = min(max_radius, 0.15 * min(width, height)) * radius_factor
            
            circles[idx] = [x, y, max(radius, 0.001)]
            idx += 1
    
    # Fill remaining circles with more strategic placement
    if idx < n:
        # Place remaining circles in strategic locations
        for i in range(idx, n):
            # Try to place in corners first
            corner_positions = [
                (0.1*width, 0.1*height),      # top-left
                (0.9*width, 0.1*height),      # top-right
                (0.1*width, 0.9*height),      # bottom-left
                (0.9*width, 0.9*height),      # bottom-right
            ]
            
            # Use one of the corner positions or spread out randomly
            if i < len(corner_positions):
                x, y = corner_positions[i]
            else:
                # Random placement near edges
                edge_type = random.randint(0, 3)
                if edge_type == 0:  # top edge
                    x = random.uniform(0.1*width, 0.9*width)
                    y = 0.1*height
                elif edge_type == 1:  # right edge
                    x = 0.9*width
                    y = random.uniform(0.1*height, 0.9*height)
                elif edge_type == 2:  # bottom edge
                    x = random.uniform(0.1*width, 0.9*width)
                    y = 0.9*height
                else:  # left edge
                    x = 0.1*width
                    y = random.uniform(0.1*height, 0.9*height)
            
            # Small radius for edge circles
            circles[i] = [x, y, min(0.02 * min(width, height), 0.05)]
    
    return circles


def optimize_with_multiple_strategies(initial_circles: np.ndarray, width: float, height: float, start_time: float) -> np.ndarray:
    """Use multiple optimization strategies to find the best solution."""
    best_solution = initial_circles.copy()
    best_sum = np.sum(initial_circles[:, 2])
    
    # Strategy 1: Differential Evolution with aggressive parameters
    try:
        strategy_1_result = optimize_de_aggressive(initial_circles, width, height, start_time)
        strategy_1_sum = np.sum(strategy_1_result[:, 2])
        if strategy_1_sum > best_sum:
            best_sum = strategy_1_sum
            best_solution = strategy_1_result
    except Exception as e:
        pass
    
    # Strategy 2: Local optimization with SLSQP (more aggressive)
    try:
        strategy_2_result = optimize_slsqp_aggressive(initial_circles, width, height, start_time)
        strategy_2_sum = np.sum(strategy_2_result[:, 2])
        if strategy_2_sum > best_sum:
            best_sum = strategy_2_sum
            best_solution = strategy_2_result
    except Exception as e:
        pass
    
    # Strategy 3: Another DE run with different parameters
    try:
        strategy_3_result = optimize_de_focused(initial_circles, width, height, start_time)
        strategy_3_sum = np.sum(strategy_3_result[:, 2])
        if strategy_3_sum > best_sum:
            best_sum = strategy_3_sum
            best_solution = strategy_3_result
    except Exception as e:
        pass
    
    # Strategy 4: Try a completely fresh approach with better initial conditions
    try:
        fresh_start = initialize_better_placement(width, height, 21)
        strategy_4_result = optimize_de_aggressive(fresh_start, width, height, start_time)
        strategy_4_sum = np.sum(strategy_4_result[:, 2])
        if strategy_4_sum > best_sum:
            best_sum = strategy_4_sum
            best_solution = strategy_4_result
    except Exception as e:
        pass
    
    # Strategy 5: Additional refinement with simulated annealing-inspired approach
    try:
        strategy_5_result = optimize_with_local_refinement(best_solution, width, height, start_time)
        strategy_5_sum = np.sum(strategy_5_result[:, 2])
        if strategy_5_sum > best_sum:
            best_sum = strategy_5_sum
            best_solution = strategy_5_result
    except Exception as e:
        pass
    
    # Strategy 6: Use a convex optimization approach for final refinement
    try:
        strategy_6_result = optimize_with_cvxpy(best_solution, width, height, start_time)
        strategy_6_sum = np.sum(strategy_6_result[:, 2])
        if strategy_6_sum > best_sum:
            best_sum = strategy_6_sum
            best_solution = strategy_6_result
    except Exception as e:
        pass
    
    return best_solution


def initialize_better_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize with even better placement strategy."""
    circles = np.zeros((n, 3))
    
    # Try to use a more sophisticated hexagonal arrangement
    # This is a classic circle packing approach for small numbers
    
    # For 21 circles, try a pattern similar to 3x7 or 4x5 arrangement
    rows = 5
    cols = 4
    if rows * cols < n:
        rows = 6
        cols = 4
    if rows * cols < n:
        rows = 7
        cols = 3
    
    # Calculate spacing
    cell_width = width / cols if cols > 0 else width
    cell_height = height / rows if rows > 0 else height
    
    # Better hexagonal packing
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Hexagonal offset
            x_offset = (j + 0.5) * cell_width
            if i % 2 == 1:
                x_offset += cell_width / 2
                
            y_offset = (i + 0.5) * cell_height
            
            # Ensure positions are within bounds with margin
            x = max(0.01, min(width - 0.01, x_offset))
            y = max(0.01, min(height - 0.01, y_offset))
            
            # Calculate initial radius - focus on larger values in center area
            dist_to_center = np.sqrt((x - width/2)**2 + (y - height/2)**2)
            max_dist = np.sqrt((width/2)**2 + (height/2)**2)
            
            # Radius inversely proportional to distance from center
            radius = 0.1 * min(width, height) * (1.0 - dist_to_center/max_dist)
            radius = max(radius, 0.005)  # minimum radius
            
            # Also consider proximity to edges
            dist_to_edge = min(x, width - x, y, height - y)
            radius = min(radius, dist_to_edge * 0.8)
            
            circles[idx] = [x, y, radius]
            idx += 1
    
    # Fill remaining with carefully placed edge circles
    if idx < n:
        for i in range(idx, n):
            # Place in corners and along edges strategically
            if i < 4:
                # Corners
                corners = [(0.1*width, 0.1*height), (0.9*width, 0.1*height), 
                          (0.1*width, 0.9*height), (0.9*width, 0.9*height)]
                x, y = corners[i]
            else:
                # Spread evenly along edges
                edge_num = i % 4
                if edge_num == 0:  # top
                    x = 0.1*width + 0.2*(i//4)*width
                    y = 0.1*height
                elif edge_num == 1:  # right
                    x = 0.9*width
                    y = 0.1*height + 0.2*(i//4)*height
                elif edge_num == 2:  # bottom
                    x = 0.1*width + 0.2*(i//4)*width
                    y = 0.9*height
                else:  # left
                    x = 0.1*width
                    y = 0.1*height + 0.2*(i//4)*height
                    
            circles[i] = [x, y, min(0.015 * min(width, height), 0.03)]
    
    return circles


def optimize_de_aggressive(initial_circles: np.ndarray, width: float, height: float, start_time: float) -> np.ndarray:
    """Optimize using differential evolution with aggressive parameters."""
    n = len(initial_circles)
    
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        
        # Vectorized constraint checking for bounds
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary constraints: x-r >= 0, width-x-r >= 0, y-r >= 0, height-y-r >= 0
        bound_constraints = np.concatenate([
            x - r,                    # x - r >= 0
            width - x - r,            # width - x - r >= 0  
            y - r,                    # y - r >= 0
            height - y - r            # height - y - r >= 0
        ])
        
        # Overlap constraints - compute all pairwise distances efficiently
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Create overlap constraints for all pairs (only upper triangle to avoid duplicates)
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r1 = circles[i, 2]
                r2 = circles[j, 2]
                # We want dist >= r1 + r2, so dist - r1 - r2 >= 0
                overlap_constraints.append(dist - r1 - r2)
        
        return np.concatenate([bound_constraints, overlap_constraints])
    
    # Create bounds for optimization - more precise bounds
    bounds = []
    for i in range(n):
        # x bounds: [r, width-r] 
        bounds.extend([(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)])
    
    # Define constraints for optimization
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Run differential evolution with aggressive parameters
    result = differential_evolution(
        objective,
        bounds,
        constraints=cons,
        seed=42,
        maxiter=300,      # More iterations for better convergence
        popsize=50,       # Even larger population
        mutation=(0.95, 1),  # Very aggressive mutation
        recombination=0.99,
        atol=1e-10,
        tol=1e-10,
        callback=lambda x, convergence: time.time() - start_time > 50  # Early termination
    )
    
    if result.success:
        return result.x.reshape(-1, 3)
    else:
        return initial_circles


def optimize_de_focused(initial_circles: np.ndarray, width: float, height: float, start_time: float) -> np.ndarray:
    """Optimize using differential evolution with focused parameters."""
    n = len(initial_circles)
    
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        
        # Vectorized constraint checking for bounds
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary constraints
        bound_constraints = np.concatenate([
            x - r,                    # x - r >= 0
            width - x - r,            # width - x - r >= 0  
            y - r,                    # y - r >= 0
            height - y - r            # height - y - r >= 0
        ])
        
        # Overlap constraints using distance matrix
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Create overlap constraints for all pairs
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r1 = circles[i, 2]
                r2 = circles[j, 2]
                overlap_constraints.append(dist - r1 - r2)
        
        return np.concatenate([bound_constraints, overlap_constraints])
    
    # Create bounds
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)])
    
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Run with different parameters - focused on exploitation
    result = differential_evolution(
        objective,
        bounds,
        constraints=cons,
        seed=42,
        maxiter=250,
        popsize=40,
        mutation=(0.8, 1),
        recombination=0.95,
        atol=1e-9,
        tol=1e-9,
        callback=lambda x, convergence: time.time() - start_time > 50
    )
    
    if result.success:
        return result.x.reshape(-1, 3)
    else:
        return initial_circles


def optimize_slsqp_aggressive(initial_circles: np.ndarray, width: float, height: float, start_time: float) -> np.ndarray:
    """Optimize using SLSQP with aggressive parameters."""
    n = len(initial_circles)
    
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        
        # Vectorized constraint checking for bounds
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary constraints
        bound_constraints = np.concatenate([
            x - r,                    # x - r >= 0
            width - x - r,            # width - x - r >= 0  
            y - r,                    # y - r >= 0
            height - y - r            # height - y - r >= 0
        ])
        
        # Overlap constraints using distance matrix
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Create overlap constraints for all pairs
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r1 = circles[i, 2]
                r2 = circles[j, 2]
                overlap_constraints.append(dist - r1 - r2)
        
        return np.concatenate([bound_constraints, overlap_constraints])
    
    # Create bounds
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)])
    
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Local optimization with more aggressive settings
    result = minimize(
        objective,
        initial_circles.flatten(),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1500, 'ftol': 1e-10, 'eps': 1e-10, 'iprint': -1},
        callback=lambda x: time.time() - start_time > 50  # Early termination
    )
    
    if result.success:
        return result.x.reshape(-1, 3)
    else:
        return initial_circles


def optimize_with_local_refinement(initial_circles: np.ndarray, width: float, height: float, start_time: float) -> np.ndarray:
    """Apply local refinement to improve the solution."""
    # Create a simpler optimization approach for local refinement
    n = len(initial_circles)
    
    # Use a hybrid approach with a few iterations of gradient-based optimization
    # but also maintain constraint handling
    
    # First create a more intelligent initial guess with better distribution
    refined_circles = initial_circles.copy()
    
    # Apply small perturbations to see if we can improve
    for _ in range(100):  # More iterations for better refinement
        if time.time() - start_time > 55:
            break
            
        # Perturb one circle at a time
        circle_idx = random.randint(0, n-1)
        perturbation_magnitude = 0.01 * min(width, height)
        
        # Try small random moves
        dx = random.uniform(-perturbation_magnitude, perturbation_magnitude)
        dy = random.uniform(-perturbation_magnitude, perturbation_magnitude)
        dr = random.uniform(-0.005, 0.005)  # Small change to radius
        
        # Apply the perturbation
        old_x, old_y, old_r = refined_circles[circle_idx]
        new_x = max(0.01, min(width - 0.01, old_x + dx))
        new_y = max(0.01, min(height - 0.01, old_y + dy))
        new_r = max(0.001, old_r + dr)
        
        # Check if this improves the configuration
        # Temporarily update
        refined_circles[circle_idx] = [new_x, new_y, new_r]
        
        # Check constraints
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((refined_circles[i, 0] - refined_circles[j, 0])**2 + 
                              (refined_circles[i, 1] - refined_circles[j, 1])**2)
                if dist < refined_circles[i, 2] + refined_circles[j, 2]:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Revert if invalid
            refined_circles[circle_idx] = [old_x, old_y, old_r]
    
    return refined_circles


def optimize_with_cvxpy(initial_circles: np.ndarray, width: float, height: float, start_time: float) -> np.ndarray:
    """Use convex optimization for final refinement."""
    try:
        n = len(initial_circles)
        # Use CVXPY for potentially better optimization
        circles = initial_circles.copy()
        
        # Convert to CVXPY variables
        x = cp.Variable(n)
        y = cp.Variable(n)
        r = cp.Variable(n)
        
        # Objective: maximize sum of radii
        objective = cp.Maximize(cp.sum(r))
        
        # Constraints
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            constraints.append(x[i] >= r[i])
            constraints.append(y[i] >= r[i])
            constraints.append(x[i] <= width - r[i])
            constraints.append(y[i] <= height - r[i])
        
        # Non-overlapping constraints
        for i in range(n):
            for j in range(i+1, n):
                # Distance constraint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                constraints.append(dist_sq >= (r[i] + r[j])**2)
        
        # Create problem and solve
        prob = cp.Problem(objective, constraints)
        
        # Solve with different solvers
        prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        
        # Extract solution
        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            new_circles = circles.copy()
            for i in range(n):
                new_circles[i] = [x.value[i], y.value[i], r.value[i]]
            return new_circles
        else:
            return initial_circles
            
    except Exception:
        return initial_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
