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
import copy
from numba import jit
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_distances_numba(positions):
    """Compute pairwise distances efficiently using Numba"""
    n = positions.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

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
    # Focus on ratios that might yield better results
    ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0]
    
    # Also test some extreme ratios to explore more possibilities
    extreme_ratios = [0.5, 0.6, 0.7, 5.0, 6.0, 8.0, 10.0]
    ratios.extend(extreme_ratios)
    
    # Try some specific ratios that are known to work well for circle packing
    special_ratios = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ratios.extend(special_ratios)
    
    # Add some ratios that have shown good results in previous tests
    additional_ratios = [1.33, 1.6, 2.1, 2.4, 2.7, 3.2, 3.8]
    ratios.extend(additional_ratios)
    
    for ratio in ratios:
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
    
    # Use a more refined hexagonal packing approach
    # For 21 circles, we can try 3 rows of 7 circles, or 4 rows of 5+1 circles
    
    # Try different arrangements to find the most promising
    arrangements = [
        (3, 7),  # 3 rows, 7 columns
        (4, 5),  # 4 rows, 5 columns  
        (5, 4),  # 5 rows, 4 columns
        (7, 3),  # 7 rows, 3 columns
        (6, 4),  # 6 rows, 4 columns
        (4, 6),  # 4 rows, 6 columns
    ]
    
    best_arrangement = None
    best_density = 0
    
    for rows, cols in arrangements:
        if rows * cols >= n:
            # Calculate how much space we have
            cell_width = width / cols if cols > 0 else width
            cell_height = height / rows if rows > 0 else height
            
            # Check density
            avg_cell_area = cell_width * cell_height
            total_circle_area = n * (min(cell_width, cell_height) * 0.2)**2 * np.pi  # rough estimate
            density = total_circle_area / avg_cell_area
            
            if density > best_density:
                best_density = density
                best_arrangement = (rows, cols)
    
    # Use the best arrangement found
    if best_arrangement is not None:
        rows, cols = best_arrangement
    else:
        # Fallback to a simple square arrangement
        rows = max(1, int(np.sqrt(n)))
        cols = math.ceil(n / rows)
    
    # Calculate spacing
    cell_width = width / cols if cols > 0 else width
    cell_height = height / rows if rows > 0 else height
    
    # Better hexagonal packing with proper offsets
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Hexagonal offset for even rows
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
            # But with more emphasis on central region
            radius_factor = max(0.2, 1.0 - dist_to_center/max_dist * 0.8)
            
            # Ensure radius is limited by proximity to edges
            max_radius = min(dist_to_edge, cell_width/3, cell_height/3)
            radius = min(max_radius, 0.1 * min(width, height)) * radius_factor
            
            # Add some randomness to avoid perfect patterns that might get stuck
            radius *= random.uniform(0.9, 1.1)
            
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
                # Random placement near edges but not too close to corners
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
            circles[i] = [x, y, min(0.01 * min(width, height), 0.03)]
    
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
    
    # Strategy 5: Simulated Annealing approach for better exploration
    try:
        strategy_5_result = optimize_simulated_annealing(initial_circles, width, height, start_time)
        strategy_5_sum = np.sum(strategy_5_result[:, 2])
        if strategy_5_sum > best_sum:
            best_sum = strategy_5_sum
            best_solution = strategy_5_result
    except Exception as e:
        pass
    
    return best_solution


def initialize_better_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize with even better placement strategy."""
    circles = np.zeros((n, 3))
    
    # Use a more systematic approach inspired by known good circle packings
    # For 21 circles, a 3x7 grid with slight hexagonal adjustment works well
    
    rows = 3
    cols = 7
    
    # Calculate spacing
    cell_width = width / cols if cols > 0 else width
    cell_height = height / rows if rows > 0 else height
    
    # Better hexagonal packing with proper adjustments
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
            
            # Radius inversely proportional to distance from center with more variation
            radius = 0.12 * min(width, height) * (1.0 - dist_to_center/max_dist * 0.7)
            radius = max(radius, 0.005)  # minimum radius
            
            # Also consider proximity to edges
            dist_to_edge = min(x, width - x, y, height - y)
            radius = min(radius, dist_to_edge * 0.8)
            
            # Add some randomness to escape local optima
            radius *= random.uniform(0.85, 1.15)
            
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
                    
            circles[i] = [x, y, min(0.012 * min(width, height), 0.025)]
    
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
        mutation=(0.98, 1),  # Very aggressive mutation
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
        mutation=(0.7, 1),
        recombination=0.9,
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


def optimize_simulated_annealing(initial_circles: np.ndarray, width: float, height: float, start_time: float) -> np.ndarray:
    """Optimize using simulated annealing approach."""
    import random
    import math
    
    n = len(initial_circles)
    
    def calculate_total_radius(circles):
        return np.sum(circles[:, 2])
    
    def calculate_penalty(circles):
        penalty = 0
        # Boundary penalties
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                penalty += 1000  # Large penalty for boundary violations
                
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < r1 + r2:
                    penalty += 1000 * (r1 + r2 - dist)  # Penalty based on overlap amount
                    
        return penalty
    
    def objective(circles):
        return -calculate_total_radius(circles) + calculate_penalty(circles)
    
    # Start with initial solution
    current_solution = initial_circles.copy()
    current_energy = objective(current_solution)
    
    # Parameters for simulated annealing
    temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 1e-10
    max_iterations = 15000
    
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    iteration = 0
    while temperature > min_temperature and iteration < max_iterations and time.time() - start_time < 55:
        # Generate neighbor solution
        new_solution = current_solution.copy()
        
        # Pick a random circle to perturb
        idx = random.randint(0, n-1)
        
        # Perturb position and/or radius
        if random.random() < 0.5:
            # Perturb position
            new_solution[idx, 0] += random.uniform(-0.08, 0.08) * width
            new_solution[idx, 1] += random.uniform(-0.08, 0.08) * height
            # Keep within bounds
            new_solution[idx, 0] = np.clip(new_solution[idx, 0], 0.01, width - 0.01)
            new_solution[idx, 1] = np.clip(new_solution[idx, 1], 0.01, height - 0.01)
        else:
            # Perturb radius
            new_solution[idx, 2] *= random.uniform(0.7, 1.3)
            new_solution[idx, 2] = np.clip(new_solution[idx, 2], 0.001, min(width, height) / 2)
        
        # Evaluate new solution
        new_energy = objective(new_solution)
        
        # Accept or reject the new solution
        if new_energy < current_energy or random.random() < math.exp(-(new_energy - current_energy) / temperature):
            current_solution = new_solution
            current_energy = new_energy
            
            # Update best solution
            if new_energy < best_energy:
                best_solution = new_solution.copy()
                best_energy = new_energy
        
        # Cool down
        temperature *= cooling_rate
        iteration += 1
    
    return best_solution


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
