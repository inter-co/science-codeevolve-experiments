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
warnings.filterwarnings('ignore')

# Use more efficient optimization libraries
try:
    import nevergrad as ng
except ImportError:
    pass

# Import additional optimization tools
from scipy.optimize import dual_annealing
from deap import base, creator, tools, algorithms
import multiprocessing as mp

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Uses an advanced hybrid approach combining global and local optimization with better initialization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Start timing
    start_time = time.time()
    
    # Try different rectangle dimensions to find optimal configuration
    best_result = None
    best_sum = 0
    
    # Focus on promising ratios that tend to work well for circle packing
    ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0]
    
    # Add some extreme ratios to explore more thoroughly
    extreme_ratios = [0.5, 0.6, 0.7, 5.0, 6.0, 8.0, 10.0]
    ratios.extend(extreme_ratios)
    
    # Also test some square-like ratios for comparison
    square_ratios = [0.95, 1.0, 1.05, 1.1, 1.2]
    ratios.extend(square_ratios)
    
    # Test a few key ratios more intensively
    key_ratios = [1.333, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    ratios.extend(key_ratios)
    
    for ratio in ratios:
        width = 2 * ratio / (1 + ratio)
        height = 2 / (1 + ratio)
        
        # Try multiple initialization strategies
        init_strategies = [
            lambda w, h, n: initialize_hexagonal_placement(w, h, n),
            lambda w, h, n: initialize_better_placement(w, h, n),
            lambda w, h, n: initialize_random_placement(w, h, n),
            lambda w, h, n: initialize_optimized_placement(w, h, n)
        ]
        
        for init_func in init_strategies:
            try:
                circles = init_func(width, height, 21)
                
                # Optimize using multiple strategies with better parameter tuning
                optimized_circles = optimize_with_multiple_strategies(circles, width, height, start_time)
                
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = optimized_circles
                    
                # Early stopping if we're running close to time limit
                if time.time() - start_time > 55:
                    break
            except Exception as e:
                continue
        
        if time.time() - start_time > 55:
            break
    
    # If no good solution found, fallback to a robust initialization
    if best_result is None:
        width, height = 1.0, 1.0
        best_result = initialize_hexagonal_placement(width, height, 21)
    
    return best_result


def initialize_hexagonal_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circle positions using a hexagonal lattice arrangement."""
    circles = np.zeros((n, 3))
    
    # Try a hexagonal packing approach - most efficient for dense arrangements
    # For 21 circles, we can try 4 rows of 5-6 circles each or similar
    
    # Calculate how many rows and columns we need
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Adjust to fit exactly 21 circles
    while rows * cols < n:
        rows += 1
    
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
            
            # Calculate initial radius based on position
            # Center of rectangle
            center_x = width / 2
            center_y = height / 2
            dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt((width/2)**2 + (height/2)**2)
            
            # Radius inversely proportional to distance from center
            radius = 0.15 * min(width, height) * (1.0 - dist_to_center/max_dist)
            radius = max(radius, 0.005)  # minimum radius
            
            # Also consider proximity to edges
            dist_to_edge = min(x, width - x, y, height - y)
            radius = min(radius, dist_to_edge * 0.8)
            radius = min(radius, cell_width/3, cell_height/3)
            
            circles[idx] = [x, y, max(radius, 0.001)]
            idx += 1
    
    return circles


def initialize_better_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize with a more sophisticated placement strategy."""
    circles = np.zeros((n, 3))
    
    # Try to place in a pattern that maximizes space utilization
    # Use a combination of grid and radial placement
    
    # First try a grid-based approach with some hexagonal offset
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
            radius = 0.12 * min(width, height) * (1.0 - dist_to_center/max_dist)
            radius = max(radius, 0.005)  # minimum radius
            
            # Also consider proximity to edges
            dist_to_edge = min(x, width - x, y, height - y)
            radius = min(radius, dist_to_edge * 0.7)
            
            circles[idx] = [x, y, radius]
            idx += 1
    
    # Fill remaining circles strategically
    if idx < n:
        # Place remaining circles near edges but not too close to corners
        edge_positions = []
        
        # Top edge
        for i in range(min(5, n - idx)):
            edge_positions.append((0.1*width + 0.2*i*width, 0.1*height))
        
        # Right edge  
        for i in range(min(5, n - idx)):
            edge_positions.append((0.9*width, 0.1*height + 0.2*i*height))
            
        # Bottom edge
        for i in range(min(5, n - idx)):
            edge_positions.append((0.1*width + 0.2*i*width, 0.9*height))
            
        # Left edge
        for i in range(min(5, n - idx)):
            edge_positions.append((0.1*width, 0.1*height + 0.2*i*height))
        
        # Fill with edge positions
        for i in range(idx, n):
            if i - idx < len(edge_positions):
                x, y = edge_positions[i - idx]
            else:
                # Fallback to random positions near edges
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
            
            # Small but reasonable radius for edge circles
            circles[i] = [x, y, min(0.01 * min(width, height), 0.03)]
    
    return circles


def initialize_optimized_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize with a more optimized placement strategy."""
    circles = np.zeros((n, 3))
    
    # Use a more strategic approach with different patterns
    # Try to place in a way that balances center and edge placement
    
    # Create a strategic pattern
    # Place 9 circles in a central region with high radii
    # Place 8 circles around the edges
    # Place 4 circles in corners
    
    # Central region circles (9 circles)
    central_rows, central_cols = 3, 3
    central_width = width * 0.6
    central_height = height * 0.6
    central_x_start = (width - central_width) / 2
    central_y_start = (height - central_height) / 2
    
    cell_width = central_width / central_cols if central_cols > 0 else central_width
    cell_height = central_height / central_rows if central_rows > 0 else central_height
    
    idx = 0
    for i in range(central_rows):
        for j in range(central_cols):
            if idx >= 9:
                break
            x = central_x_start + (j + 0.5) * cell_width
            y = central_y_start + (i + 0.5) * cell_height
            
            # High radius in center
            dist_to_center = np.sqrt((x - width/2)**2 + (y - height/2)**2)
            max_dist = np.sqrt((width/2)**2 + (height/2)**2)
            radius = 0.15 * min(width, height) * (1.0 - dist_to_center/max_dist)
            radius = max(radius, 0.005)
            
            # Consider edge proximity
            dist_to_edge = min(x, width - x, y, height - y)
            radius = min(radius, dist_to_edge * 0.7)
            
            circles[idx] = [x, y, radius]
            idx += 1
    
    # Edge circles (8 circles)
    edge_positions = []
    
    # Top edge (excluding corners)
    for i in range(1, 4):
        edge_positions.append((0.2 * width + 0.2 * i * width, 0.1 * height))
    
    # Right edge (excluding corners)  
    for i in range(1, 3):
        edge_positions.append((0.9 * width, 0.2 * height + 0.2 * i * height))
    
    # Bottom edge (excluding corners)
    for i in range(1, 4):
        edge_positions.append((0.2 * width + 0.2 * i * width, 0.9 * height))
    
    # Left edge (excluding corners)
    for i in range(1, 3):
        edge_positions.append((0.1 * width, 0.2 * height + 0.2 * i * height))
    
    # Fill with edge positions
    for i in range(9, min(17, n)):
        if i - 9 < len(edge_positions):
            x, y = edge_positions[i - 9]
        else:
            # Fallback to random positions near edges
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
        
        # Smaller but reasonable radius for edge circles
        circles[i] = [x, y, min(0.008 * min(width, height), 0.02)]
    
    # Corner circles (4 circles)
    corner_positions = [
        (0.1 * width, 0.1 * height),   # top-left
        (0.9 * width, 0.1 * height),   # top-right
        (0.1 * width, 0.9 * height),   # bottom-left
        (0.9 * width, 0.9 * height)    # bottom-right
    ]
    
    for i in range(17, min(21, n)):
        if i - 17 < len(corner_positions):
            x, y = corner_positions[i - 17]
        else:
            # Fallback to random positions
            x = random.uniform(0.05*width, 0.95*width)
            y = random.uniform(0.05*height, 0.95*height)
        
        # Very small radius for corner circles
        circles[i] = [x, y, min(0.005 * min(width, height), 0.01)]
    
    return circles


def initialize_random_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize with random placements and then optimize."""
    circles = np.zeros((n, 3))
    
    # Place circles randomly but ensure they don't overlap initially
    for i in range(n):
        attempts = 0
        placed = False
        
        while not placed and attempts < 100:
            # Random position
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            
            # Random radius (smaller than average)
            max_radius = min(x, width - x, y, height - y)
            radius = random.uniform(0.005, min(0.05, max_radius * 0.5))
            
            # Check if this overlaps with existing circles
            valid_position = True
            for j in range(i):
                existing_x, existing_y, existing_r = circles[j]
                dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if dist < (radius + existing_r):
                    valid_position = False
                    break
            
            if valid_position:
                circles[i] = [x, y, radius]
                placed = True
            else:
                attempts += 1
        
        if not placed:
            # Fallback to systematic placement
            row = i // 5
            col = i % 5
            x = 0.1 * width + 0.2 * col * width
            y = 0.1 * height + 0.2 * row * height
            radius = 0.02
            circles[i] = [x, y, radius]
    
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
    
    # Strategy 4: Try a fresh approach with better initial conditions
    try:
        fresh_start = initialize_hexagonal_placement(width, height, 21)
        strategy_4_result = optimize_de_aggressive(fresh_start, width, height, start_time)
        strategy_4_sum = np.sum(strategy_4_result[:, 2])
        if strategy_4_sum > best_sum:
            best_sum = strategy_4_sum
            best_solution = strategy_4_result
    except Exception as e:
        pass
    
    # Strategy 5: Try Nevergrad if available for global optimization
    try:
        strategy_5_result = optimize_nevergrad(initial_circles, width, height, start_time)
        strategy_5_sum = np.sum(strategy_5_result[:, 2])
        if strategy_5_sum > best_sum:
            best_sum = strategy_5_sum
            best_solution = strategy_5_result
    except Exception as e:
        pass
    
    # Strategy 6: Try dual annealing for global optimization
    try:
        strategy_6_result = optimize_dual_annealing(initial_circles, width, height, start_time)
        strategy_6_sum = np.sum(strategy_6_result[:, 2])
        if strategy_6_sum > best_sum:
            best_sum = strategy_6_sum
            best_solution = strategy_6_result
    except Exception as e:
        pass
    
    return best_solution


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
        mutation=(0.9, 1),  # Aggressive mutation
        recombination=0.95,
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


def optimize_nevergrad(initial_circles: np.ndarray, width: float, height: float, start_time: float) -> np.ndarray:
    """Optimize using Nevergrad if available."""
    try:
        # Try using Nevergrad for potentially better global optimization
        n = len(initial_circles)
        
        def objective(params):
            circles = params.reshape(-1, 3)
            # Maximize sum of radii
            return -np.sum(circles[:, 2])
        
        # Define bounds
        bounds = []
        for i in range(n):
            bounds.extend([(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)])
        
        # Create parameter space
        instrumentation = ng.p.Array(shape=(n * 3,))
        
        # Create optimizer with higher budget
        optimizer = ng.optimizers.CMA(instrumentation=instrumentation, budget=300)
        
        # Convert initial circles to flat array
        initial_flat = initial_circles.flatten()
        
        # Optimize
        recommendation = optimizer.minimize(objective, initial=initial_flat)
        
        result = recommendation.value.reshape(-1, 3)
        return result
    except:
        # Fall back to the previous approach if Nevergrad isn't available or fails
        return initial_circles


def optimize_dual_annealing(initial_circles: np.ndarray, width: float, height: float, start_time: float) -> np.ndarray:
    """Optimize using dual annealing for global optimization."""
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
    
    # Define constraints
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Run dual annealing
    result = dual_annealing(
        objective,
        bounds,
        constraints=cons,
        maxiter=500,
        maxfun=1000,
        seed=42,
        callback=lambda x, f, context: time.time() - start_time > 50
    )
    
    if result.success:
        return result.x.reshape(-1, 3)
    else:
        return initial_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
