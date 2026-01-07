# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import warnings
from itertools import combinations
import time

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with gradient-based optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Multi-start approach: try many different initializations with more thorough search
    best_circles = None
    best_sum = 0
    
    # Try multiple initialization strategies with different seeds and more combinations
    seeds = [42, 123, 456, 789, 999, 111, 222, 333]
    strategies = ['hexagonal', 'random', 'grid']
    
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        
        for strategy in strategies:
            if strategy == 'hexagonal':
                circles = initialize_circles_hexagonal(n)
            elif strategy == 'random':
                circles = initialize_circles_random(n)
            else:  # grid
                circles = initialize_circles_grid_adaptive(n)
            
            optimized = optimize_circles(circles)
            sum_val = np.sum(optimized[:, 2])
            
            if sum_val > best_sum:
                best_sum = sum_val
                best_circles = optimized.copy()
    
    # Additional optimization: try one more round with the best found so far
    if best_circles is not None:
        # Try with more aggressive optimization settings
        more_aggressive = optimize_circles_more_aggressive(best_circles)
        more_aggressive_sum = np.sum(more_aggressive[:, 2])
        if more_aggressive_sum > best_sum:
            best_circles = more_aggressive
    
    return best_circles if best_circles is not None else initialize_circles_hexagonal(n)


def initialize_circles_hexagonal(n: int) -> np.ndarray:
    """Initialize circles using a more sophisticated hexagonal packing pattern"""
    circles = np.zeros((n, 3))
    
    # More careful hexagonal packing
    sqrt3 = np.sqrt(3)
    rows = int(np.ceil(np.sqrt(n * 2 / sqrt3)))  # Adjusted for hexagonal efficiency
    cols = int(np.ceil(n / rows))
    
    # Grid dimensions
    margin = 0.05
    grid_width = 1 - 2 * margin
    grid_height = 1 - 2 * margin
    
    cell_width = grid_width / cols
    cell_height = grid_height / rows
    
    # Use smaller cell height to approximate hexagonal packing
    cell_height_eff = cell_height * sqrt3 / 2
    
    # Fill grid with circles
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset odd rows for hexagonal packing
            x_offset = (i % 2) * cell_width / 2
            x = margin + j * cell_width + x_offset
            y = margin + i * cell_height_eff
            
            # Set initial radius based on spacing
            radius = min(cell_width, cell_height_eff) * 0.4
            
            # Ensure we're within bounds
            x = max(radius, min(1-radius, x))
            y = max(radius, min(1-radius, y))
            
            circles[idx] = [x, y, radius]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining circles with random positions but reasonable radii
    for i in range(idx, n):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        radius = random.uniform(0.01, 0.1)
        circles[i] = [x, y, radius]
    
    return circles


def initialize_circles_random(n: int) -> np.ndarray:
    """Initialize circles with random placement but better constraints"""
    circles = np.zeros((n, 3))
    
    # Start with some structured placement
    for i in range(n):
        # Try to place in a way that reduces overlap probability
        attempts = 0
        placed = False
        while not placed and attempts < 100:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            radius = random.uniform(0.01, 0.15)
            
            # Check if this placement would be valid with existing circles
            valid = True
            for j in range(i):
                existing_x, existing_y, existing_r = circles[j]
                distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if distance < (radius + existing_r):
                    valid = False
                    break
            
            if valid:
                # Make sure it's within bounds
                x = max(radius, min(1-radius, x))
                y = max(radius, min(1-radius, y))
                circles[i] = [x, y, radius]
                placed = True
            attempts += 1
        
        # If couldn't place properly, use fallback
        if not placed:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            radius = random.uniform(0.01, 0.1)
            circles[i] = [x, y, radius]
    
    return circles


def initialize_circles_grid_adaptive(n: int) -> np.ndarray:
    """Initialize circles with grid-based approach that adapts to circle count"""
    circles = np.zeros((n, 3))
    
    # Calculate grid dimensions based on circle count
    side_length = int(np.ceil(np.sqrt(n)))
    if side_length * side_length < n:
        side_length += 1
    
    margin = 0.05
    grid_width = 1 - 2 * margin
    cell_size = grid_width / side_length
    
    # Place circles in a grid with some randomness
    idx = 0
    for i in range(side_length):
        for j in range(side_length):
            if idx >= n:
                break
            x = margin + (j + 0.5) * cell_size
            y = margin + (i + 0.5) * cell_size
            
            # Add slight randomness to avoid perfect grid
            x += random.uniform(-cell_size*0.1, cell_size*0.1)
            y += random.uniform(-cell_size*0.1, cell_size*0.1)
            
            # Initial radius based on cell size
            radius = cell_size * 0.3
            
            # Ensure within bounds
            x = max(radius, min(1-radius, x))
            y = max(radius, min(1-radius, y))
            
            circles[idx] = [x, y, radius]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining with random
    for i in range(idx, n):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        radius = random.uniform(0.01, 0.1)
        circles[i] = [x, y, radius]
    
    return circles


def optimize_circles_more_aggressive(initial_circles: np.ndarray) -> np.ndarray:
    """Try more aggressive optimization with different parameters"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define bounds for each parameter (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0.001, 0.999))  # Keep small margin to prevent boundary issues
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds
        bounds.append((0.001, 0.499))
    
    # Optimization function
    def objective(params):
        # Convert params back to circles array
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [params[3*i], params[3*i+1], params[3*i+2]]
        
        # Return negative sum of radii (since we want to maximize)
        return -np.sum(circles[:, 2])
    
    # Constraint functions
    def create_constraints():
        constraints = []
        
        # Non-overlap constraints
        def non_overlap_constraint(params):
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [params[3*i], params[3*i+1], params[3*i+2]]
            
            violations = []
            for i, j in combinations(range(n), 2):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                radii_sum = circles[i, 2] + circles[j, 2]
                violation = distance - radii_sum
                violations.append(violation)
            
            return np.array(violations)
        
        # Containment constraints
        def containment_constraint(params):
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [params[3*i], params[3*i+1], params[3*i+2]]
            
            violations = []
            for i in range(n):
                x, y, r = circles[i]
                violations.append(x - r)       # x >= r
                violations.append(y - r)       # y >= r
                violations.append(1 - x - r)   # x <= 1-r
                violations.append(1 - y - r)   # y <= 1-r
                
            return np.array(violations)
        
        return [
            {'type': 'ineq', 'fun': non_overlap_constraint},
            {'type': 'ineq', 'fun': containment_constraint}
        ]
    
    cons = create_constraints()
    
    # Try with different optimization methods and more iterations
    result = None
    try:
        # Try with higher iteration limits and different approaches
        result = minimize(objective, initial_params, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-8})
        
        # If that doesn't work, try L-BFGS-B
        if not result.success:
            result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-8})
    except Exception as e:
        warnings.warn(f"Aggressive optimization failed with error: {e}")
        pass
    
    # If optimization was successful, return optimized result
    if result is not None and result.success:
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
        return circles
    
    # Fallback: return initial configuration
    return initial_circles.copy()


def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using constrained optimization with better approach"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define bounds for each parameter (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0.001, 0.999))  # Keep small margin to prevent boundary issues
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds
        bounds.append((0.001, 0.499))
    
    # Optimization function
    def objective(params):
        # Convert params back to circles array
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [params[3*i], params[3*i+1], params[3*i+2]]
        
        # Return negative sum of radii (since we want to maximize)
        return -np.sum(circles[:, 2])
    
    # Constraint functions
    def create_constraints():
        constraints = []
        
        # Non-overlap constraints
        def non_overlap_constraint(params):
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [params[3*i], params[3*i+1], params[3*i+2]]
            
            violations = []
            for i, j in combinations(range(n), 2):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                radii_sum = circles[i, 2] + circles[j, 2]
                violation = distance - radii_sum
                violations.append(violation)
            
            return np.array(violations)
        
        # Containment constraints
        def containment_constraint(params):
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [params[3*i], params[3*i+1], params[3*i+2]]
            
            violations = []
            for i in range(n):
                x, y, r = circles[i]
                violations.append(x - r)       # x >= r
                violations.append(y - r)       # y >= r
                violations.append(1 - x - r)   # x <= 1-r
                violations.append(1 - y - r)   # y <= 1-r
                
            return np.array(violations)
        
        return [
            {'type': 'ineq', 'fun': non_overlap_constraint},
            {'type': 'ineq', 'fun': containment_constraint}
        ]
    
    cons = create_constraints()
    
    # Perform optimization with multiple strategies
    result = None
    try:
        # Try different optimization methods with different settings
        methods = ['SLSQP', 'L-BFGS-B']
        
        for method in methods:
            try:
                if method == 'SLSQP':
                    result = minimize(objective, initial_params, method=method, bounds=bounds, constraints=cons, 
                                     options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6})
                else:
                    result = minimize(objective, initial_params, method=method, bounds=bounds,
                                     options={'maxiter': 500, 'ftol': 1e-6})
                
                if result.success:
                    break
            except Exception as e:
                continue
                
    except Exception as e:
        warnings.warn(f"Optimization failed with error: {e}")
        pass
    
    # If optimization was successful, return optimized result
    if result is not None and result.success:
        # Convert back to circles array
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
        return circles
    
    # Fallback: return initial configuration if optimization fails
    return initial_circles


# EVOLVE-BLOCK-END
