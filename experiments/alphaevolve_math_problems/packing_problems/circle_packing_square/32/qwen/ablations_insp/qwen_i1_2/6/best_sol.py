# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 32
    
    # Initialize with a structured grid pattern inspired by hexagonal packing
    def initialize_circles():
        circles = np.zeros((n, 3))
        
        # Create a hexagonal-like grid pattern for initial placement
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                # Initial radius - small enough to fit in grid cell
                r = min(spacing_x, spacing_y) * 0.2
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
        
        # Ensure we have exactly n circles
        while idx < n:
            # Fill remaining positions with random placements near edges
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = 0.05
            circles[idx] = [x, y, r]
            idx += 1
            
        return circles
    
    # Objective function to maximize sum of radii
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Negative because we minimize
    
    # Constraint functions for optimization
    def get_bounds_and_constraints():
        # Bounds for variables (x, y, r) for each circle
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y in [0.001, 0.999], r in [0.001, 0.499]
        
        # Constraints list
        constraints = []
        
        # Boundary constraints for each circle
        for i in range(n):
            # x >= r and x + r <= 1
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})  # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})  # x + r <= 1
            # y >= r and y + r <= 1  
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})  # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})  # y + r <= 1
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                # Distance constraint: sqrt((x_i-x_j)^2 + (y_i-y_j)^2) >= r_i + r_j
                def overlap_constraint(x, i=i, j=j):
                    x_i, y_i, r_i = x[i*3], x[i*3+1], x[i*3+2]
                    x_j, y_j, r_j = x[j*3], x[j*3+1], x[j*3+2]
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    min_dist_sq = (r_i + r_j)**2
                    return dist_sq - min_dist_sq
                
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return bounds, constraints
    
    # Initialize
    circles = initialize_circles()
    
    # Flatten for optimization
    x0 = circles.flatten()
    
    # Get bounds and constraints
    bounds, constraints = get_bounds_and_constraints()
    
    # Run optimization with multiple attempts for better results
    best_result = None
    best_sum = 0
    
    # Try multiple optimization runs with different initializations
    for attempt in range(3):
        try:
            # Add some noise to initial solution for diversity
            if attempt > 0:
                noise = np.random.normal(0, 0.01, x0.shape)
                x0_noisy = np.clip(x0 + noise, 0.001, 0.999)
                # Ensure radii don't exceed bounds
                x0_noisy[2::3] = np.clip(x0_noisy[2::3], 0.001, 0.499)
                x0 = x0_noisy
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6})
            
            if result.success:
                current_sum = -objective(result.x)  # Convert back to positive sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception:
            continue
    
    # Return best result or initial configuration
    if best_result is not None and best_result.success:
        optimized_circles = best_result.x.reshape(-1, 3)
        return optimized_circles
    else:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
