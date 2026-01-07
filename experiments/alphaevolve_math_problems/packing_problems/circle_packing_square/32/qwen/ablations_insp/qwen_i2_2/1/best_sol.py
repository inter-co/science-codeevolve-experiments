# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import time
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with gradient-based optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 32
    
    # Improved initialization using a more systematic approach
    circles = np.zeros((n, 3))
    
    # Create a hexagonal packing pattern for better initial configuration
    # This creates a more natural starting point for optimization
    rows = 6
    cols = 6
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset odd rows for hexagonal packing
            x_offset = spacing_x * 0.5 if i % 2 == 1 else 0
            x = (j + 0.5) * spacing_x + x_offset
            y = (i + 0.5) * spacing_y
            
            # Initial radius - reduce slightly to allow optimization
            r = min(spacing_x, spacing_y) * 0.4
            
            # Ensure it fits in the square
            if x + r <= 1 and y + r <= 1 and x - r >= 0 and y - r >= 0:
                circles[idx] = [x, y, r]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with strategic random placements
    for i in range(idx, n):
        attempts = 0
        while attempts < 100:
            # Place in a way that tries to maximize potential for optimization
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            
            # Calculate maximum possible radius at this location
            max_r = min(x, 1-x, y, 1-y)
            
            # Use a more aggressive initial radius to encourage better packing
            r = max(0.01, min(max_r * 0.7, 0.15))
            
            if r > 0.01:
                circles[i] = [x, y, r]
                break
            attempts += 1
    
    # Optimization with improved constraint handling
    def objective(x):
        # Maximize sum of radii (minimize negative sum)
        return -np.sum(x[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    def create_constraints():
        """Create constraints more efficiently"""
        cons = []
        
        # Boundary constraints
        for i in range(n):
            def boundary_constraint(i):
                def func(x):
                    x_pos, y_pos, r = x[3*i], x[3*i+1], x[3*i+2]
                    return min(x_pos - r, 1 - x_pos - r, y_pos - r, 1 - y_pos - r)
                return func
            
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Non-overlap constraints - use more efficient approach
        # Add non-overlap constraints for all pairs
        for i, j in combinations(range(n), 2):
            def overlap_constraint(i, j):
                def func(x):
                    x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                    x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    # Return positive when constraint satisfied (distance >= r_i + r_j)
                    return dist_sq - (r_i + r_j)**2
                return func
                
            cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
            
        return cons
    
    # Create constraints once
    cons = create_constraints()
    
    # Flatten initial guess
    x0 = np.array([circles[i][j] for i in range(n) for j in range(3)])
    
    # Set bounds: x, y in [0,1], r in [0, 0.5] (reasonable upper bound)
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Optimization parameters with more tolerance for convergence
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6, 'disp': False}
    
    try:
        # Try multiple optimization methods to improve chances of success
        # First try L-BFGS-B which often works well for this type of problem
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options=options)
        
        # If that fails or doesn't converge well, try SLSQP
        if not result.success or result.fun > -100:  # Very rough check for convergence
            result = minimize(objective, x0, method='SLSQP', constraints=cons, 
                             bounds=bounds, options=options)
        
        # Extract final solution
        final_circles = np.zeros((n, 3))
        for i in range(n):
            final_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            
        return final_circles
        
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        print(f"Optimization failed: {e}")
        return circles


# EVOLVE-BLOCK-END
