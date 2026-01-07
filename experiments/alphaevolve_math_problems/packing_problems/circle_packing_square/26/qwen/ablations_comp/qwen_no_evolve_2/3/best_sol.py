# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial placement via grid-based heuristic followed by optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a grid-based heuristic
    # Place circles in a roughly hexagonal pattern to get a good starting configuration
    circles = np.zeros((n, 3))
    
    # Create a rough grid layout
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Distribute points more evenly in a grid pattern
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            
            # Set initial radius to be small but feasible
            circles[idx] = [x, y, min(x, 1-x, y, 1-y) * 0.4]
            idx += 1
        if idx >= n:
            break
    
    # Ensure we have exactly n circles
    while idx < n:
        circles[idx] = [0.5, 0.5, 0.1]
        idx += 1
    
    # Define constraint functions
    def constraint_radius(i):
        def constraint_func(params):
            # params: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            return min(r, x - r, 1 - x - r, y - r, 1 - y - r)
        return constraint_func
    
    def constraint_overlap(i, j):
        def constraint_func(params):
            x1 = params[3*i]
            y1 = params[3*i+1]
            r1 = params[3*i+2]
            x2 = params[3*j]
            y2 = params[3*j+1]
            r2 = params[3*j+2]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            return dist_sq - (r1 + r2)**2
        return constraint_func
    
    # Flatten initial guess
    initial_guess = circles.flatten()
    
    # Create bounds for optimization
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Create constraints
    constraints = []
    
    # Add containment constraints for each circle
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': constraint_radius(i)})
    
    # Add overlap constraints for each pair of circles
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_overlap(i, j)})
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i+2]  # radius is at index 3*i+2
        return -total_radius
    
    # Run optimization
    try:
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_params = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [optimized_params[3*i], optimized_params[3*i+1], optimized_params[3*i+2]]
        else:
            # If optimization fails, return the initial configuration
            pass
    except Exception as e:
        # If optimization fails due to any reason, return initial configuration
        pass
    
    return circles


# EVOLVE-BLOCK-END
