# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining spatial partitioning and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a structured grid pattern as starting point
    # This provides a good initial configuration that's feasible
    circles = np.zeros((n, 3))
    
    # Create a grid layout for initial placement
    grid_size = int(math.ceil(math.sqrt(n)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            # Initial radius - small enough to fit in the grid cell
            r = min(spacing_x, spacing_y) * 0.4
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Ensure we have exactly 32 circles
    while idx < n:
        # Fill remaining positions with small circles at random locations
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = 0.02
        circles[idx] = [x, y, r]
        idx += 1
    
    # Optimization function to maximize sum of radii
    def objective(params):
        # Reshape parameters back into circles array
        pos_radii = params.reshape(-1, 3)
        radii = pos_radii[:, 2]
        return -np.sum(radii)  # Negative because we want to maximize
    
    def constraint_func(params):
        # Check all constraints: containment and non-overlap
        pos_radii = params.reshape(-1, 3)
        x = pos_radii[:, 0]
        y = pos_radii[:, 1]
        r = pos_radii[:, 2]
        
        # Containment constraints: each circle must be fully inside the unit square
        containment = []
        for i in range(n):
            containment.append(x[i] - r[i])  # x - r >= 0
            containment.append(y[i] - r[i])  # y - r >= 0
            containment.append(1 - x[i] - r[i])  # 1 - x - r >= 0
            containment.append(1 - y[i] - r[i])  # 1 - y - r >= 0
        
        # Non-overlap constraints: distance between centers >= sum of radii
        non_overlap = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                non_overlap.append(dist - r[i] - r[j])  # dist - r_i - r_j >= 0
        
        return np.array(containment + non_overlap)
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # Bounds for x and y: [r, 1-r] to ensure containment
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.4)])
    
    # Flatten initial guess
    initial_guess = circles.flatten()
    
    # Define constraints dictionary
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Perform optimization
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return the initial configuration if optimization fails
            return circles
    except Exception as e:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
