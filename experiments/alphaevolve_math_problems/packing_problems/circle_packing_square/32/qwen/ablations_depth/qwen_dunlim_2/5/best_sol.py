# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining physics simulation and mathematical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Use a better initial configuration - attempt to create a more efficient starting point
    # This uses a more structured approach with better spatial distribution
    circles = np.zeros((n, 3))
    
    # Create a more sophisticated initial layout
    # Arrange in a pattern that's known to work well for circle packing
    rows = 6
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Place in grid pattern with slight randomness
            x = (j + 1) * spacing_x + np.random.normal(0, 0.01)
            y = (i + 1) * spacing_y + np.random.normal(0, 0.01)
            
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            circles[idx] = [x, y, 0.05]
            idx += 1
        if idx >= n:
            break
    
    # Refine initial radii based on local density
    # For each circle, compute minimum distance to neighbors and set appropriate radius
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                min_dist = min(min_dist, dist)
        
        # Set radius to allow maximum without overlapping
        if min_dist > 0.01:
            circles[i, 2] = min(0.45, min_dist / 2.0 - 0.005)
        else:
            circles[i, 2] = 0.05
    
    # Ensure all circles are within bounds
    for i in range(n):
        circles[i, 0] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 0]))
        circles[i, 1] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 1]))
    
    # More robust optimization approach with multiple tries
    best_result = None
    best_sum = 0
    
    # Try multiple optimization attempts with different settings
    for attempt in range(3):
        # Flatten initial guess
        x0 = []
        for i in range(n):
            x0.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
        
        # Define objective function
        def objective(params):
            total_radius = 0
            for i in range(n):
                total_radius += params[i*3 + 2]  # Extract radius
            return -total_radius  # Negative because we minimize
        
        # Define bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Define constraints
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            def bound_constraint(i):
                def func(params):
                    x, y, r = params[i*3], params[i*3+1], params[i*3+2]
                    return np.array([
                        x - r,      # x - r >= 0
                        1 - x - r,  # 1 - x - r >= 0
                        y - r,      # y - r >= 0
                        1 - y - r   # 1 - y - r >= 0
                    ])
                return func
            
            constraints.append({'type': 'ineq', 'fun': bound_constraint(i)})
        
        # Overlap constraints
        for i, j in combinations(range(n), 2):
            def overlap_constraint(i, j):
                def func(params):
                    x_i, y_i, r_i = params[i*3], params[i*3+1], params[i*3+2]
                    x_j, y_j, r_j = params[j*3], params[j*3+1], params[j*3+2]
                    # Distance squared between centers minus (r_i + r_j)^2 should be >= 0
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    return np.array([dist_sq - (r_i + r_j)**2])
                return func
            
            constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        try:
            # Try different optimization methods
            methods = ['SLSQP', 'trust-constr']
            for method in methods:
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6},
                    tol=1e-6
                )
                
                if result.success:
                    # Calculate sum of radii
                    total_radius = 0
                    for i in range(n):
                        total_radius += result.x[i*3 + 2]
                    
                    if total_radius > best_sum:
                        best_sum = total_radius
                        best_result = result
                        
        except Exception as e:
            continue
    
    # If we found a good result, use it; otherwise use the initial configuration
    if best_result is not None and best_result.success:
        final_circles = np.zeros((n, 3))
        for i in range(n):
            final_circles[i] = [best_result.x[i*3], best_result.x[i*3+1], best_result.x[i*3+2]]
            
        # Final validation and correction
        for i in range(n):
            final_circles[i, 0] = max(final_circles[i, 2], min(1 - final_circles[i, 2], final_circles[i, 0]))
            final_circles[i, 1] = max(final_circles[i, 2], min(1 - final_circles[i, 2], final_circles[i, 1]))
            
        return final_circles
    else:
        # Return initial configuration if optimization failed
        return circles


# EVOLVE-BLOCK-END
