# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining physics simulation and mathematical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize positions and radii
    # Start with a simple hexagonal packing pattern as initial guess
    circles = np.zeros((n, 3))
    
    # Create initial configuration using a grid-like approach with some randomness
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Distribute points in a grid pattern with some jitter
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Position with jitter
            x = 0.1 + 0.8 * j / (cols - 1) if cols > 1 else 0.5
            y = 0.1 + 0.8 * i / (rows - 1) if rows > 1 else 0.5
            # Add small random jitter to prevent perfect symmetry
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
            
            # Clamp to valid range
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            circles[idx] = [x, y, 0.05]  # Initial radius
            idx += 1
        if idx >= n:
            break
    
    # Adjust initial radii based on proximity to other circles
    # Compute pairwise distances
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                min_dist = min(min_dist, dist)
        
        # Set radius such that circles don't overlap, with some margin
        if min_dist > 0:
            circles[i, 2] = min(0.45, min_dist / 2.0 - 0.01)
        else:
            circles[i, 2] = 0.05
    
    # Ensure all circles are within bounds
    for i in range(n):
        circles[i, 0] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 0]))
        circles[i, 1] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 1]))
    
    # Define constraint functions
    def constraint_radius(i):
        """Ensure circle i stays within bounds"""
        def func(x):
            return np.array([
                x[0] - x[2],  # x - r >= 0
                1 - x[0] - x[2],  # 1 - x - r >= 0
                x[1] - x[2],  # y - r >= 0
                1 - x[1] - x[2]   # 1 - y - r >= 0
            ])
        return func
    
    def constraint_overlap(i, j):
        """Ensure circles i and j don't overlap"""
        def func(x_i_x_j):
            x_i, y_i, r_i = x_i_x_j[:3]
            x_j, y_j, r_j = x_i_x_j[3:]
            # Distance between centers minus sum of radii should be >= 0
            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            return np.array([dist_sq - (r_i + r_j)**2])
        return func
    
    # Objective function: negative of sum of radii (we want to maximize)
    def objective(params):
        total_radius = 0
        for i in range(n):
            total_radius += params[i*3 + 2]  # Extract radius
        return -total_radius
    
    # Constraints for bounds
    bounds = []
    for i in range(n):
        # Each circle has (x, y, r) parameters
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Build constraint list
    constraints = []
    
    # Add boundary constraints for each circle
    for i in range(n):
        # Circle i must stay within unit square with its radius
        def bound_constraint(i):
            def func(params):
                x, y, r = params[i*3], params[i*3+1], params[i*3+2]
                return np.array([
                    x - r,  # x - r >= 0
                    1 - x - r,  # 1 - x - r >= 0
                    y - r,  # y - r >= 0
                    1 - y - r   # 1 - y - r >= 0
                ])
            return func
        
        constraints.append({'type': 'ineq', 'fun': bound_constraint(i)})
    
    # Add overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(i, j):
                def func(params):
                    x_i, y_i, r_i = params[i*3], params[i*3+1], params[i*3+2]
                    x_j, y_j, r_j = params[j*3], params[j*3+1], params[j*3+2]
                    # Distance squared between centers minus (r_i + r_j)^2 should be >= 0
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    return np.array([dist_sq - (r_i + r_j)**2])
                return func
            
            constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
    
    # Flatten initial guess
    x0 = []
    for i in range(n):
        x0.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
    
    # Optimization with bounds and constraints
    try:
        # Use SLSQP method which handles both bounds and constraints well
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )
        
        # Extract results
        final_circles = np.zeros((n, 3))
        for i in range(n):
            final_circles[i] = [result.x[i*3], result.x[i*3+1], result.x[i*3+2]]
            
        # Final check and adjustment
        for i in range(n):
            # Ensure all circles are within bounds
            final_circles[i, 0] = max(final_circles[i, 2], min(1 - final_circles[i, 2], final_circles[i, 0]))
            final_circles[i, 1] = max(final_circles[i, 2], min(1 - final_circles[i, 2], final_circles[i, 1]))
            
        return final_circles
        
    except Exception as e:
        # Fallback to the initial configuration if optimization fails
        warnings.warn(f"Optimization failed: {str(e)}. Returning initial configuration.")
        return circles


# EVOLVE-BLOCK-END
