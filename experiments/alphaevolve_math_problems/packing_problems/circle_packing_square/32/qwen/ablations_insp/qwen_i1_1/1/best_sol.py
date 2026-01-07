# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial placement with geometric constraints followed by optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Initialize circles with a good starting configuration using hexagonal grid
    # Inspired by the better performing inspiration program
    circles = np.zeros((n, 3))
    
    # Create a hexagonal grid pattern as initial configuration
    # This provides a good starting point for optimization
    rows = 6
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Initial radius - small enough to fit in the grid cell
            r = min(spacing_x, spacing_y) * 0.3
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with random placements
    for i in range(idx, n):
        circles[i] = [
            random.uniform(0.05, 0.95),
            random.uniform(0.05, 0.95),
            random.uniform(0.01, 0.1)
        ]
    
    # Convert to flat parameter vector for optimization
    # Parameters: [x0, y0, r0, x1, y1, r1, ...]
    def get_params(circles):
        params = []
        for i in range(n):
            params.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
        return np.array(params)
    
    def set_circles_from_params(params):
        for i in range(n):
            circles[i, 0] = params[3*i]
            circles[i, 1] = params[3*i + 1]
            circles[i, 2] = params[3*i + 2]
        return circles
    
    # Constraint functions with improved numerical stability
    def constraint_containment(params):
        """Ensure all circles fit within the unit square"""
        set_circles_from_params(params)
        constraints = []
        for i in range(n):
            x, y, r = circles[i, 0], circles[i, 1], circles[i, 2]
            # Add small safety margin to avoid numerical issues
            constraints.extend([
                x - r + 1e-8,      # x - r >= 0
                y - r + 1e-8,      # y - r >= 0
                1 - x - r + 1e-8,  # 1 - x - r >= 0
                1 - y - r + 1e-8   # 1 - y - r >= 0
            ])
        return np.array(constraints)
    
    def constraint_nonoverlap(params):
        """Ensure no overlap between circles"""
        set_circles_from_params(params)
        constraints = []
        # Use a more efficient approach to check constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i, 0], circles[i, 1], circles[i, 2]
                x2, y2, r2 = circles[j, 0], circles[j, 1], circles[j, 2]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                # We want distance >= r1 + r2
                # So we enforce distance - r1 - r2 >= 0
                # Using squared distance to avoid sqrt computation
                min_distance = r1 + r2
                constraints.append(np.sqrt(distance_sq) - min_distance)
        return np.array(constraints)
    
    # Objective function (negative because we minimize)
    def objective(params):
        set_circles_from_params(params)
        return -np.sum(circles[:, 2])
    
    # Initial parameters
    initial_params = get_params(circles)
    
    # Define bounds for each parameter (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # r max ~ 0.5
    
    # Create constraints dictionary
    cons = [
        {'type': 'ineq', 'fun': lambda p: constraint_containment(p)},
        {'type': 'ineq', 'fun': lambda p: constraint_nonoverlap(p)}
    ]
    
    # Try multiple optimization approaches for better results
    best_result = None
    best_sum = -np.inf
    
    # Try with different optimizers
    try:
        # SLSQP optimization
        result_slsqp = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result_slsqp.success:
            final_circles = set_circles_from_params(result_slsqp.x)
            current_sum = np.sum(final_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = final_circles
        
        # Try L-BFGS-B as backup
        if best_result is None:
            result_lbfgsb = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if result_lbfgsb.success:
                final_circles = set_circles_from_params(result_lbfgsb.x)
                current_sum = np.sum(final_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = final_circles
                    
    except Exception as e:
        # If optimization fails, use the hexagonal grid configuration
        pass
    
    # Return best result or fallback to initial configuration
    if best_result is not None:
        return best_result
    else:
        return circles


# EVOLVE-BLOCK-END
