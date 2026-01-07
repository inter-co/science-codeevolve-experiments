# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial geometric placement followed by constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles with a structured approach
    # Start with a hexagonal-like arrangement in a grid pattern
    circles = np.zeros((n, 3))
    
    # Create a hexagonal grid pattern as initial guess
    # This helps avoid poor local minima and provides good starting configuration
    rows = 6
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + 1) * spacing_x + offset * spacing_x * 0.5
            y = (i + 1) * spacing_y
            # Initial radius - small but feasible
            r = min(spacing_x, spacing_y) * 0.3
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Truncate if we have more than needed
    circles = circles[:n]
    
    # Optimization setup
    def objective(params):
        # params contains [x1, y1, r1, x2, y2, r2, ...]
        radii = params[2::3]  # Extract all radii
        return -np.sum(radii)  # Negative because we want to maximize
    
    def constraint_containment(params):
        # Ensure all circles are within the unit square
        positions = params[0::3].reshape(-1, 1)  # x coordinates
        radii = params[2::3]
        # Lower bounds: radius <= x <= 1-radius
        lower_bound = radii
        upper_bound = 1 - radii
        return np.concatenate([positions - lower_bound, upper_bound - positions])
    
    def constraint_nonoverlap(params):
        # Ensure no two circles overlap
        n_circles = len(params) // 3
        positions = params[0::3].reshape(-1, 1)  # x coordinates
        positions_y = params[1::3].reshape(-1, 1)  # y coordinates
        radii = params[2::3]
        
        # Compute pairwise distances
        pos_array = np.column_stack([positions.flatten(), positions_y.flatten()])
        distances = cdist(pos_array, pos_array)
        
        # Create constraint array: distance >= radii[i] + radii[j] for i != j
        constraints = []
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                # Distance constraint: sqrt((x_i-x_j)^2 + (y_i-y_j)^2) >= r_i + r_j
                # Rearranged: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
                dist_sq = distances[i, j]**2
                radius_sum = radii[i] + radii[j]
                constraints.append(dist_sq - radius_sum**2)
        
        return np.array(constraints)
    
    def constraint_bounds(params):
        # Bounds for optimization variables
        # x, y in [0,1], r in [0,0.5] (reasonable upper bound)
        bounds = []
        for i in range(0, len(params), 3):
            # x bounds
            bounds.extend([0, 1])  # x_min, x_max
            # y bounds  
            bounds.extend([0, 1])  # y_min, y_max
            # r bounds
            bounds.extend([0, 0.5])  # r_min, r_max
        return bounds
    
    # Flatten initial configuration
    initial_params = np.zeros(n * 3)
    for i in range(n):
        initial_params[3*i] = circles[i][0]  # x
        initial_params[3*i+1] = circles[i][1]  # y
        initial_params[3*i+2] = circles[i][2]  # r
    
    # Set up constraints
    # We'll use a simplified approach with bounds and simple feasibility checks
    # For actual implementation, we'd create proper constraint objects
    
    # Alternative: Use scipy.optimize.minimize with bounds and constraints
    # Define bounds for parameters [x1, y1, r1, x2, y2, r2, ...]
    bounds = [(0, 1), (0, 1), (0.001, 0.5)] * n  # x, y, r bounds
    
    # Define constraints for optimization
    cons = []
    
    # Add containment constraints (radius <= x, y <= 1-radius)
    def containment_constraint(params):
        # For each circle, check containment
        result = []
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            result.append(x - r)      # x - r >= 0
            result.append(1 - x - r)  # 1 - x - r >= 0
            result.append(y - r)      # y - r >= 0
            result.append(1 - y - r)  # 1 - y - r >= 0
        return np.array(result)
    
    # Add non-overlap constraints
    def nonoverlap_constraint(params):
        # For each pair of circles, ensure distance >= sum of radii
        result = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                radius_sum = r1 + r2
                # We want dist_sq >= radius_sum^2 (for non-overlap)
                # So we return dist_sq - radius_sum^2 (should be >= 0)
                result.append(dist_sq - radius_sum**2)
        return np.array(result)
    
    # Add constraints to the list
    cons.append({'type': 'ineq', 'fun': containment_constraint})
    cons.append({'type': 'ineq', 'fun': nonoverlap_constraint})
    
    # Optimization using SLSQP method
    try:
        # First attempt with basic optimization
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6},
            callback=lambda x: None  # No callback for now
        )
        
        # If optimization succeeded, extract results
        if result.success:
            final_params = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
        else:
            # Fallback: use initial configuration if optimization fails
            pass
            
    except Exception as e:
        # If optimization fails, fall back to initial configuration
        warnings.warn(f"Optimization failed: {str(e)}")
        pass
    
    # Final validation and cleanup
    # Ensure all circles are properly contained
    for i in range(n):
        x, y, r = circles[i]
        # Adjust if necessary to keep within bounds
        circles[i] = [
            max(r, min(1-r, x)), 
            max(r, min(1-r, y)), 
            max(0.001, min(0.5, r))
        ]
    
    return circles


# EVOLVE-BLOCK-END
