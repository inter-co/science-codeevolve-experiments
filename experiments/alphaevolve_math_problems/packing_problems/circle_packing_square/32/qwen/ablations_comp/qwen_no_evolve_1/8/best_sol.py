# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial structured placement + constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initial placement strategy: arrange in a hexagonal lattice pattern
    # First, determine how to distribute 32 points in a grid-like structure
    rows = int(math.sqrt(n))
    cols = math.ceil(n / rows)
    
    # Create initial positions in a grid
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Space points in a way that allows for some flexibility
            x = (j + 0.5) / cols * 0.8 + 0.1  # Keep away from edges
            y = (i + 0.5) / rows * 0.8 + 0.1
            positions.append([x, y])
    
    # Ensure we have exactly n positions
    positions = positions[:n]
    
    # Initialize radii to small values
    radii = np.full(n, 0.02)
    
    # Combine positions and radii into a single parameter vector
    initial_params = np.array(positions).flatten()
    initial_params = np.concatenate([initial_params, radii])
    
    # Define constraints
    def constraint_radius(i):
        def constraint(x):
            pos_idx = 2*i
            rad_idx = 2*n + i
            x_c, y_c = x[pos_idx], x[pos_idx+1]
            r = x[rad_idx]
            # Circle must fit in the unit square
            return min(r, 1-r-x_c, 1-r-y_c, x_c-r, y_c-r)
        return constraint
    
    def constraint_overlap(i, j):
        def constraint(x):
            pos_idx_i = 2*i
            pos_idx_j = 2*j
            rad_idx_i = 2*n + i
            rad_idx_j = 2*n + j
            x_i, y_i = x[pos_idx_i], x[pos_idx_i+1]
            x_j, y_j = x[pos_idx_j], x[pos_idx_j+1]
            r_i = x[rad_idx_i]
            r_j = x[rad_idx_j]
            # Distance between centers minus sum of radii should be non-negative
            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            return dist_sq - (r_i + r_j)**2
        return constraint
    
    # Set up constraints for optimization
    constraints = []
    
    # Add radius constraints (each circle must fit in the unit square)
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': constraint_radius(i)})
    
    # Add overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_overlap(i, j)})
    
    # Objective function: negative sum of radii (we minimize to maximize sum)
    def objective(x):
        total_radius = 0
        for i in range(n):
            rad_idx = 2*n + i
            total_radius += x[rad_idx]
        return -total_radius
    
    # Bounds for positions (keep away from edges to allow for radii)
    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.99))  # x coordinates
        bounds.append((0.01, 0.99))  # y coordinates
    for i in range(n):
        bounds.append((0.001, 0.49))  # radii
    
    # Optimization with different approaches
    try:
        # Try with SLSQP method which handles constraints well
        result = minimize(objective, initial_params, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            final_params = result.x
        else:
            # If optimization fails, fallback to the structured approach
            final_params = initial_params
    except Exception:
        # Fallback to simple heuristic approach
        final_params = initial_params
    
    # Extract final solution
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i][0] = final_params[2*i]
        circles[i][1] = final_params[2*i+1]
        circles[i][2] = final_params[2*n+i]
    
    # Final validation to ensure constraints are met
    circles = validate_and_adjust(circles)
    
    return circles

def validate_and_adjust(circles):
    """Ensure all constraints are satisfied after optimization"""
    n = len(circles)
    
    # Simple validation and adjustment
    for i in range(n):
        x, y, r = circles[i]
        # Ensure circle fits in square
        r = min(r, x, y, 1-x, 1-y)
        circles[i] = [x, y, r]
    
    # Check overlaps and adjust
    max_iter = 100
    iter_count = 0
    while iter_count < max_iter:
        any_change = False
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Calculate distance between centers
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                if dist_sq < min_dist_sq:
                    # Need to reduce one of the radii
                    overlap = min_dist_sq - dist_sq
                    reduction = overlap / (2 * (r1 + r2))
                    
                    # Reduce both radii proportionally
                    if r1 > 0.001 and r2 > 0.001:
                        r1_new = max(0.001, r1 * (1 - reduction))
                        r2_new = max(0.001, r2 * (1 - reduction))
                        
                        # Adjust positions to maintain center locations
                        circles[i][2] = r1_new
                        circles[j][2] = r2_new
                        any_change = True
                        
        if not any_change:
            break
        iter_count += 1
    
    return circles


# EVOLVE-BLOCK-END
