# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses improved hexagonal initialization and refined optimization approach.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Improved hexagonal initialization as per inspiration 2
    def initialize_hexagonal():
        circles = []
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # More precise hexagonal packing
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = (i % 2) * (spacing_x / 2)
                x = (j + 0.5) * spacing_x + x_offset
                y = (i + 0.5) * spacing_y
                
                # Ensure within bounds
                if x < 0 or x > 1 or y < 0 or y > 1:
                    continue
                    
                # Initial radius - based on proximity to edges
                radius = min(x, 1-x, y, 1-y) * 0.4
                if radius > 0.001:
                    circles.append([x, y, radius])
        
        # Fill remaining positions with strategic placement
        while len(circles) < n:
            # Place randomly but with better distribution
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            radius = min(x, 1-x, y, 1-y) * 0.3
            if radius > 0.001:
                circles.append([x, y, radius])
                
        return np.array(circles[:n])
    
    # Create initial configuration
    circles = initialize_hexagonal()
    
    # Flatten initial circles for optimization
    x0 = circles.flatten()
    
    # Define bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0.001, 0.999))  # Avoid exact boundaries
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds
        bounds.append((0.001, 0.499))  # Radius can't exceed 0.5
    
    # Helper function to create constraint closures properly
    def make_boundary_constraint(i):
        def constraint(circles_flat):
            x, y, r = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
            # Return positive when satisfied (g(x) <= 0 for scipy)
            return np.array([
                x - r,      # x >= r
                y - r,      # y >= r
                1 - x - r,  # 1-x >= r
                1 - y - r   # 1-y >= r
            ])
        return constraint
    
    # Helper function to create non-overlap constraint closures properly
    def make_nonoverlap_constraint(i, j):
        def constraint(circles_flat):
            x_i, y_i, r_i = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
            x_j, y_j, r_j = circles_flat[3*j], circles_flat[3*j+1], circles_flat[3*j+2]
            # Distance between centers squared
            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            # Non-overlap constraint: distance^2 >= (r_i + r_j)^2
            return dist_sq - (r_i + r_j)**2
        return constraint
    
    # Create constraints properly to avoid late binding issues
    constraints = []
    
    # Boundary constraints for each circle
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i)})
    
    # Non-overlap constraints for all pairs
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_nonoverlap_constraint(i, j)})
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    try:
        # Run optimization with improved settings
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6, 'disp': False},
            tol=1e-6
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure all circles are valid
            for i in range(n):
                x, y, r = optimized_circles[i]
                # Clamp values to valid ranges
                optimized_circles[i] = [
                    np.clip(x, r, 1-r),
                    np.clip(y, r, 1-r),
                    np.clip(r, 0.001, 0.499)
                ]
            return optimized_circles
        else:
            # If optimization fails, return initial placement
            return circles
            
    except Exception as e:
        # Fallback to initial placement if optimization fails
        return circles


# EVOLVE-BLOCK-END
