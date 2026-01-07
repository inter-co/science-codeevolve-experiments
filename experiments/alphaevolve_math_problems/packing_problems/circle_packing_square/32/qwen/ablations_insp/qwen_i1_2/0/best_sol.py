# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining spatial partitioning and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a structured grid pattern as starting point
    def initialize_grid():
        # Create a grid layout for initial placement
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        
        # Adjust grid to fit exactly 32 circles
        circles = []
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Initial radius estimate based on available space
                r = min(x, 1-x, y, 1-y) * 0.4
                circles.append([x, y, r])
                count += 1
            if count >= n:
                break
        
        # Ensure we have exactly 32 circles
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.05])
            
        return np.array(circles[:n])
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        total_radius = sum(circles_flat[2::3])  # Extract all radii
        return -total_radius
    
    # Constraint functions for optimization
    def radius_constraint(circles_flat):
        """Ensure all circles are within bounds"""
        result = []
        for i in range(n):
            x, y, r = circles_flat[3*i:3*i+3]
            result.append(min(r, x-r, 1-x-r, y-r, 1-y-r))  # Minimum of all boundary constraints
        return np.array(result)
    
    def overlap_constraint(circles_flat):
        """Ensure no overlapping circles"""
        result = []
        circles_array = circles_flat.reshape(-1, 3)
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles_array[i]
                xj, yj, rj = circles_array[j]
                distance_sq = (xi-xj)**2 + (yi-yj)**2
                distance = math.sqrt(distance_sq)
                result.append(distance - (ri + rj))  # Should be >= 0 for non-overlap
        return np.array(result)
    
    # Initialize
    circles = initialize_grid()
    
    # Flatten for optimization
    initial_flat = circles.flatten()
    
    # Set up constraints
    cons = []
    
    # Radius constraints (each circle must stay within bounds)
    cons.append({'type': 'ineq', 'fun': lambda x: radius_constraint(x)})
    
    # Overlap constraints (circles must not overlap)
    cons.append({'type': 'ineq', 'fun': lambda x: overlap_constraint(x)})
    
    # Optimization bounds (x,y in [0,1], r > 0)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (1e-6, 0.5)])  # x, y, r bounds
    
    # Perform optimization
    try:
        result = minimize(objective, initial_flat, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        if result.success:
            circles_opt = result.x.reshape(-1, 3)
        else:
            # If optimization fails, return the initial grid
            circles_opt = circles
    except Exception as e:
        # If optimization fails due to any error, return initial configuration
        circles_opt = circles
    
    return circles_opt


# EVOLVE-BLOCK-END
