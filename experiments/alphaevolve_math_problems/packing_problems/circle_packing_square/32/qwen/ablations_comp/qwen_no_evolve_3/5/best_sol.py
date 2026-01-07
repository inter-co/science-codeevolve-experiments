# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles using a hexagonal packing pattern as starting point
    # This provides a good initial configuration that's likely to be feasible
    circles = np.zeros((n, 3))
    
    # Create a hexagonal grid pattern
    rows = 6
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset odd rows for hexagonal packing
            x_offset = 0.0 if i % 2 == 0 else spacing_x / 2
            x = (j + 1) * spacing_x + x_offset
            y = (i + 1) * spacing_y
            
            # Initial radius - small enough to fit in the square
            r = min(x, 1-x, y, 1-y) / 2.0
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Trim to exactly 32 circles if needed
    circles = circles[:n]
    
    # Define constraint functions
    def containement_constraints(circles_flat):
        """Ensure all circles are contained within the unit square"""
        constraints = []
        for i in range(n):
            x, y, r = circles_flat[3*i:3*i+3]
            # Circle must be fully inside the unit square
            constraints.append({'type': 'ineq', 'fun': lambda xyr, i=i: xyr[3*i] - xyr[3*i+2]})  # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda xyr, i=i: xyr[3*i+1] - xyr[3*i+2]})  # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda xyr, i=i: 1 - xyr[3*i] - xyr[3*i+2]})  # 1 - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda xyr, i=i: 1 - xyr[3*i+1] - xyr[3*i+2]})  # 1 - y - r >= 0
        return constraints
    
    def overlap_constraints(circles_flat):
        """Ensure no two circles overlap"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda xyr, i=i, j=j: 
                        np.sqrt((xyr[3*i] - xyr[3*j])**2 + (xyr[3*i+1] - xyr[3*j+1])**2) - xyr[3*i+2] - xyr[3*j+2]
                })
        return constraints
    
    # Objective function to maximize (negative because minimize)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Sum of all radii (negative for maximization)
    
    # Flatten initial circles for optimization
    initial_flat = circles.flatten()
    
    # Set up bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y in [0,1], r in [0, 0.5]
    
    # Create constraint dictionaries
    cons = []
    cons.extend(containement_constraints(initial_flat))
    cons.extend(overlap_constraints(initial_flat))
    
    # Perform optimization
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # If optimization fails, return the initial configuration
            return circles
    except Exception as e:
        # If there's an error in optimization, return the initial configuration
        return circles


# EVOLVE-BLOCK-END
