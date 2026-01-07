# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining grid initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Grid-based initial placement
    # Arrange circles in a grid pattern to get a good starting configuration
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Create initial positions on a grid
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            positions.append([x, y])
    
    # Initialize with small radii
    radii = [0.02] * n
    
    # Create initial circles array
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [positions[i][0], positions[i][1], radii[i]]
    
    # Define constraint functions
    def contain_constraints(circles_flat):
        """Ensure all circles are contained within the unit square"""
        constraints = []
        for i in range(n):
            x, y, r = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
            # Radius constraint
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2]})  # r >= 0
            # Containment constraints
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})  # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})  # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})  # 1-x >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})  # 1-y >= r
        return constraints
    
    def overlap_constraints(circles_flat):
        """Ensure no overlapping circles"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                # Distance constraint: sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
                def dist_constraint(x, i=i, j=j):
                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    min_dist_sq = (ri + rj)**2
                    return dist_sq - min_dist_sq
                
                constraints.append({'type': 'ineq', 'fun': dist_constraint})
        return constraints
    
    # Objective function: negative sum of radii (we want to maximize sum of radii)
    def objective(circles_flat):
        return -sum(circles_flat[2::3])  # Sum of all radii (negative for maximization)
    
    # Flatten the circles array for optimization
    initial_flat = circles.flatten()
    
    # Set up bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y in [0,1], r in [0, 0.5]
    
    # Get constraints
    cons = []
    cons.extend(contain_constraints(initial_flat))
    cons.extend(overlap_constraints(initial_flat))
    
    # Perform optimization
    try:
        result = minimize(objective, initial_flat, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_flat = result.x
            circles = optimized_flat.reshape((n, 3))
        else:
            # If optimization fails, return the initial configuration
            pass
    except Exception as e:
        # If optimization fails due to any reason, return the initial configuration
        pass
    
    return circles


# EVOLVE-BLOCK-END
