# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach: initial placement followed by constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initial heuristic placement - arrange in a roughly hexagonal pattern
    def initial_placement():
        # Try to place circles in a hexagonal lattice pattern
        circles = []
        
        # Determine grid dimensions
        rows = int(math.sqrt(n)) + 1
        cols = int(n / rows) + 1
        
        # Create initial positions in a grid
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles.append([x, y, 0.0])  # Initialize with zero radius
        
        # Set initial radii to small values
        for i in range(len(circles)):
            circles[i][2] = min(circles[i][0], 1 - circles[i][0], 
                               circles[i][1], 1 - circles[i][1]) * 0.4
            
        return np.array(circles)
    
    # Constraint functions for optimization
    def constraint_radius(circle_data):
        """Ensure each circle fits within the unit square"""
        x, y, r = circle_data
        return min(r, x - r, 1 - x - r, y - r, 1 - y - r)
    
    def constraint_overlap(circle_i, circle_j):
        """Ensure two circles don't overlap"""
        x1, y1, r1 = circle_i
        x2, y2, r2 = circle_j
        distance_sq = (x1 - x2)**2 + (y1 - y2)**2
        return distance_sq - (r1 + r2)**2
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        total_radius = np.sum(circles_flat[2::3])  # Extract all radii
        return -total_radius
    
    # Constraints for optimization
    def radius_constraint(circles_flat):
        # Each circle must fit in the square
        result = []
        for i in range(n):
            x = circles_flat[i*3]
            y = circles_flat[i*3 + 1]
            r = circles_flat[i*3 + 2]
            # Minimum of four boundary distances
            min_bound = min(x, 1-x, y, 1-y)
            result.append(min_bound - r)
        return np.array(result)
    
    def overlap_constraint(circles_flat):
        # Non-overlapping constraints
        result = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_flat[i*3], circles_flat[i*3+1], circles_flat[i*3+2]
                x2, y2, r2 = circles_flat[j*3], circles_flat[j*3+1], circles_flat[j*3+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                result.append(dist_sq - (r1 + r2)**2)
        return np.array(result)
    
    # Initial configuration
    circles = initial_placement()
    
    # Flatten for optimization
    circles_flat = circles.flatten()
    
    # Set up bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: radius_constraint(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
    ]
    
    # Optimize using SLSQP
    try:
        result = minimize(
            objective,
            circles_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure final positions respect boundaries
            for i in range(n):
                x, y, r = optimized_circles[i]
                # Adjust if necessary to stay within bounds
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                optimized_circles[i] = [x, y, r]
            return optimized_circles
        else:
            # Return initial placement if optimization fails
            return circles
    except Exception:
        # Return initial placement if optimization fails
        return circles


# EVOLVE-BLOCK-END
