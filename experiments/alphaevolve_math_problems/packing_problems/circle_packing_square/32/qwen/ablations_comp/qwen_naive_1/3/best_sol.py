# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using a hexagonal lattice pattern for good initial distribution
    def initialize_hexagonal_layout():
        # Create a hexagonal grid pattern that fits well in the unit square
        rows = 6
        cols = 6
        # Adjust spacing to fit 32 circles reasonably
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        circles = []
        count = 0
        
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = (i % 2) * spacing_x / 2
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                
                # Ensure we're within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius based on spacing
                    radius = min(spacing_x, spacing_y) / 3
                    circles.append([x, y, radius])
                    count += 1
            if count >= n:
                break
        
        # Fill remaining positions with random placements
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Small random radius
            radius = np.random.uniform(0.01, 0.05)
            circles.append([x, y, radius])
            
        return np.array(circles)
    
    # Constraint functions
    def containment_constraints(circles_flat):
        """Ensure all circles are fully contained in the unit square"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Each circle must be contained in [0,1]x[0,1]
        for i in range(len(circles)):
            x, y, r = circles[i]
            # r <= x <= 1-r and r <= y <= 1-r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1 - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})  # 1 - y - r >= 0
            
        return constraints
    
    def overlap_constraints(circles_flat):
        """Ensure no two circles overlap"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # For each pair of circles, enforce minimum distance
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                def distance_constraint(c, i=i, j=j):
                    x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                    x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                    distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_distance_sq = (r1 + r2)**2
                    return distance_sq - min_distance_sq
                
                constraints.append({'type': 'ineq', 'fun': distance_constraint})
                
        return constraints
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Initial guess
    initial_circles = initialize_hexagonal_layout()
    initial_guess = initial_circles.flatten()
    
    # Set up constraints
    cons = []
    
    # Add containment constraints
    for i in range(n):
        x, y, r = initial_circles[i]
        # x >= r, 1-x >= r, y >= r, 1-y >= r
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})  # 1 - y - r >= 0
    
    # Add overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def dist_constr(c, i=i, j=j):
                x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                return dist_sq - min_dist_sq
            cons.append({'type': 'ineq', 'fun': dist_constr})
    
    # Bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y in [0,1], r in [0,0.5] (safe upper bound)
    
    try:
        # Optimize using SLSQP method which handles constraints well
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # If optimization fails, return the initial configuration
            return initial_circles
            
    except Exception as e:
        # Return initial configuration if anything goes wrong
        return initial_circles


# EVOLVE-BLOCK-END
