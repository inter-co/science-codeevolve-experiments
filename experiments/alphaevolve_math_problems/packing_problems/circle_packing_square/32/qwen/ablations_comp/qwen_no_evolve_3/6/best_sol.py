# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach: initial hexagonal lattice placement followed by optimization.
    """
    n = 32
    
    # Initialize with hexagonal lattice arrangement
    def initialize_hexagonal():
        # Create a hexagonal grid pattern
        circles = []
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Hexagonal arrangement with offset rows
        for i in range(rows):
            for j in range(cols):
                if i % 2 == 0:
                    x = (j + 0.5) * spacing_x
                    y = (i + 0.5) * spacing_y
                else:
                    x = (j + 1.0) * spacing_x
                    y = (i + 0.5) * spacing_y
                
                # Ensure we're within bounds and can fit a circle
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Start with a small radius
                    r = min(x, 1-x, y, 1-y) * 0.4
                    if r > 0:
                        circles.append([x, y, r])
        
        # Fill remaining positions if needed
        while len(circles) < n:
            # Add some random positions
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = min(x, 1-x, y, 1-y) * 0.3
            if r > 0:
                circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    # Initialize
    circles = initialize_hexagonal()
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints
        def boundary_constraint(i):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                return min(r_i, 1-r_i-x_i, 1-r_i-y_i, x_i-r_i, y_i-r_i)
            return constraint
        
        # Non-overlap constraints
        def overlap_constraint(i, j):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                return dist_sq - (r_i + r_j)**2
            return constraint
        
        # Add boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Add non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        return cons
    
    # Objective function (negative because we want to maximize)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]  # radius is third component
        return -total_radius
    
    # Flatten initial solution
    x0 = circles.flatten()
    
    # Set bounds for variables
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Get constraints
    cons = get_constraints()
    
    # Optimize using SLSQP method
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Clip values to valid ranges
            optimized_circles[:, 0] = np.clip(optimized_circles[:, 0], 0, 1)
            optimized_circles[:, 1] = np.clip(optimized_circles[:, 1], 0, 1)
            optimized_circles[:, 2] = np.clip(optimized_circles[:, 2], 0, 0.5)
            
            # Ensure all circles are within bounds
            for i in range(len(optimized_circles)):
                x, y, r = optimized_circles[i]
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    # Adjust radius to fit
                    r_new = min(x, 1-x, y, 1-y)
                    optimized_circles[i] = [x, y, max(0, r_new)]
            
            return optimized_circles
    except Exception as e:
        pass
    
    # If optimization fails, return initial configuration
    return circles


# EVOLVE-BLOCK-END
