# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a hexagonal packing pattern as starting point
    def initialize_hexagonal():
        circles = np.zeros((n, 3))
        
        # Try to arrange in a hexagonal pattern
        rows = 5
        cols = 5
        if n < 25:
            rows = math.ceil(math.sqrt(n))
            cols = math.ceil(n / rows)
        else:
            rows = 5
            cols = 5
            
        # Create a hexagonal grid
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset odd rows for hexagonal packing
                x_offset = 0.0 if i % 2 == 0 else 0.5 * spacing_x
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Initial radius - small enough to fit in space
                r = min(spacing_x, spacing_y) * 0.4
                
                circles[idx] = [x, y, r]
                idx += 1
                
                if idx >= n:
                    break
            if idx >= n:
                break
        
        # Fill remaining slots with random valid positions
        for i in range(idx, n):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.1)
            circles[i] = [x, y, r]
            
        return circles
    
    # Constraint functions
    def get_constraints(circles):
        """Get all constraint functions"""
        cons = []
        
        # Boundary constraints: each circle must fit completely in the unit square
        def boundary_constraint(i):
            def func(x):
                x_c, y_c, r = x[3*i], x[3*i+1], x[3*i+2]
                # Circle must be within bounds
                return min(r, 1-r, x_c-r, 1-x_c-r, y_c-r, 1-y_c-r)
            return func
        
        # Non-overlap constraints
        def overlap_constraint(i, j):
            def func(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                # Distance between centers minus sum of radii must be >= 0
                dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                return dist - (r_i + r_j)
            return func
        
        # Add boundary constraints for all circles
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Add non-overlap constraints for all pairs
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        return cons
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]  # radius is third component
        return -total_radius
    
    # Initialize circles
    circles = initialize_hexagonal()
    
    # Flatten initial guess for optimization
    x0 = []
    for i in range(n):
        x0.extend([circles[i][0], circles[i][1], circles[i][2]])
    
    # Get constraints
    constraints = get_constraints(circles)
    
    # Define bounds for variables (x, y, r) - all must be positive and within square
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0.001, 0.999))  # slightly inside to avoid boundary issues
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds
        bounds.append((0.001, 0.499))  # max radius is 0.5 (would touch opposite edges)
    
    # Optimization parameters
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Run optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options=options, 
                         tol=1e-6)
        
        # Extract optimized circles
        optimized_circles = np.zeros((n, 3))
        for i in range(n):
            optimized_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            
        return optimized_circles
        
    except Exception as e:
        # If optimization fails, return initial configuration
        return circles


# EVOLVE-BLOCK-END
