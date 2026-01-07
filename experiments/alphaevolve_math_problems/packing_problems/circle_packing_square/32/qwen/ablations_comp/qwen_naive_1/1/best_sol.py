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
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using a hexagonal packing pattern as starting point
    def initialize_hexagonal():
        # Try to arrange in a hexagonal pattern
        circles = []
        rows = int(math.sqrt(n))
        cols = (n + rows - 1) // rows  # Ceiling division
        
        # Calculate spacing based on desired density
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Create grid points
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Adjust for hexagonal pattern (odd rows offset)
                if i % 2 == 1:
                    x += spacing_x * 0.5
                
                # Initial radius - small enough to fit in space
                r = min(spacing_x, spacing_y) * 0.4
                
                circles.append([x, y, r])
                
        # Fill remaining slots if needed
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.05])
            
        return np.array(circles[:n])
    
    # Get initial configuration
    circles = initialize_hexagonal()
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: radius must be such that circle fits
        def boundary_constraint(i):
            def constraint(x):
                x_c, y_c, r = x[3*i], x[3*i+1], x[3*i+2]
                return min(x_c - r, 1 - x_c - r, y_c - r, 1 - y_c - r)
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
    
    # Objective function to maximize (negative because minimize)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]  # radius is third component
        return -total_radius
    
    # Constraints
    constraints = get_constraints()
    
    # Variable bounds: x, y in [0,1], r > 0
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (1e-6, 0.5)])  # x, y, r
    
    # Flatten initial guess
    x0 = []
    for circle in circles:
        x0.extend(circle)
    
    # Optimization parameters
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Run optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options=options, 
                         tol=1e-6)
        
        # Extract optimized circles
        if result.success:
            circles_opt = []
            for i in range(n):
                x = result.x[3*i]
                y = result.x[3*i+1]
                r = result.x[3*i+2]
                circles_opt.append([x, y, r])
            return np.array(circles_opt)
        else:
            # Return initial configuration if optimization fails
            return circles
    except Exception as e:
        # Fallback to initial configuration
        return circles


# EVOLVE-BLOCK-END
