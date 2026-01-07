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
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a grid-based approach
    def initialize_grid():
        # Create a grid pattern for initial placement
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        
        # Adjust grid size to fit within unit square with margin
        margin = 0.05
        cell_size_x = (1 - 2*margin) / cols
        cell_size_y = (1 - 2*margin) / rows
        
        circles = []
        idx = 0
        
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                    
                x = margin + (j + 0.5) * cell_size_x
                y = margin + (i + 0.5) * cell_size_y
                
                # Initial radius estimate based on available space
                min_dist = min(x, 1-x, y, 1-y)
                max_radius = min_dist * 0.4  # Conservative estimate
                
                circles.append([x, y, max_radius])
                idx += 1
                
        return np.array(circles)
    
    # Generate initial configuration
    circles = initialize_grid()
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must fit entirely in the unit square
        def boundary_constraint(i):
            def constraint(x):
                x_c, y_c, r = x[3*i], x[3*i+1], x[3*i+2]
                return min(r, 1-r-x_c, x_c-r, 1-r-y_c, y_c-r)
            return constraint
        
        # Non-overlap constraints
        def overlap_constraint(i, j):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                distance = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                return distance - (r_i + r_j)
            return constraint
            
        # Add boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
            
        # Add non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
                
        return cons
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]  # radius is at index 3*i+2
        return -total_radius
    
    # Flatten initial circles for optimization
    x0 = circles.flatten()
    
    # Set bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Get constraints
    constraints = get_constraints()
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            # Extract optimized solution
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Fallback to initial configuration if optimization fails
            return circles
    except Exception:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
