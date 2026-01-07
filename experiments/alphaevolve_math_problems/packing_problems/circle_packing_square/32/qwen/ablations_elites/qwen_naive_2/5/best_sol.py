# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and optimization with SLSQP.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using a hexagonal grid pattern for good starting configuration
    def initialize_hexagonal():
        # Create a hexagonal grid pattern that fits well in unit square
        circles = []
        
        # Use a fixed 6x6 grid for consistency and better results
        rows, cols = 6, 6
        
        # Grid spacing based on hexagonal packing
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Adjust spacing to allow for proper circle packing
        max_radius = min(spacing_x, spacing_y) * 0.4
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = (i % 2) * spacing_x * 0.5
                x = (j * spacing_x) + x_offset + max_radius
                y = i * spacing_y + max_radius
                
                # Ensure we're within bounds
                if x <= 1 - max_radius and y <= 1 - max_radius:
                    circles.append([x, y, max_radius])
        
        # Fill remaining positions with random placements if needed
        while len(circles) < n:
            x = np.random.uniform(max_radius, 1 - max_radius)
            y = np.random.uniform(max_radius, 1 - max_radius)
            circles.append([x, y, max_radius])
            
        return np.array(circles)
    
    # Generate initial configuration
    circles = initialize_hexagonal()
    
    # Define constraint functions for optimization - more efficient implementation
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must be fully contained
        def boundary_constraint(i):
            def constraint(x):
                x_c, y_c, r = x[3*i], x[3*i+1], x[3*i+2]
                # Return minimum of all boundary violations (should be positive when satisfied)
                return min(x_c - r, 1 - x_c - r, y_c - r, 1 - y_c - r)
            return constraint
        
        # Non-overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(i, j):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                # Direct distance calculation
                dx = x_i - x_j
                dy = y_i - y_j
                dist = math.sqrt(dx*dx + dy*dy)
                return dist - (r_i + r_j)
            return constraint
        
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
        # Sum of all radii (negative because we minimize)
        return -np.sum(x[2::3])  # Every third element starting from index 2
    
    # Flatten initial circles for optimization
    x0 = circles.flatten()
    
    # Set bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds  
        bounds.append((0.001, 0.999))
        # r bounds (must be positive and small enough to fit in square)
        bounds.append((0.001, 0.499))
    
    # Get constraints
    constraints = get_constraints()
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-6, 'disp': False})
        
        # Extract optimized solution
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial configuration if optimization fails
            return circles
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
