# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using a hexagonal packing pattern for good starting configuration
    def initialize_hexagonal():
        # Create a hexagonal grid pattern
        circles = []
        
        # Parameters for hexagonal packing
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        # Adjust spacing to fit within unit square
        max_radius = min(spacing_x, spacing_y) / 2
        
        # Place circles in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Ensure we're within bounds
                if x - max_radius >= 0 and x + max_radius <= 1 and y - max_radius >= 0 and y + max_radius <= 1:
                    circles.append([x, y, max_radius])
        
        # Fill remaining positions with random valid placements
        while len(circles) < n:
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            max_radius = min(x, 1-x, y, 1-y)
            
            # Check if this placement is valid (not too close to existing circles)
            valid = True
            for cx, cy, _ in circles:
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < max_radius + 0.01:  # Small buffer
                    valid = False
                    break
            
            if valid:
                circles.append([x, y, max_radius])
                
        return np.array(circles)
    
    # Get initial configuration
    circles = initialize_hexagonal()
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must be fully inside the unit square
        def boundary_constraint(i):
            def constraint(x):
                x_c, y_c, r = x[3*i], x[3*i+1], x[3*i+2]
                return min(r, 1-r-x_c, 1-r-y_c, x_c-r, y_c-r)
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
    
    # Optimization objective (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]  # radius is third component
        return -total_radius
    
    # Flatten initial guess
    x0 = circles.flatten()
    
    # Set bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r
    
    # Get constraints
    constraints = get_constraints()
    
    # Perform optimization
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial configuration if optimization fails
            return circles
    except Exception as e:
        # Fallback to initial configuration if anything goes wrong
        return circles


# EVOLVE-BLOCK-END
