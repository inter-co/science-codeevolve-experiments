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
    
    # Initialize circles using a hexagonal packing pattern for good starting configuration
    def initialize_hexagonal_pattern():
        circles = []
        # Try to arrange in roughly a 6x6 grid pattern (but we have 32 circles)
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Adjust spacing to account for circle radii
        max_radius = min(spacing_x, spacing_y) / 2.0
        
        # Place circles in a hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = 0.0 if i % 2 == 0 else spacing_x / 2.0
                x = spacing_x * j + x_offset + max_radius
                y = spacing_y * i + max_radius
                
                # Ensure circles are within bounds
                if x <= 1 - max_radius and y <= 1 - max_radius:
                    circles.append([x, y, max_radius])
        
        # Fill remaining positions with smaller circles
        while len(circles) < n:
            # Add circles near boundaries for better space utilization
            x = np.random.uniform(max_radius, 1 - max_radius)
            y = np.random.uniform(max_radius, 1 - max_radius)
            # Small initial radius
            r = min(0.05, 1 - x - max_radius, 1 - y - max_radius, x - max_radius, y - max_radius)
            if r > 0:
                circles.append([x, y, r])
                
        return np.array(circles)
    
    # Create initial configuration
    circles = initialize_hexagonal_pattern()
    
    # Define constraint functions
    def get_constraints():
        """Generate constraints for optimization"""
        cons = []
        
        # Boundary constraints: each circle must fit entirely in the unit square
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
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(x):
        # x contains [x1, y1, r1, x2, y2, r2, ...]
        total_radius = sum(x[3*i+2] for i in range(n))
        return -total_radius  # Negative because we minimize
    
    # Flatten initial circles for optimization
    x0 = []
    for circle in circles:
        x0.extend(circle)
    
    # Set bounds for variables (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y in [0,1], r in [0, 0.5]
    
    # Get constraints
    constraints = get_constraints()
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        # Extract final solution
        if result.success:
            circles_opt = np.zeros((n, 3))
            for i in range(n):
                circles_opt[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            return circles_opt
        else:
            # Return initial configuration if optimization fails
            return circles
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
