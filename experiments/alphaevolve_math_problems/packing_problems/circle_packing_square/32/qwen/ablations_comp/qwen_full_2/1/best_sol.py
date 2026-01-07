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
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using hexagonal lattice pattern for good starting configuration
    def initialize_hexagonal_layout():
        # Try to arrange circles in a hexagonal pattern
        # For 32 circles, we can try a 6x6 grid with some adjustments
        circles = np.zeros((n, 3))
        
        # Create hexagonal packing pattern
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Adjust spacing to allow for circle radii
        max_radius = min(spacing_x, spacing_y) / 2.0
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = 0.0 if i % 2 == 0 else spacing_x / 2.0
                x = (j * spacing_x) + x_offset + max_radius
                y = (i * spacing_y) + max_radius
                
                # Ensure circles fit within bounds
                if x - max_radius >= 0 and x + max_radius <= 1 and y - max_radius >= 0 and y + max_radius <= 1:
                    circles[idx] = [x, y, max_radius]
                    idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with smaller circles
        while idx < n:
            # Place remaining circles randomly within valid bounds
            x = np.random.uniform(max_radius, 1 - max_radius)
            y = np.random.uniform(max_radius, 1 - max_radius)
            # Small initial radius
            r = max_radius * 0.3
            circles[idx] = [x, y, r]
            idx += 1
            
        return circles
    
    # Constraint functions for optimization
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must be within the unit square
        def boundary_constraint(i):
            def constraint(x):
                # x[3*i:3*i+2] = (x,y), x[3*i+2] = r
                x_c, y_c, r = x[3*i], x[3*i+1], x[3*i+2]
                return min(r, 1 - r - x_c, 1 - r - y_c, x_c - r, y_c - r)
            return constraint
        
        # Non-overlap constraints
        def overlap_constraint(i, j):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                # Distance between centers minus sum of radii
                dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                return dist - (r_i + r_j)
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
        # Sum of all radii
        total_radius = sum(x[3*i+2] for i in range(n))
        return -total_radius
    
    # Initialize circles
    circles = initialize_hexagonal_layout()
    
    # Flatten initial guess for optimization
    x0 = np.array([circles[i][j] for i in range(n) for j in range(3)])
    
    # Set bounds for each variable (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y in [0,1], r in [0, 0.5]
    
    # Get constraints
    constraints = get_constraints()
    
    # Perform optimization using SLSQP method
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            # Extract results
            final_circles = np.zeros((n, 3))
            for i in range(n):
                final_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            return final_circles
    except Exception as e:
        pass
    
    # If optimization fails, return the initial configuration
    return circles


# EVOLVE-BLOCK-END
