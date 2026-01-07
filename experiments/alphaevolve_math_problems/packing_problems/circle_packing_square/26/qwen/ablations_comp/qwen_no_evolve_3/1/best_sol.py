# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a good heuristic layout based on hexagonal packing
    def initialize_layout():
        # Create a grid-like initial configuration
        # We'll arrange circles in a pattern that approximates optimal packing
        circles = []
        
        # Try to arrange in roughly a 5x5 grid pattern with some adjustments
        rows = 5
        cols = 5
        
        # Calculate spacing to fit 26 circles approximately
        # Start with a rough estimate of how many circles we can fit
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Adjust for better packing
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Add some randomness to avoid perfect grid
                x = (j + 0.5 + (np.random.random() - 0.5) * 0.2) * spacing_x
                y = (i + 0.5 + (np.random.random() - 0.5) * 0.2) * spacing_y
                circles.append([x, y, min(x, 1-x, y, 1-y) * 0.4])  # Initial radius
        
        # Fill remaining positions
        while len(circles) < n:
            x = np.random.random()
            y = np.random.random()
            radius = min(x, 1-x, y, 1-y) * 0.4
            circles.append([x, y, radius])
            
        return np.array(circles)
    
    # Constraint functions for optimization
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must be fully inside the unit square
        def boundary_constraint(i):
            def constraint(vars):
                x, y, r = vars[3*i:3*i+3]
                return min(r, x - r, 1 - x - r, y - r, 1 - y - r)
            return constraint
        
        # Non-overlap constraints
        def overlap_constraint(i, j):
            def constraint(vars):
                x1, y1, r1 = vars[3*i:3*i+3]
                x2, y2, r2 = vars[3*j:3*j+3]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                return dist - (r1 + r2)
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
    def objective(vars):
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i+2]  # Radius is third component
        return -total_radius
    
    # Initial guess
    initial_circles = initialize_layout()
    initial_vars = initial_circles.flatten()
    
    # Set bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])  # x, y, r
    
    # Get constraints
    constraints = get_constraints()
    
    # Optimization using SLSQP method which handles constraints well
    try:
        result = minimize(objective, initial_vars, method='SLSQP', bounds=bounds, constraints=constraints, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            final_vars = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = final_vars[3*i:3*i+3]
            return circles
        else:
            # Fallback to initial configuration if optimization fails
            return initial_circles
    except Exception as e:
        # Return initial configuration if anything goes wrong
        return initial_circles


# EVOLVE-BLOCK-END
