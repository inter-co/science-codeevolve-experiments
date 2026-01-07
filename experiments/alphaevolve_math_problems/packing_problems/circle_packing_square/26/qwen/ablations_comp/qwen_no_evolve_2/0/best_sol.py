# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initial geometric placement using hexagonal packing pattern
    def generate_hexagonal_initial():
        # Try to place circles in a hexagonal lattice pattern
        circles = []
        
        # Parameters for hexagonal packing
        rows = 5
        cols = 5
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
                if 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append([x, y, max_radius])
        
        # If we don't have enough circles, fill with more
        if len(circles) < n:
            # Add more circles in a grid pattern
            for i in range(n - len(circles)):
                x = 0.1 + (i % 5) * 0.2
                y = 0.1 + (i // 5) * 0.2
                circles.append([x, y, 0.05])
        
        return np.array(circles[:n])
    
    # Generate initial configuration
    initial_circles = generate_hexagonal_initial()
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must be fully contained
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
                dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                return dist - r_i - r_j
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
        total_radius = sum(x[3*i+2] for i in range(n))
        return -total_radius
    
    # Flatten initial configuration
    x0 = np.array([item for circle in initial_circles for item in circle])
    
    # Create constraints
    constraints = get_constraints()
    
    # Bounds for variables: x, y in [0,1], r in [0,0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Optimization parameters
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Run optimization
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=1e-6
        )
        
        if result.success:
            # Extract final configuration
            final_circles = []
            for i in range(n):
                x, y, r = result.x[3*i], result.x[3*i+1], result.x[3*i+2]
                final_circles.append([x, y, r])
            
            return np.array(final_circles)
        else:
            # If optimization fails, return initial configuration
            return initial_circles
            
    except Exception as e:
        # Return initial configuration if anything goes wrong
        return initial_circles


# EVOLVE-BLOCK-END
