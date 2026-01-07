# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a hexagonal packing pattern as starting point
    def initialize_hexagonal_packing():
        # Create a hexagonal grid pattern that fits within the unit square
        circles = []
        
        # Parameters for hexagonal packing
        sqrt3 = math.sqrt(3)
        # Try to fit circles in a hexagonal pattern
        rows = int(math.sqrt(n) * 1.2)
        cols = int(n / rows) + 1
        
        # Adjust spacing to fit within unit square
        spacing_x = 1.0 / max(cols, 1)
        spacing_y = 1.0 / max(rows, 1)
        
        # Use smaller spacing for better initial configuration
        spacing_x = min(0.2, 1.0 / cols) if cols > 0 else 0.1
        spacing_y = min(0.2, 1.0 / rows) if rows > 0 else 0.1
        
        # Create hexagonal grid with offset rows
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row
                x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Ensure we're within bounds and add some randomness
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius estimate based on spacing
                    r = min(spacing_x, spacing_y) * 0.4
                    # Make sure radius is valid for placement
                    if r <= x and r <= 1-x and r <= y and r <= 1-y:
                        circles.append([x, y, r])
            
            if len(circles) >= n:
                break
        
        # Fill remaining positions with random valid circles
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = min(x, 1-x, y, 1-y) * 0.3
            if r > 0.01:
                circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    # Define constraint functions
    def get_constraints():
        """Generate constraints for the optimization"""
        cons = []
        
        # Boundary constraints: each circle must fit completely in unit square
        def boundary_constraint(i):
            def constraint(vars):
                x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
                return min(r, x-r, 1-x-r, y-r, 1-y-r)
            return constraint
        
        # Non-overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(i, j):
            def constraint(vars):
                x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
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
    
    # Optimization objective
    def objective(vars):
        # Sum of radii (we want to maximize this)
        return -sum(vars[2::3])  # Negative because minimize
    
    # Generate initial guess
    initial_circles = initialize_hexagonal_packing()
    initial_guess = initial_circles.flatten()
    
    # Set bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x, y in [0,1], r in [0,0.5] (reasonable upper bound)
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
    
    # Get constraints
    constraints = get_constraints()
    
    try:
        # Perform optimization
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            # Ensure all circles are properly contained
            for i in range(n):
                x, y, r = final_circles[i]
                # Clamp values to ensure containment
                r = min(r, x, 1-x, y, 1-y)
                final_circles[i] = [x, y, max(0, r)]
            return final_circles
        else:
            # Return initial configuration if optimization fails
            return initial_circles
            
    except Exception as e:
        # Return initial configuration if optimization fails
        return initial_circles


# EVOLVE-BLOCK-END
