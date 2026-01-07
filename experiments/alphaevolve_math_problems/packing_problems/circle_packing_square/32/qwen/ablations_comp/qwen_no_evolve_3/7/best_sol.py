# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial hexagonal grid placement with optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a hexagonal grid pattern for good initial placement
    def initialize_hexagonal_placement():
        # Create a hexagonal grid pattern
        # For 32 circles, we can arrange in roughly 6 rows and 6 columns
        rows = 6
        cols = 6
        circles = []
        
        # Hexagonal packing parameters
        spacing_x = 1.0 / cols
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        # Adjust spacing so that circles don't exceed bounds
        max_radius = min(spacing_x, spacing_y) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                
                # Ensure we stay within bounds
                x = max(max_radius, min(1 - max_radius, x))
                y = max(max_radius, min(1 - max_radius, y))
                
                circles.append([x, y, max_radius])
        
        # Fill remaining circles with smaller radii if needed
        while len(circles) < n:
            # Place remaining circles near edges with small radii
            x = 0.1 + 0.8 * np.random.random()
            y = 0.1 + 0.8 * np.random.random()
            r = 0.01 + 0.05 * np.random.random()
            
            # Ensure it's within bounds
            x = max(r, min(1 - r, x))
            y = max(r, min(1 - r, y))
            
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must fit entirely in the unit square
        def bound_constraint(i):
            def constraint(x):
                idx = i * 3
                x_c, y_c, r = x[idx], x[idx+1], x[idx+2]
                return min(x_c - r, 1 - x_c - r, y_c - r, 1 - y_c - r)
            return constraint
        
        # Non-overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(i, j):
            def constraint(x):
                idx_i = i * 3
                idx_j = j * 3
                x_i, y_i, r_i = x[idx_i], x[idx_i+1], x[idx_i+2]
                x_j, y_j, r_j = x[idx_j], x[idx_j+1], x[idx_j+2]
                dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                return dist - r_i - r_j
            return constraint
        
        # Add boundary constraints for all circles
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': bound_constraint(i)})
        
        # Add non-overlap constraints for all pairs
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        return cons
    
    # Initialize
    circles = initialize_hexagonal_placement()
    
    # Flatten initial configuration for optimization
    initial_guess = circles.flatten()
    
    # Define objective function (negative because we want to maximize)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[i*3 + 2]  # radius is at index 2 for each circle
        return -total_radius  # Negative because we minimize
    
    # Get constraints
    constraints = get_constraints()
    
    # Run optimization
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6},
            tol=1e-6
        )
        
        if result.success:
            # Extract final positions and radii
            final_circles = result.x.reshape(-1, 3)
            return final_circles
        else:
            # Return initial guess if optimization fails
            return circles
    except Exception as e:
        # Return initial guess if optimization fails due to error
        return circles


# EVOLVE-BLOCK-END
