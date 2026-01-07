# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal packing initialization with scipy optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with hexagonal packing pattern
    def initialize_hexagonal():
        # Create a hexagonal grid pattern
        circles = []
        rows = int(math.sqrt(n) * 1.2)
        cols = int(n / rows) + 1
        
        # Hexagonal packing parameters
        radius_guess = 0.05  # Initial guess
        spacing_x = radius_guess * 2 * 1.1  # Slightly reduced to allow for optimization
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x = spacing_x * j + radius_guess
                y = spacing_y * i + radius_guess
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                # Check bounds
                if x + radius_guess <= 1 and y + radius_guess <= 1:
                    circles.append([x, y, radius_guess])
                    count += 1
            if count >= n:
                break
        
        # Fill remaining positions with random valid circles
        while len(circles) < n:
            x = np.random.uniform(radius_guess, 1-radius_guess)
            y = np.random.uniform(radius_guess, 1-radius_guess)
            circles.append([x, y, radius_guess])
            
        return np.array(circles)
    
    # Create initial configuration
    circles = initialize_hexagonal()
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: radius must be small enough to fit
        def bound_constraint(i):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                return min(x_i - r_i, 1 - x_i - r_i, y_i - r_i, 1 - y_i - r_i)
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
            cons.append({'type': 'ineq', 'fun': bound_constraint(i)})
            
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
    
    # Constraints
    constraints = get_constraints()
    
    # Bounds for variables: [x, y, r] for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y in [0,1], r in [0, 0.5]
    
    # Flatten initial solution
    x0 = circles.flatten()
    
    # Optimize using SLSQP method
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # If optimization fails, return the initial configuration
            return circles
    except Exception as e:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
