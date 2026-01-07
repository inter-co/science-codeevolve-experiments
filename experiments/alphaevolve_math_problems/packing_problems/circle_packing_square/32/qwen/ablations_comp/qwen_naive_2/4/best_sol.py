# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a hexagonal grid pattern for good initial distribution
    def initialize_hexagonal_grid():
        # Create a hexagonal grid pattern that fits within the unit square
        circles = []
        
        # Hexagonal packing parameters
        sqrt3 = math.sqrt(3)
        # Calculate grid spacing based on number of circles
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        
        # Adjust spacing so circles fit in unit square
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Create hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Offset every other row for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                # Ensure we're within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius - small enough to fit in square
                    r = min(x, 1-x, y, 1-y) * 0.4
                    circles.append([x, y, r])
        
        # Fill remaining slots if needed
        while len(circles) < n:
            # Add circles at random positions with small radii
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = min(x, 1-x, y, 1-y) * 0.2
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Constraint functions for optimization
    def get_constraints():
        """Generate constraint functions for scipy.optimize"""
        cons = []
        
        # Boundary constraints: each circle must stay within unit square
        def boundary_constraint(i):
            def constraint(x):
                idx = i * 3
                x_pos, y_pos, radius = x[idx], x[idx+1], x[idx+2]
                # Radius must be positive and circles must stay within bounds
                return min(radius, x_pos - radius, 1 - x_pos - radius,
                          y_pos - radius, 1 - y_pos - radius)
            return constraint
        
        # Non-overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(i, j):
            def constraint(x):
                idx_i = i * 3
                idx_j = j * 3
                x_i, y_i, r_i = x[idx_i], x[idx_i+1], x[idx_i+2]
                x_j, y_j, r_j = x[idx_j], x[idx_j+1], x[idx_j+2]
                # Distance between centers minus sum of radii (should be >= 0)
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
    
    # Initialize circles
    circles = initialize_hexagonal_grid()
    
    # Flatten initial values for optimization
    initial_guess = circles.flatten()
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[i*3 + 2]  # Extract radius for each circle
        return -total_radius  # Negative because we're minimizing
    
    # Get constraints
    constraints = get_constraints()
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(objective, initial_guess, method='SLSQP', 
                         constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure valid ranges for final output
            for i in range(n):
                # Clip radii to reasonable values
                optimized_circles[i, 2] = max(0.001, min(0.5, optimized_circles[i, 2]))
                # Ensure positions are within bounds
                optimized_circles[i, 0] = max(optimized_circles[i, 2], 
                                             min(1 - optimized_circles[i, 2], 
                                                 optimized_circles[i, 0]))
                optimized_circles[i, 1] = max(optimized_circles[i, 2], 
                                             min(1 - optimized_circles[i, 2], 
                                                 optimized_circles[i, 1]))
            return optimized_circles
        else:
            # Return initial guess if optimization fails
            return circles
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
