# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    n = 32
    
    # Initial geometric placement using hexagonal packing pattern
    def initialize_hexagonal_placement():
        # Create a hexagonal grid pattern that fits in the unit square
        circles = []
        
        # Try different arrangements to find a good starting point
        rows = 6
        cols = 6
        
        # Hexagonal packing with spacing
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Adjust spacing for hexagonal packing
        hex_spacing_x = spacing_x
        hex_spacing_y = spacing_y * math.sqrt(3) / 2
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = 0.0 if i % 2 == 0 else hex_spacing_x / 2
                x = (j + 1) * hex_spacing_x + x_offset
                y = (i + 1) * hex_spacing_y
                
                # Ensure we're within bounds
                if x <= 1 and y <= 1:
                    # Initial radius based on available space
                    min_dist = min(x, 1-x, y, 1-y)
                    radius = min_dist / 2.0
                    if radius > 0:
                        circles.append([x, y, radius])
                        count += 1
                        
        # Fill remaining positions with random placements near edges
        while len(circles) < n:
            # Random positions near boundaries to start with
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            min_dist = min(x, 1-x, y, 1-y)
            radius = min(min_dist / 3.0, 0.2)  # Cap at reasonable value
            if radius > 0:
                circles.append([x, y, radius])
                
        return np.array(circles)
    
    # Initialize with geometric placement
    circles = initialize_hexagonal_placement()
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(params):
        # params contains [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
        return -np.sum(params[2::3])  # Negative because scipy minimizes
    
    # Define constraints properly
    def constraint_containment(i):
        def constraint_func(params):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            # Circle must be fully contained in unit square:
            # x >= r, y >= r, x <= 1-r, y <= 1-r
            # Return the minimum of these four constraints (positive when satisfied)
            return min(x - r, y - r, 1 - x - r, 1 - y - r)
        return constraint_func
    
    def constraint_nonoverlap(i, j):
        def constraint_func(params):
            x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
            x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
            # Distance squared between centers must be at least (r1 + r2)^2
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            return dist_sq - (r1 + r2)**2
        return constraint_func
    
    # Build constraints list - use a simpler approach to avoid numerical issues
    constraints = []
    
    # Add containment constraints for each circle
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': constraint_containment(i)})
    
    # Add non-overlap constraints but with a more careful approach
    # Limit the number of constraints by only adding those that are most likely to be binding
    # For 32 circles, we can use a subset approach or just use all constraints with better handling
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_nonoverlap(i, j)})
    
    # Bounds for parameters [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
    bounds = []
    for i in range(n):
        # x, y in [0,1], r > 0 (bounded to prevent numerical issues)
        bounds.extend([(0, 1), (0, 1), (1e-6, 0.499)])  # r bounded by 0.499 to prevent boundary issues
    
    # Flatten initial guess
    x0 = circles.flatten()
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 500, 'ftol': 1e-7, 'disp': False})
        
        if result.success:
            # Extract final solution
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial solution if optimization fails
            return circles
            
    except Exception as e:
        # Return initial solution if optimization fails
        return circles


# EVOLVE-BLOCK-END
