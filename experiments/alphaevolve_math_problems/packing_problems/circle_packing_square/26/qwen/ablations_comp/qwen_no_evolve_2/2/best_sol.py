# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and numerical optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a hexagonal lattice pattern as starting point
    def initialize_hexagonal():
        # Create a hexagonal grid pattern
        circles = []
        # We'll place circles in a roughly hexagonal pattern
        rows = 5  # Number of rows
        cols = 6  # Number of columns (adjust for better fit)
        
        # Calculate spacing based on circle size
        spacing_x = 0.8  # Initial spacing
        spacing_y = 0.8  # Initial spacing
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x / (cols - 1) * 0.8
                y = 0.1 + i * spacing_y / (rows - 1) * 0.8
                # Adjust positions to avoid boundary issues
                if i % 2 == 1:
                    x += spacing_x / (cols - 1) * 0.4  # Offset every other row
                circles.append([x, y, 0.05])  # Start with small radius
        
        # Fill remaining positions
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.05])
            
        return np.array(circles[:n])
    
    # Create initial configuration
    circles = initialize_hexagonal()
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(params):
        # params contains [x1, y1, r1, x2, y2, r2, ...]
        total_radius = 0
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            total_radius += r
        return -total_radius  # Negative because we minimize
    
    # Define constraints
    def constraint_containment(i):
        def c(params):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            # Circle must be fully contained in unit square
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        return c
    
    def constraint_nonoverlap(i, j):
        def c(params):
            x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
            x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
            # Distance between centers must be at least r1 + r2
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            return math.sqrt(dist_sq) - (r1 + r2)
        return c
    
    # Create constraints list
    constraints = []
    
    # Add containment constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': constraint_containment(i)})
    
    # Add non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_nonoverlap(i, j)})
    
    # Set bounds for parameters [x, y, r] for each circle
    bounds = []
    for i in range(n):
        # x, y bounds: [r, 1-r] to ensure containment
        # r bounds: [0, 0.5] (reasonable upper bound)
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])
    
    # Extract initial values
    initial_params = []
    for i in range(n):
        initial_params.extend([circles[i][0], circles[i][1], circles[i][2]])
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            # Extract final configuration
            final_circles = np.zeros((n, 3))
            for i in range(n):
                final_circles[i] = [
                    result.x[3*i],
                    result.x[3*i+1],
                    result.x[3*i+2]
                ]
            return final_circles
    except Exception as e:
        pass
    
    # If optimization fails, return initial configuration
    return circles


# EVOLVE-BLOCK-END
