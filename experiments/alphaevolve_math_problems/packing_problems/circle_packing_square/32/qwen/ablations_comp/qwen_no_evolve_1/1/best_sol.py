# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a physics-inspired optimization approach with constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize positions and radii
    # Start with a hexagonal-like arrangement for good initial placement
    circles = np.zeros((n, 3))
    
    # Create a hexagonal grid pattern as starting point
    rows = 6
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Adjust for hexagonal packing effect
            if i % 2 == 1:
                x += spacing_x * 0.5
            circles[idx] = [x, y, 0.02]  # Start with small radii
            idx += 1
        if idx >= n:
            break
    
    # Truncate if we have too many points
    circles = circles[:n]
    
    # Define constraint functions
    def constraint_radius(i):
        """Ensure each circle stays within bounds"""
        def func(vars):
            x, y, r = vars[3*i:3*i+3]
            # Radius constraint: r <= min(x, 1-x, y, 1-y)
            return min(x, 1-x, y, 1-y) - r
        return func
    
    def constraint_nonoverlap(i, j):
        """Ensure circles don't overlap"""
        def func(vars):
            x1, y1, r1 = vars[3*i:3*i+3]
            x2, y2, r2 = vars[3*j:3*j+3]
            # Distance constraint: sqrt((x1-x2)^2 + (y1-y2)^2) >= r1 + r2
            dist_sq = (x1-x2)**2 + (y1-y2)**2
            return np.sqrt(dist_sq) - (r1 + r2)
        return func
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    def objective(vars):
        # We want to maximize sum of radii (minimize negative sum)
        total_radius = sum(vars[2::3])  # Every third element starting from index 2
        return -total_radius
    
    # Build constraints
    constraints = []
    
    # Add boundary constraints for all circles
    for i in range(n):
        # Each circle's radius cannot exceed its distance to boundaries
        constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: constraint_radius(i)(vars)})
    
    # Add non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': lambda vars, i=i, j=j: constraint_nonoverlap(i, j)(vars)})
    
    # Initial guess
    initial_vars = circles.flatten()
    
    # Set bounds for variables: x in [0,1], y in [0,1], r in [0,0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Optimization with bounds and constraints
    try:
        result = minimize(objective, initial_vars, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_vars = result.x
            circles = optimized_vars.reshape(-1, 3)
        else:
            # If optimization fails, return the initial configuration
            pass
            
    except Exception as e:
        # If anything goes wrong, fall back to initial configuration
        pass
    
    # Final validation and cleanup
    # Ensure all circles are valid
    for i in range(n):
        x, y, r = circles[i]
        # Make sure radius is positive and within bounds
        if r < 0:
            r = 0.001
        # Ensure circle fits in unit square
        r = min(r, x, 1-x, y, 1-y)
        circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
