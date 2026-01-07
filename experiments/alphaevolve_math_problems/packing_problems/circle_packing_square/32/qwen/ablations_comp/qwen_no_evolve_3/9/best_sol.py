# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a hexagonal grid pattern for good initial placement
    def initialize_hexagonal_grid():
        # Create a hexagonal grid pattern for initial positions
        circles = []
        rows = int(math.sqrt(n)) + 2
        cols = int(n / rows) + 2
        
        # Hexagonal spacing
        spacing_x = 1.0 / cols
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + (i % 2) * 0.5) * spacing_x
                y = i * spacing_y
                if x <= 1 and y <= 1:
                    circles.append([x, y, min(x, 1-x, y, 1-y) * 0.4])  # Initial radius based on distance to edges
        
        # Fill remaining positions if needed
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.1])
            
        return np.array(circles[:n])
    
    # Initialize
    circles = initialize_hexagonal_grid()
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(params):
        # Reshape params back into circles array
        positions = params.reshape(-1, 2)
        radii = params[-n:]  # Last n elements are radii
        
        # Calculate sum of radii (negative for minimization)
        return -np.sum(radii)
    
    # Define constraint functions
    def constraint_containment(i):
        def constraint(params):
            positions = params.reshape(-1, 2)
            radii = params[-n:]
            x, y = positions[i]
            r = radii[i]
            # Circle must fit in unit square
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        return constraint
    
    def constraint_nonoverlap(i, j):
        def constraint(params):
            positions = params.reshape(-1, 2)
            radii = params[-n:]
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            r1 = radii[i]
            r2 = radii[j]
            # Distance between centers must be at least sum of radii
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            return dist - (r1 + r2)
        return constraint
    
    # Create constraints
    constraints = []
    
    # Containment constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': constraint_containment(i)})
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_nonoverlap(i, j)})
    
    # Bounds for variables (positions and radii)
    bounds = []
    # Positions: 0 <= x <= 1, 0 <= y <= 1
    for i in range(n):
        bounds.extend([(0, 1), (0, 1)])  # x, y bounds
    
    # Radii: 0 <= r <= 0.5 (max possible radius for a circle in unit square)
    for i in range(n):
        bounds.append((0, 0.5))
    
    # Flatten initial guess
    initial_guess = np.concatenate([circles[:, :2].flatten(), circles[:, 2]])
    
    # Optimization parameters
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Perform optimization
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints, 
                         options=options, tol=1e-6)
        
        if result.success:
            # Extract final solution
            final_positions = result.x[:-n].reshape(-1, 2)
            final_radii = result.x[-n:]
            
            # Construct final circles array
            circles = np.column_stack([final_positions, final_radii])
        else:
            # If optimization fails, return the initial configuration
            pass
            
    except Exception as e:
        # In case of error, return the initial configuration
        pass
    
    return circles


# EVOLVE-BLOCK-END
