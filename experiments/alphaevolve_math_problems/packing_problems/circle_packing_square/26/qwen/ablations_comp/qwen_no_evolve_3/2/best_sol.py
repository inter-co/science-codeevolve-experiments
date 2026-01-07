# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a hexagonal-like pattern for good starting configuration
    def initialize_hexagonal():
        # Create a roughly hexagonal arrangement
        circles = []
        rows = 5  # approximate number of rows
        cols = 5  # approximate number of columns
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Adjust for hexagonal pattern (odd rows offset)
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                # Start with small radius
                r = min(x, 1-x, y, 1-y) * 0.4
                circles.append([x, y, r])
                
        # Fill remaining positions
        while len(circles) < n:
            # Add random positions with small radii
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = min(x, 1-x, y, 1-y) * 0.3
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Initialize
    initial_circles = initialize_hexagonal()
    
    # Optimization variables: [x1, y1, r1, x2, y2, r2, ...]
    def pack_to_vars(circles):
        return np.array([circles[i][j] for i in range(n) for j in range(3)])
    
    def vars_to_pack(vars_array):
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [vars_array[3*i], vars_array[3*i+1], vars_array[3*i+2]]
        return circles
    
    # Constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: radius <= x <= 1-radius, radius <= y <= 1-radius
        for i in range(n):
            def boundary_constraint(vars, idx=i):
                x, y, r = vars[3*idx], vars[3*idx+1], vars[3*idx+2]
                return [r - x, r - y, x + r - 1, y + r - 1]
            
            # Add bounds as constraints
            cons.append({'type': 'ineq', 'fun': lambda vars, idx=i: vars[3*idx] - vars[3*idx+2]})  # x >= r
            cons.append({'type': 'ineq', 'fun': lambda vars, idx=i: vars[3*idx+1] - vars[3*idx+2]})  # y >= r
            cons.append({'type': 'ineq', 'fun': lambda vars, idx=i: 1 - vars[3*idx] - vars[3*idx+2]})  # x + r <= 1
            cons.append({'type': 'ineq', 'fun': lambda vars, idx=i: 1 - vars[3*idx+1] - vars[3*idx+2]})  # y + r <= 1
            
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                def overlap_constraint(vars, idx1=i, idx2=j):
                    x1, y1, r1 = vars[3*idx1], vars[3*idx1+1], vars[3*idx1+2]
                    x2, y2, r2 = vars[3*idx2], vars[3*idx2+1], vars[3*idx2+2]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    return dist_sq - (r1 + r2)**2  # Should be >= 0
                
                cons.append({'type': 'ineq', 'fun': overlap_constraint})
                
        return cons
    
    # Objective function (negative because we want to maximize)
    def objective(vars):
        return -np.sum(vars[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Get constraints
    constraints = get_constraints()
    
    # Set up bounds for variables: [x, y, r] for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Initial guess
    x0 = pack_to_vars(initial_circles)
    
    # Optimize
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-4},
            callback=lambda x: None  # Simple callback
        )
        
        if result.success:
            final_circles = vars_to_pack(result.x)
            # Ensure all circles fit properly
            for i in range(n):
                final_circles[i][2] = max(0.001, min(0.499, final_circles[i][2]))
            return final_circles
        else:
            # Return initial configuration if optimization fails
            return initial_circles
            
    except Exception as e:
        # Fallback to initial configuration if anything goes wrong
        return initial_circles


# EVOLVE-BLOCK-END
