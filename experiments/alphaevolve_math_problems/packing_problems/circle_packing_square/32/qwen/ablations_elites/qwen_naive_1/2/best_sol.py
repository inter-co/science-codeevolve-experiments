# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and mathematical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Step 1: Create a better initial hexagonal arrangement
    def initialize_hexagonal_layout():
        # Create a more optimal hexagonal packing pattern
        # Use 6 rows and 6 columns to get 36 positions, then take first 32
        rows = 6
        cols = 6
        
        # Hexagonal spacing - optimized for unit square
        spacing_x = 1.0 / (cols + 0.5)  # Leave some margin
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                # Offset every other row for hexagonal packing
                x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                if 0 <= x <= 1 and 0 <= y <= 1:
                    positions.append([x, y])
        
        # Trim to exact number needed
        positions = positions[:n]
        
        # Set initial radii - start with a reasonable value based on spacing
        radii = [spacing_x * 0.3] * n
        
        return np.array(positions), radii
    
    # Initialize with better hexagonal pattern
    positions, radii = initialize_hexagonal_layout()
    
    # Step 2: Use mathematical optimization to refine the solution
    # Flatten initial values: [x1, y1, r1, x2, y2, r2, ...]
    initial_vars = []
    for i in range(n):
        initial_vars.extend([positions[i][0], positions[i][1], radii[i]])
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y in [0,1], r in [0, 0.5]
    
    # Optimization objective: minimize negative sum of radii (equivalent to maximizing sum)
    def objective(vars):
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i+2]  # radius is third component
        return -total_radius  # negative because we're minimizing
    
    # Constraint functions
    def create_constraints():
        cons = []
        
        # Boundary constraints for each circle: x >= r, x <= 1-r, y >= r, y <= 1-r
        for i in range(n):
            # x >= r
            def bound_x_min(i):
                def constraint(vars):
                    return vars[3*i] - vars[3*i+2]
                return constraint
            cons.append({'type': 'ineq', 'fun': bound_x_min(i)})
            
            # x <= 1-r
            def bound_x_max(i):
                def constraint(vars):
                    return 1 - vars[3*i] - vars[3*i+2]
                return constraint
            cons.append({'type': 'ineq', 'fun': bound_x_max(i)})
            
            # y >= r
            def bound_y_min(i):
                def constraint(vars):
                    return vars[3*i+1] - vars[3*i+2]
                return constraint
            cons.append({'type': 'ineq', 'fun': bound_y_min(i)})
            
            # y <= 1-r
            def bound_y_max(i):
                def constraint(vars):
                    return 1 - vars[3*i+1] - vars[3*i+2]
                return constraint
            cons.append({'type': 'ineq', 'fun': bound_y_max(i)})
        
        # Non-overlap constraints: distance between centers >= sum of radii
        for i in range(n):
            for j in range(i+1, n):
                def overlap_constraint(i, j):
                    def constraint(vars):
                        x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                        x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                        # Return positive if circles don't overlap (constraint satisfied)
                        return dist_sq - (r1 + r2)**2
                    return constraint
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        return cons
    
    # Create constraints
    constraints = create_constraints()
    
    # Perform optimization using SLSQP with more iterations
    try:
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-6, 'eps': 1e-6},
            tol=1e-6
        )
        
        if result.success:
            final_vars = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_vars[3*i], final_vars[3*i+1], final_vars[3*i+2]]
            return circles
    except Exception as e:
        pass
    
    # Fallback: if optimization fails, return the initial hexagonal configuration
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [positions[i][0], positions[i][1], radii[i]]
    return circles


# EVOLVE-BLOCK-END
