# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a good heuristic layout - hexagonal packing pattern
    def initialize_layout():
        # Create a hexagonal grid pattern
        rows = 6
        cols = 6
        circles = []
        
        # Hexagonal packing parameters
        spacing_x = 0.15
        spacing_y = 0.15 * np.sqrt(3)/2
        
        # Generate positions in hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Offset odd rows
                if i % 2 == 1:
                    x += spacing_x / 2
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, 0.05])
        
        # Fill remaining slots with random positions
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        return np.array(circles)
    
    # Get initial configuration
    circles = initialize_layout()
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Containment constraints: radius <= x <= 1-radius, radius <= y <= 1-radius
        def containment_constraint(i):
            def constraint(x):
                idx = i * 3
                x_pos, y_pos, radius = x[idx], x[idx+1], x[idx+2]
                return min(radius, 1-radius-x_pos, 1-radius-y_pos, x_pos-radius, y_pos-radius)
            return constraint
        
        # Non-overlap constraints
        def non_overlap_constraint(i, j):
            def constraint(x):
                idx_i = i * 3
                idx_j = j * 3
                x_i, y_i, r_i = x[idx_i], x[idx_i+1], x[idx_i+2]
                x_j, y_j, r_j = x[idx_j], x[idx_j+1], x[idx_j+2]
                dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                # Return negative value when circles overlap (constraint violated)
                return np.sqrt(dist_sq) - (r_i + r_j)
            return constraint
        
        # Add containment constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': containment_constraint(i)})
        
        # Add non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': non_overlap_constraint(i, j)})
        
        return cons
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[i*3 + 2]  # radius is third element of each circle
        return -total_radius
    
    # Bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x coordinate: [radius, 1-radius]
        bounds.append((0.001, 0.999))  # slightly away from boundaries for numerical stability
        # y coordinate: [radius, 1-radius]
        bounds.append((0.001, 0.999))
        # radius: [0.001, 0.5] (reasonable upper bound)
        bounds.append((0.001, 0.499))
    
    # Get constraints
    constraints = get_constraints()
    
    # Flatten initial circles array for optimization
    x0 = []
    for circle in circles:
        x0.extend(circle)
    
    # Optimization options
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=1e-6
        )
        
        if result.success:
            # Extract final solution
            circles_opt = []
            for i in range(n):
                x = result.x[i*3]
                y = result.x[i*3+1]
                r = result.x[i*3+2]
                circles_opt.append([x, y, r])
            return np.array(circles_opt)
        else:
            # Return initial solution if optimization fails
            return circles
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
