# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles using a grid-based approach for good starting configuration
    circles = np.zeros((n, 3))
    
    # Create a grid layout as initial guess
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            # Initial radius - small enough to fit in grid cell
            r = min(spacing_x, spacing_y) * 0.4
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Ensure we have exactly 32 circles
    if idx < n:
        # Fill remaining positions with random valid placements
        for i in range(idx, n):
            # Random placement within valid bounds
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.1)
            circles[i] = [x, y, r]
    
    # Define objective function to maximize sum of radii
    def objective(params):
        # params contains [x1, y1, r1, x2, y2, r2, ...]
        total_radius = 0
        for i in range(0, len(params), 3):
            total_radius += params[i+2]  # Add radius component
        return -total_radius  # Negative because we minimize
    
    # Define constraints
    def containment_constraint(params):
        # Check that all circles are within the unit square
        constraints = []
        for i in range(0, len(params), 3):
            x, y, r = params[i], params[i+1], params[i+2]
            # Circle must be contained in unit square
            constraints.append(x - r)  # x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
        return np.array(constraints)
    
    def overlap_constraint(params):
        # Check that no two circles overlap
        constraints = []
        # Convert params back to circles array for easier processing
        circles_array = []
        for i in range(0, len(params), 3):
            circles_array.append([params[i], params[i+1], params[i+2]])
        
        # For each pair of circles, check if they overlap
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                # Distance between centers
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # Minimum allowed distance (sum of radii)
                min_dist_sq = (r1 + r2)**2
                # We want dist >= r1 + r2, so we enforce dist_sq >= min_dist_sq
                # This means: dist_sq - min_dist_sq >= 0
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)
    
    # Combine all constraints
    def combined_constraints(params):
        # Return all constraints concatenated
        cont = containment_constraint(params)
        overlap = overlap_constraint(params)
        return np.concatenate([cont, overlap])
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # Bounds for x coordinates: [r, 1-r]
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.4))    # r (reasonable upper bound)
    
    # Optimization using SLSQP method which handles constraints well
    try:
        # Flatten the initial configuration
        initial_params = circles.flatten()
        
        # Define constraint dictionaries
        containment_cons = {
            'type': 'ineq',
            'fun': lambda x: containment_constraint(x)
        }
        
        overlap_cons = {
            'type': 'ineq', 
            'fun': lambda x: overlap_constraint(x)
        }
        
        # Run optimization
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=[containment_cons, overlap_cons],
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        # Extract final solution
        if result.success:
            final_params = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
        else:
            # If optimization fails, use the initial configuration
            pass
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        pass
    
    return circles


# EVOLVE-BLOCK-END
