# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach: initial hexagonal packing + multi-start constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores 
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initial configuration: hexagonal packing pattern (from inspiration)
    def generate_initial_config():
        # Create a hexagonal lattice pattern similar to successful implementations
        circles = np.zeros((n, 3))
        
        # Better spacing for 26 circles in a roughly 5x5 grid
        rows = 5
        cols = 5
        spacing_x = 0.8 / cols
        spacing_y = 0.8 / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = 0.0 if i % 2 == 0 else spacing_x * 0.5
                x = 0.1 + (j + 0.5) * spacing_x + x_offset
                y = 0.1 + (i + 0.5) * spacing_y
                
                # Ensure circles fit in the unit square
                max_radius = min(x, 1-x, y, 1-y)
                if max_radius > 0.01:
                    # Use a reasonable initial radius
                    radius = min(max_radius * 0.3, 0.15)
                    circles[idx] = [x, y, radius]
                    idx += 1
            if idx >= n:
                break
        
        # Fill remaining circles with better random positions
        for i in range(idx, n):
            # Try to place with better constraint checking
            placed = False
            attempts = 0
            while not placed and attempts < 50:
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                max_radius = min(x, 1-x, y, 1-y)
                if max_radius > 0.01:
                    # Quick overlap check with few existing circles
                    radius = min(max_radius * 0.2, 0.15)
                    overlap = False
                    
                    # Check against first few circles for speed
                    for k in range(min(i, 10)):
                        cx, cy, cr = circles[k]
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                        if dist < radius + cr + 1e-8:
                            overlap = True
                            break
                    
                    if not overlap:
                        circles[i] = [x, y, radius]
                        placed = True
                attempts += 1
            
            if not placed:
                # Fallback to simple positioning
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                max_radius = min(x, 1-x, y, 1-y)
                radius = max_radius * 0.1
                circles[i] = [x, y, radius]
            
        return circles
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Sum of all radii (indices 2, 5, 8, ...)

    # Multi-start optimization with different seeds (from inspiration)
    best_result = None
    best_sum = -np.inf
    
    # Try multiple random seeds to avoid local optima
    seeds = [42, 123, 456, 789, 999, 1001, 2023, 3030]
    
    for seed in seeds:
        np.random.seed(seed)
        
        try:
            # Generate initial configuration
            initial_circles = generate_initial_config()
            initial_flat = initial_circles.flatten()
            
            # Set bounds for optimization (x, y, r for each circle)
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
            
            # Constraint functions - more robust version from inspiration
            constraints = []
            
            # Containment constraints
            for i in range(n):
                # x >= r
                def containment_x_lb(x, i=i):
                    return x[3*i] - x[3*i+2]
                constraints.append({'type': 'ineq', 'fun': containment_x_lb})
                
                # 1-x >= r
                def containment_x_ub(x, i=i):
                    return 1 - x[3*i] - x[3*i+2]
                constraints.append({'type': 'ineq', 'fun': containment_x_ub})
                
                # y >= r
                def containment_y_lb(x, i=i):
                    return x[3*i+1] - x[3*i+2]
                constraints.append({'type': 'ineq', 'fun': containment_y_lb})
                
                # 1-y >= r
                def containment_y_ub(x, i=i):
                    return 1 - x[3*i+1] - x[3*i+2]
                constraints.append({'type': 'ineq', 'fun': containment_y_ub})
            
            # Overlap constraints - from inspiration, with improved handling
            for i in range(n):
                for j in range(i+1, n):
                    def overlap_constraint(x, i=i, j=j):
                        dx = x[3*i] - x[3*j]
                        dy = x[3*i+1] - x[3*j+1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        return distance - x[3*i+2] - x[3*j+2] - 1e-10
                    constraints.append({'type': 'ineq', 'fun': overlap_constraint})
            
            # Optimize using SLSQP method which handles constraints well
            result = minimize(
                objective,
                initial_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-8},
                tol=1e-8
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                # Ensure all circles are within bounds
                for i in range(n):
                    x, y, r = optimized_circles[i]
                    # Clip to valid ranges
                    optimized_circles[i] = [
                        np.clip(x, r, 1-r),
                        np.clip(y, r, 1-r),
                        np.clip(r, 0.001, 0.499)
                    ]
                
                # Check if this is better than our current best
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = optimized_circles.copy()
                    
        except Exception as e:
            continue
    
    # If we found a better solution through multi-start, return it
    if best_result is not None:
        return best_result
    
    # Otherwise, fall back to the initial configuration from the first run
    return generate_initial_config()


# EVOLVE-BLOCK-END
