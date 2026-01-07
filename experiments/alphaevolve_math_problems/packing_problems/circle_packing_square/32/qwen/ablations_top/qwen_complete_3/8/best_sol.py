# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining grid initialization and constrained optimization.
    """
    n = 32
    
    # Initialize using a hexagonal grid pattern for good initial distribution
    def initialize_hexagonal_grid():
        # Create a hexagonal grid pattern
        circles = []
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Hexagonal packing with offset rows
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                if i % 2 == 0:
                    x = (j + 0.5) * spacing_x
                    y = (i + 0.5) * spacing_y
                else:
                    x = (j + 1.0) * spacing_x
                    y = (i + 0.5) * spacing_y
                
                # Ensure we're within bounds and add some randomness
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius estimate based on available space
                    min_dist = min(x, 1-x, y, 1-y)
                    max_radius = min_dist / 2.0
                    if max_radius > 0:
                        circles.append([x, y, max_radius])
        
        # Fill remaining circles with random placement near boundaries
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            min_dist = min(x, 1-x, y, 1-y)
            max_radius = min_dist / 2.0
            if max_radius > 0:
                circles.append([x, y, max_radius])
                
        return np.array(circles[:n])
    
    # Get initial configuration
    initial_config = initialize_hexagonal_grid()
    
    # Flatten initial configuration for optimization
    initial_vars = initial_config.flatten()
    
    # Define constraint functions for optimization - more stable version
    def boundary_constraints(vars):
        """Ensure all circles are within the unit square"""
        cons = []
        for i in range(n):
            x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
            # Circle must be contained: x >= r, y >= r, 1-x >= r, 1-y >= r
            cons.extend([
                x - r,      # x >= r
                y - r,      # y >= r  
                1 - x - r,  # 1-x >= r
                1 - y - r   # 1-y >= r
            ])
        return np.array(cons)
    
    def overlap_constraints(vars):
        """Ensure no overlaps between circles"""
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                # Distance between centers must be >= sum of radii
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                min_dist_sq = (r1+r2)**2
                # Constraint: dist_sq >= min_dist_sq (so we want: dist_sq - min_dist_sq >= 0)
                # Add small epsilon to prevent numerical issues
                cons.append(dist_sq - min_dist_sq)
        return np.array(cons)
    
    # Optimization objective: maximize sum of radii (minimize negative sum)
    def objective(vars):
        return -sum(vars[2::3])  # Negative because we minimize
    
    # Set up bounds: x in [0.001, 0.999], y in [0.001, 0.999], r in [0.001, 0.499]
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Define constraints
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    # Run optimization using SLSQP which handles constraints well
    try:
        result = minimize(objective, initial_vars, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            final_vars = result.x
        else:
            # Fallback to initial configuration if optimization fails
            final_vars = initial_vars
    except Exception as e:
        # If optimization fails, fall back to initial configuration
        final_vars = initial_vars
    
    # Convert back to circles array
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [final_vars[3*i], final_vars[3*i+1], final_vars[3*i+2]]
    
    # Final validation and adjustment to ensure all constraints are met
    def validate_and_adjust(circles_array):
        # Ensure all circles respect boundary constraints
        for i in range(n):
            x, y, r = circles_array[i]
            # Adjust radius if needed to respect boundaries
            r_new = min(r, x, 1-x, y, 1-y)
            if r_new < r:
                circles_array[i] = [x, y, r_new]
        
        # Ensure no overlaps by adjusting positions/radii - more careful approach
        max_iter = 30
        for _ in range(max_iter):
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles_array[i]
                    x2, y2, r2 = circles_array[j]
                    
                    dist_sq = (x1-x2)**2 + (y1-y2)**2
                    min_dist_sq = (r1+r2)**2
                    
                    if dist_sq < min_dist_sq:
                        # Overlap detected - reduce radii or adjust positions
                        overlap = min_dist_sq - dist_sq
                        # Reduce both radii proportionally with more careful handling
                        reduction = min(overlap / 2.0, min(r1, r2) * 0.5)
                        if reduction > 0.0001:  # Only make significant changes
                            circles_array[i][2] -= reduction
                            circles_array[j][2] -= reduction
                            changed = True
                            
            if not changed:
                break
                
        return circles_array
    
    circles = validate_and_adjust(circles)
    
    # Try multiple random restarts to find better solutions
    best_result = circles
    best_sum = np.sum(circles[:, 2])
    
    # Try 10 different random restarts (more extensive search)
    for restart in range(10):
        # Slightly perturb the initial configuration
        np.random.seed(42 + restart)  # Different seed for each restart
        
        # Create slightly different initial configuration
        restart_config = initialize_hexagonal_grid()
        
        # Perturb slightly with more controlled variation
        for i in range(n):
            if np.random.random() < 0.3:  # 30% chance to perturb
                restart_config[i][0] += np.random.normal(0, 0.005)
                restart_config[i][1] += np.random.normal(0, 0.005)
                restart_config[i][0] = np.clip(restart_config[i][0], 0.001, 0.999)
                restart_config[i][1] = np.clip(restart_config[i][1], 0.001, 0.999)
        
        restart_vars = restart_config.flatten()
        
        try:
            result_restart = minimize(objective, restart_vars, method='SLSQP', bounds=bounds, 
                                    constraints=constraints, options={'maxiter': 500, 'ftol': 1e-6})
            
            if result_restart.success:
                restart_circles = result_restart.x.reshape(-1, 3)
                restart_circles = validate_and_adjust(restart_circles)
                restart_sum = np.sum(restart_circles[:, 2])
                
                if restart_sum > best_sum:
                    best_sum = restart_sum
                    best_result = restart_circles
        except:
            continue
    
    return best_result


# EVOLVE-BLOCK-END
