# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
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
    
    # Initialize with a better hexagonal grid pattern
    def initialize_better_hexagonal_placement():
        circles = np.zeros((n, 3))
        
        # Create a more systematic hexagonal grid pattern
        rows = 6
        cols = 6
        
        # Adjust spacing to fit within unit square
        spacing_x = 0.85 / cols
        spacing_y = 0.85 / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = 0.075 + (j + 1) * spacing_x
                y = 0.075 + (i + 1) * spacing_y
                
                # Slightly stagger odd rows for hexagonal pattern
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                # Initial radius - start with smaller radius to allow for optimization
                max_radius = min(x, 1-x, y, 1-y) * 0.35
                circles[idx] = [x, y, max_radius]
                idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with carefully placed circles
        random.seed(42)
        for i in range(idx, n):
            # Try to place near existing circles or in free areas
            if i < 20:  # Use more systematic placement for first few
                x = 0.1 + (i % 5) * 0.18
                y = 0.1 + (i // 5) * 0.18
                max_radius = min(x, 1-x, y, 1-y) * 0.3
            else:  # Random placement with good bounds
                x = random.uniform(0.1, 0.9)
                y = random.uniform(0.1, 0.9)
                max_radius = min(x, 1-x, y, 1-y) * 0.25
            
            circles[i] = [x, y, max_radius]
        
        return circles

    # Create constraint functions for scipy optimization
    def get_constraints():
        cons = []
        
        # Boundary constraints for each circle
        for i in range(n):
            # x >= r and x <= 1-r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})  # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})  # 1 - x - r >= 0
            # y >= r and y <= 1-r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})  # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})  # 1 - y - r >= 0
        
        # Non-overlap constraints (more efficient implementation)
        def overlap_constraint(i, j):
            def constraint(x):
                x1, y1, r1 = x[3*i], x[3*i+1], x[3*i+2]
                x2, y2, r2 = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                dist = np.sqrt(dist_sq)
                # Return positive value when constraint is satisfied (no overlap)
                return dist - (r1 + r2)
            return constraint
        
        # Add all non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        return cons

    # Multi-start optimization to avoid local minima
    def multi_start_optimization(initial_circles):
        best_circles = initial_circles.copy()
        best_sum = np.sum(initial_circles[:, 2])
        
        # Try multiple optimization runs with different initializations
        # Use a more extensive search to find better solutions
        for run in range(5):  # Increase number of runs
            # Slight perturbation of the initial configuration
            circles = initial_circles.copy()
            
            # Add small random perturbations to positions and radii
            random.seed(42 + run)
            for i in range(n):
                circles[i, 0] += random.uniform(-0.005, 0.005)  # Even smaller perturbations
                circles[i, 1] += random.uniform(-0.005, 0.005)
                circles[i, 2] += random.uniform(-0.002, 0.002)
            
            # Ensure they're still valid
            for i in range(n):
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
                circles[i, 2] = max(0.001, min(0.499, circles[i, 2]))
            
            try:
                # Prepare optimization parameters
                initial_params = circles.flatten()
                
                # Define bounds for each parameter (x, y, r) for each circle
                bounds = []
                for i in range(n):
                    bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
                
                # Get constraint functions
                constraints = get_constraints()
                
                # Objective function (negative because we want to maximize sum of radii)
                def objective(params):
                    # Extract radii
                    radii = params[2::3]  # Every third element starting from index 2
                    # Return negative sum (since minimize minimizes)
                    return -np.sum(radii)
                
                # Try multiple optimization methods for better results
                methods = ['SLSQP', 'trust-constr']
                best_result = None
                best_result_sum = -float('inf')
                
                for method in methods:
                    try:
                        result = minimize(
                            objective,
                            initial_params,
                            method=method,
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 750, 'ftol': 1e-7, 'eps': 1e-7},  # More iterations, tighter tolerance
                            tol=1e-7
                        )
                        
                        if result.success:
                            final_params = result.x
                            optimized_circles = final_params.reshape((n, 3))
                            current_sum = np.sum(optimized_circles[:, 2])
                            
                            if current_sum > best_result_sum:
                                best_result_sum = current_sum
                                best_result = result
                    except:
                        continue
                
                if best_result and best_result.success:
                    final_params = best_result.x
                    optimized_circles = final_params.reshape((n, 3))
                    
                    # Check if this is better
                    current_sum = np.sum(optimized_circles[:, 2])
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_circles = optimized_circles
                        
            except Exception:
                # If optimization fails, continue with next run
                continue
        
        return best_circles

    # Main optimization process
    try:
        # Initialize with better pattern
        circles = initialize_better_hexagonal_placement()
        
        # Run multi-start optimization to find better solution
        circles = multi_start_optimization(circles)
            
    except Exception as e:
        # If anything goes wrong, return the initial configuration
        circles = initialize_better_hexagonal_placement()
        pass
    
    # Final validation and cleanup
    for i in range(n):
        # Ensure radius is positive and reasonable
        circles[i, 2] = max(0.001, min(0.499, circles[i, 2]))
        
        # Ensure circle is within bounds
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
        # Ensure radius doesn't exceed boundary constraints
        boundary_radius = min(circles[i, 0], 1 - circles[i, 0], 
                             circles[i, 1], 1 - circles[i, 1])
        circles[i, 2] = min(circles[i, 2], boundary_radius)
    
    return circles


# EVOLVE-BLOCK-END
