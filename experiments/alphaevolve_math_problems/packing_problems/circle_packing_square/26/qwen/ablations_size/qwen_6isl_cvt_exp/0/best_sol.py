# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining global optimization with local refinement.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a hexagonal close packing approximation for better starting configuration
    def initialize_hexagonal():
        # Create a hexagonal arrangement that's known to be efficient for circle packing
        circles = []
        
        # Hexagonal packing parameters
        sqrt3 = math.sqrt(3)
        rows = int(math.ceil(math.sqrt(n) * 2 / sqrt3))
        cols = int(math.ceil(n / rows))
        
        # Calculate spacing
        spacing_x = 1.0 / cols
        spacing_y = sqrt3 / (2 * cols)
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Hexagonal offset
                x_offset = (j + 0.5) * spacing_x
                y_offset = (i + (j % 2) * 0.5) * spacing_y
                
                # Ensure within bounds
                x = max(0.05, min(0.95, x_offset))
                y = max(0.05, min(0.95, y_offset))
                
                # Initial radius - smaller than spacing to allow for optimization
                r = min(spacing_x, spacing_y) * 0.3
                circles.append([x, y, r])
                count += 1
            if count >= n:
                break
                
        return np.array(circles[:n])
    
    # Alternative initialization: grid-based
    def initialize_grid():
        # Create a grid pattern that roughly fits 26 circles
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        
        # Adjust grid spacing to fit within unit square with margin
        margin = 0.05
        cell_width = (1 - 2*margin) / cols
        cell_height = (1 - 2*margin) / rows
        
        circles = []
        count = 0
        
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x = margin + (j + 0.5) * cell_width
                y = margin + (i + 0.5) * cell_height
                # Initial radius - small enough to fit in cell
                r = min(cell_width, cell_height) * 0.4
                circles.append([x, y, r])
                count += 1
            if count >= n:
                break
                
        return np.array(circles)
    
    # Flatten for optimization (x1, y1, r1, x2, y2, r2, ...)
    def flatten_circles(circles):
        return np.concatenate([circles.flatten()])
    
    # Unflatten from optimization variables back to circles
    def unflatten_circles(vars):
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [vars[3*i], vars[3*i+1], vars[3*i+2]]
        return circles
    
    # Constraint functions with improved numerical stability
    def get_constraints():
        cons = []
        
        # Boundary constraints: radius <= x <= 1-radius, radius <= y <= 1-radius
        for i in range(n):
            def boundary_constraint(vars, idx=i):
                x, y, r = vars[3*idx], vars[3*idx+1], vars[3*idx+2]
                # Return minimum of all boundary constraints for better numerical stability
                constraints = [
                    r,                    # r >= 0
                    1 - r,                # 1 - r >= 0
                    x - r,                # x - r >= 0 (left)
                    1 - x - r,            # 1 - x - r >= 0 (right)
                    y - r,                # y - r >= 0 (bottom)
                    1 - y - r             # 1 - y - r >= 0 (top)
                ]
                return min(constraints)
            cons.append({'type': 'ineq', 'fun': boundary_constraint})
            
        # Circle-to-circle distance constraints
        for i in range(n):
            for j in range(i+1, n):
                def overlap_constraint(vars, idx1=i, idx2=j):
                    x1, y1, r1 = vars[3*idx1], vars[3*idx1+1], vars[3*idx1+2]
                    x2, y2, r2 = vars[3*idx2], vars[3*idx2+1], vars[3*idx2+2]
                    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    # Distance should be >= r1 + r2 for no overlap (positive when valid)
                    return dist - (r1 + r2)
                cons.append({'type': 'ineq', 'fun': overlap_constraint})
                
        return cons
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(vars):
        total_radius = sum(vars[3*i+2] for i in range(n))
        return -total_radius
    
    # Multi-start optimization with different strategies
    best_result = None
    best_sum = -float('inf')
    
    # Strategy 1: Differential Evolution (global search) - very effective for this problem
    try:
        # Use differential evolution for global search first
        constraints = get_constraints()
        
        # Set up bounds for differential evolution
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Run differential evolution with multiple restarts - increased iterations for better convergence
        de_results = []
        for restart in range(7):  # More restarts for even better chance of finding global optimum
            np.random.seed(restart)
            result = differential_evolution(
                objective,
                bounds,
                constraints=constraints,
                maxiter=600,  # Even more iterations
                popsize=25,   # Even larger population
                seed=restart,
                disp=False
            )
            if result.success:
                de_results.append(result)
        
        # Evaluate all DE results
        for result in de_results:
            current_sum = -result.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = result
                
    except Exception as e:
        pass  # Fall back to other methods if DE fails
    
    # Strategy 2: Local optimization with multiple starting points (if DE didn't work well)
    if best_result is None:
        # Try several different initial configurations
        initial_configs = [initialize_hexagonal(), initialize_grid()]
        
        # Add some random perturbations to create diversity
        for i in range(4):  # Even more random starting points
            np.random.seed(i)
            random_config = initialize_hexagonal().copy()
            for j in range(n):
                random_config[j][0] += np.random.normal(0, 0.012)  # Slightly smaller perturbation
                random_config[j][1] += np.random.normal(0, 0.012)
                random_config[j][2] += np.random.normal(0, 0.006)
            initial_configs.append(random_config)
        
        # Run local optimization from each starting point
        for i, config in enumerate(initial_configs):
            try:
                initial_vars = flatten_circles(config)
                constraints = get_constraints()
                
                result = minimize(
                    objective,
                    initial_vars,
                    method='SLSQP',
                    bounds=[(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n,
                    constraints=constraints,
                    options={'maxiter': 600, 'ftol': 1e-9, 'eps': 1e-9}  # Tighter tolerances
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
                        
            except Exception as e:
                continue  # Skip failed optimizations
    
    # If no good result found, return the best initial configuration
    if best_result is None:
        # Use hexagonal initialization and do one final optimization with trust-constr
        initial_circles = initialize_hexagonal()
        initial_vars = flatten_circles(initial_circles)
        constraints = get_constraints()
        
        try:
            result = minimize(
                objective,
                initial_vars,
                method='trust-constr',  # Use trust-constr for potentially better results
                bounds=[(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n,
                constraints=constraints,
                options={'maxiter': 600, 'verbose': 0}
            )
            
            if result.success:
                best_result = result
                best_sum = -result.fun
            else:
                best_result = None
                
        except Exception as e:
            best_result = None
    
    # Final processing
    if best_result is not None:
        final_circles = unflatten_circles(best_result.x)
    else:
        # If all optimization failed, return the hexagonal initialization
        final_circles = initialize_hexagonal()
    
    # Ensure final result meets constraints properly
    def validate_and_correct(circles):
        # Make sure all circles are valid
        corrected = circles.copy()
        for i in range(n):
            x, y, r = corrected[i]
            # Ensure radius is valid
            r = max(0.001, min(0.499, r))
            # Ensure position is valid
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            corrected[i] = [x, y, r]
        return corrected
    
    final_circles = validate_and_correct(final_circles)
    
    return final_circles


# EVOLVE-BLOCK-END
