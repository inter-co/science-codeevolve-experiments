# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize using hexagonal packing pattern for good starting configuration
    def initialize_hexagonal():
        # Create hexagonal grid that fits in unit square
        # For 26 circles, we'll use approximately 5 rows and 5 columns
        rows = 5
        cols = 5
        if rows * cols < n:
            rows += 1
            cols = math.ceil(n / rows)
        
        # Hexagonal packing parameters
        spacing_x = 1.0 / cols
        spacing_y = spacing_x * math.sqrt(3) / 2
        offset_y = spacing_y / 2
        
        circles = []
        count = 0
        
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                    
                # Offset every other row
                x_offset = (i % 2) * spacing_x / 2
                x = (j + 0.5) * spacing_x + x_offset
                y = (i + 0.5) * spacing_y
                
                # Ensure within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius estimate based on available space
                    min_dist = min(x, 1-x, y, 1-y)
                    radius = min_dist / 2.0
                    circles.append([x, y, radius])
                    count += 1
            
            if count >= n:
                break
        
        # Fill remaining circles if needed
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.1])
            
        return np.array(circles[:n])
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        # params contains [x1,y1,r1,x2,y2,r2,...,x26,y26,r26]
        radii = params[2::3]  # Extract all radii
        return -sum(radii)  # Negative because minimize
    
    # Constraint functions for scipy - using default arguments to avoid lambda binding issues
    def create_constraints():
        constraints = []
        
        # Containment constraints: x >= r, y >= r, x <= 1-r, y <= 1-r
        for i in range(n):
            # Using default arguments in lambda to avoid late binding issues (INSPIRATION 2 approach)
            def containment_x_ge_r(p, i=i):
                return p[3*i] - p[3*i+2]  # x - r >= 0
            
            def containment_y_ge_r(p, i=i):
                return p[3*i+1] - p[3*i+2]  # y - r >= 0
            
            def containment_x_le_1_minus_r(p, i=i):
                return 1 - p[3*i] - p[3*i+2]  # 1 - x - r >= 0
            
            def containment_y_le_1_minus_r(p, i=i):
                return 1 - p[3*i+1] - p[3*i+2]  # 1 - y - r >= 0
            
            constraints.append({'type': 'ineq', 'fun': containment_x_ge_r})
            constraints.append({'type': 'ineq', 'fun': containment_y_ge_r})
            constraints.append({'type': 'ineq', 'fun': containment_x_le_1_minus_r})
            constraints.append({'type': 'ineq', 'fun': containment_y_le_1_minus_r})
        
        # Non-overlap constraints: distance >= r1 + r2
        for i in range(n):
            for j in range(i+1, n):
                # Using default arguments in lambda to avoid late binding issues
                def overlap_constraint(p, i=i, j=j):
                    x1, y1, r1 = p[3*i], p[3*i+1], p[3*i+2]
                    x2, y2, r2 = p[3*j], p[3*j+1], p[3*j+2]
                    dist_sq = (x2 - x1)**2 + (y2 - y1)**2
                    min_dist_sq = (r1 + r2)**2
                    return dist_sq - min_dist_sq
                
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return constraints
    
    # Initialize
    initial_circles = initialize_hexagonal()
    initial_params = initial_circles.flatten()
    
    # Set bounds for optimization (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x, y: [0.001, 0.999] to avoid boundary issues
        # r: [0.001, 0.499] (maximum possible radius for any circle)
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Define constraints
    cons = create_constraints()
    
    # Optimize with multiple starting points to improve chances of finding better solution
    best_result = None
    best_sum = -np.inf
    
    # Try several different initializations with random perturbations (INSPIRATION 2 approach)
    for attempt in range(10):  # Increased from 5 to 10 attempts
        # Start with base initialization
        current_initial = initial_circles.copy()
        
        # Add some random perturbation to initial positions and radii (larger variance)
        for i in range(n):
            # Perturb x and y slightly with larger variance
            current_initial[i, 0] = max(0.01, min(0.99, current_initial[i, 0] + np.random.normal(0, 0.05)))
            current_initial[i, 1] = max(0.01, min(0.99, current_initial[i, 1] + np.random.normal(0, 0.05)))
            # Perturb radius slightly with larger variance
            current_initial[i, 2] = max(0.01, min(0.4, current_initial[i, 2] + np.random.normal(0, 0.03)))
        
        current_params = current_initial.flatten()
        
        try:
            # Run optimization with multiple methods to increase success rate (INSPIRATION 2 approach)
            methods = ['SLSQP', 'trust-constr']  # Try both methods
            method_success = False
            for method in methods:
                if method_success:
                    break
                try:
                    result = minimize(
                        objective,
                        current_params,
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
                    )
                    
                    if result.success:
                        current_sum = -result.fun
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_result = result.x.copy()
                            method_success = True
                        break  # Success with this method
                except Exception:
                    continue
                    
        except Exception:
            continue
    
    # If we have a valid result, use it; otherwise return the initial configuration
    if best_result is not None:
        final_circles = best_result.reshape(-1, 3)
    else:
        final_circles = initial_circles
    
    # Final validation and adjustment - more robust version (INSPIRATION 2 approach)
    def validate_and_adjust(circles):
        # Ensure all circles are valid
        adjusted = circles.copy()
        
        # First pass: fix containment issues
        for i in range(len(adjusted)):
            x, y, r = adjusted[i]
            # Ensure containment by reducing radius if necessary
            r = min(r, x, 1-x, y, 1-y)
            adjusted[i] = [x, y, r]
        
        # Second pass: resolve overlaps by reducing radii
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            for i in range(len(adjusted)):
                x, y, r = adjusted[i]
                # Check for overlaps with all other circles
                for j in range(len(adjusted)):
                    if i != j:
                        x2, y2, r2 = adjusted[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        min_dist_sq = (r + r2)**2
                        if dist_sq < min_dist_sq:
                            # Reduce radius to maintain separation
                            max_radius = math.sqrt(dist_sq) - r2 - 1e-8
                            if max_radius > 0 and max_radius < r:
                                adjusted[i] = [x, y, max_radius]
                                changed = True
            iterations += 1
        
        return adjusted
    
    final_circles = validate_and_adjust(final_circles)
    
    return final_circles


# EVOLVE-BLOCK-END
