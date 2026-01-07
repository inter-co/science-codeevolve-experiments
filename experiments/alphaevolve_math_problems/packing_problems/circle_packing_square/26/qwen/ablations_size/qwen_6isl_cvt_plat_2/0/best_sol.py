# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

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
    
    # Alternative initialization - more optimized hexagonal pattern
    def initialize_optimized_hexagonal():
        # Try a more optimized hexagonal arrangement
        rows = 6
        cols = 5
        
        spacing_x = 0.9 / cols  # Leave 0.05 margin on each side
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        circles = []
        count = 0
        
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                    
                # Offset every other row
                x_offset = (i % 2) * spacing_x / 2
                x = (j + 0.5) * spacing_x + 0.05 + x_offset
                y = (i + 0.5) * spacing_y + 0.05
                
                # Ensure within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius estimate
                    min_dist = min(x, 1-x, y, 1-y)
                    radius = min(0.1, min_dist / 2.0)
                    circles.append([x, y, radius])
                    count += 1
            
            if count >= n:
                break
        
        # Fill remaining circles if needed
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.05])
            
        return np.array(circles[:n])
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        # params contains [x1,y1,r1,x2,y2,r2,...,x26,y26,r26]
        radii = params[2::3]  # Extract all radii
        return -sum(radii)  # Negative because minimize
    
    # Constraint functions for scipy - using proper closures to avoid late binding issues
    def create_constraints():
        constraints = []
        
        # Containment constraints: x >= r, y >= r, x <= 1-r, y <= 1-r
        for i in range(n):
            # Create proper closures with explicit binding
            def make_containment_x_ge_r(i):
                return lambda p: p[3*i] - p[3*i+2]
            
            def make_containment_y_ge_r(i):
                return lambda p: p[3*i+1] - p[3*i+2]
            
            def make_containment_x_le_1_minus_r(i):
                return lambda p: 1 - p[3*i] - p[3*i+2]
            
            def make_containment_y_le_1_minus_r(i):
                return lambda p: 1 - p[3*i+1] - p[3*i+2]
            
            constraints.append({'type': 'ineq', 'fun': make_containment_x_ge_r(i)})
            constraints.append({'type': 'ineq', 'fun': make_containment_y_ge_r(i)})
            constraints.append({'type': 'ineq', 'fun': make_containment_x_le_1_minus_r(i)})
            constraints.append({'type': 'ineq', 'fun': make_containment_y_le_1_minus_r(i)})
        
        # Non-overlap constraints: distance >= r1 + r2
        for i in range(n):
            for j in range(i+1, n):
                # Create proper closure with explicit binding
                def make_overlap_constraint(i, j):
                    return lambda p: (p[3*j] - p[3*i])**2 + (p[3*j+1] - p[3*i+1])**2 - (p[3*i+2] + p[3*j+2])**2
                
                constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
        
        return constraints
    
    # Improved validation and adjustment function
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
    
    # Multi-start optimization with different strategies
    best_result = None
    best_sum = -np.inf
    
    # Try multiple initialization strategies
    initialization_strategies = [
        ("standard_hex", initialize_hexagonal),
        ("optimized_hex", initialize_optimized_hexagonal)
    ]
    
    for strategy_name, initializer in initialization_strategies:
        # Try several random perturbations for each strategy
        for perturbation in range(10):
            try:
                initial_circles = initializer()
                
                # Add small random perturbations
                if perturbation > 0:
                    for i in range(n):
                        initial_circles[i, 0] = max(0.01, min(0.99, 
                            initial_circles[i, 0] + np.random.normal(0, 0.03)))
                        initial_circles[i, 1] = max(0.01, min(0.99, 
                            initial_circles[i, 1] + np.random.normal(0, 0.03)))
                        initial_circles[i, 2] = max(0.005, min(0.3, 
                            initial_circles[i, 2] + np.random.normal(0, 0.02)))
                
                initial_params = initial_circles.flatten()
                
                # Set bounds for optimization (x, y, r) for each circle
                bounds = []
                for i in range(n):
                    # x, y: [0.001, 0.999] to avoid boundary issues
                    # r: [0.001, 0.499] (maximum possible radius for any circle)
                    bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                
                # Define constraints
                cons = create_constraints()
                
                # Run optimization with different methods to increase success rate
                methods = ['SLSQP', 'trust-constr']
                method_success = False
                for method in methods:
                    if method_success:
                        break
                    try:
                        result = minimize(
                            objective,
                            initial_params,
                            method=method,
                            bounds=bounds,
                            constraints=cons,
                            options={'maxiter': 2000, 'ftol': 1e-6, 'eps': 1e-6}
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
        final_circles = initialize_hexagonal()
    
    # Final validation and adjustment
    final_circles = validate_and_adjust(final_circles)
    
    # Additional local optimization step
    def local_optimization_step(circles):
        # Try to improve each circle's radius individually
        improved = True
        iterations = 0
        while improved and iterations < 5:
            improved = False
            for i in range(len(circles)):
                x, y, r = circles[i]
                
                # Calculate maximum possible radius
                max_radius = min(x, 1-x, y, 1-y)
                
                # Check overlap with all others
                for j in range(len(circles)):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dist = math.sqrt((x - x2)**2 + (y - y2)**2)
                        max_radius = min(max_radius, dist - r2 - 1e-8)
                
                # If we can increase the radius, do so
                if max_radius > r + 1e-8:
                    circles[i] = [x, y, max_radius]
                    improved = True
            
            iterations += 1
        return circles
    
    final_circles = local_optimization_step(final_circles)
    
    return final_circles


# EVOLVE-BLOCK-END
