# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, local optimization, and multi-start strategy.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
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
        
        # Fill remaining circles if needed with strategic placements
        while len(circles) < n:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            radius = np.random.uniform(0.05, 0.2)
            circles.append([x, y, radius])
            
        return np.array(circles[:n])
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        # params contains [x1,y1,r1,x2,y2,r2,...,x26,y26,r26]
        radii = params[2::3]  # Extract all radii
        return -sum(radii)  # Negative because minimize
    
    # Constraint functions for scipy
    def create_constraints():
        constraints = []
        
        # Containment constraints: x >= r, y >= r, x <= 1-r, y <= 1-r
        for i in range(n):
            # Create a proper closure with explicit binding
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
                # Create a proper closure with explicit binding
                def make_overlap_constraint(i, j):
                    return lambda p: (p[3*j] - p[3*i])**2 + (p[3*j+1] - p[3*i+1])**2 - (p[3*i+2] + p[3*j+2])**2
                
                constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
        
        return constraints
    
    # More robust refinement step
    def refine_initial_solution(initial_circles):
        circles = initial_circles.copy()
        
        # Perform multiple rounds of local optimization
        for round_num in range(100):
            improved = False
            for i in range(n):
                # Try to increase radius for this circle
                current_radius = circles[i][2]
                
                # Compute maximum possible radius for this circle
                max_radius = min(circles[i][0], 1 - circles[i][0], 
                               circles[i][1], 1 - circles[i][1])
                
                # Check overlap with all other circles
                for j in range(n):
                    if i != j:
                        dx = circles[i][0] - circles[j][0]
                        dy = circles[i][1] - circles[j][1]
                        dist_sq = dx*dx + dy*dy
                        
                        if dist_sq > 0:
                            min_dist = np.sqrt(dist_sq)
                            max_allowed = min_dist - circles[j][2]
                            max_radius = min(max_radius, max_allowed)
                
                max_radius = max(0.001, max_radius)
                
                if max_radius > current_radius + 1e-6:
                    # Try to set to maximum
                    circles[i][2] = max_radius
                    improved = True
            
            if not improved:
                break
                
        return circles
    
    # Initialize
    initial_circles = initialize_hexagonal()
    initial_circles = refine_initial_solution(initial_circles)
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
    
    # Try several different initializations with random perturbations
    for attempt in range(20):  # Increase attempts significantly
        # Start with base initialization
        current_initial = initial_circles.copy()
        
        # Add some random perturbation to initial positions and radii
        for i in range(n):
            # Perturb x and y with varying magnitudes
            current_initial[i, 0] = max(0.01, min(0.99, current_initial[i, 0] + np.random.normal(0, 0.03)))
            current_initial[i, 1] = max(0.01, min(0.99, current_initial[i, 1] + np.random.normal(0, 0.03)))
            # Perturb radius slightly with larger variance
            current_initial[i, 2] = max(0.01, min(0.4, current_initial[i, 2] + np.random.normal(0, 0.05)))
        
        current_params = current_initial.flatten()
        
        try:
            # Run optimization with different methods to increase success rate
            methods = ['SLSQP', 'trust-constr']
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
                        options={'maxiter': 2000, 'ftol': 1e-9, 'eps': 1e-9}
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
    
    # Final validation and adjustment - more robust version
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
        while changed and iterations < 15:  # More iterations for better convergence
            changed = False
            for i in range(len(adjusted)):
                x, y, r = adjusted[i]
                # Check for overlaps with all other circles
                for j in range(len(adjusted)):
                    if i != j:
                        x2, y2, r2 = adjusted[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        min_dist_sq = (r + r2)**2
                        if dist_sq < min_dist_sq - 1e-10:
                            # Reduce radius to maintain separation
                            max_radius = np.sqrt(dist_sq) - r2 - 1e-8
                            if max_radius > 0 and max_radius < r:
                                adjusted[i] = [x, y, max_radius]
                                changed = True
            iterations += 1
        
        return adjusted
    
    final_circles = validate_and_adjust(final_circles)
    
    return final_circles


# EVOLVE-BLOCK-END
