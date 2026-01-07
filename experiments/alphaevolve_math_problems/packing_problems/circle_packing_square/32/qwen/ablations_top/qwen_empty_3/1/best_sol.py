# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with robust optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles using a hexagonal grid pattern for good initial placement
    def initialize_hexagonal_grid():
        circles = []
        # Create a hexagonal grid pattern
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Hexagonal spacing
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Adjust for hexagonal packing
        hex_radius = min(spacing_x, spacing_y) * 0.4
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Offset every other row
                x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Ensure we're within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius - small enough to fit
                    r = min(hex_radius, x, 1-x, y, 1-y)
                    if r > 0:
                        circles.append([x, y, r])
                        count += 1
            if count >= n:
                break
                
        # Fill remaining slots with random positions
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = min(0.05, x, 1-x, y, 1-y)
            if r > 0:
                circles.append([x, y, r])
                
        return np.array(circles)
    
    # Constraint functions with better numerical handling and reduced constraint count
    def get_constraints():
        cons = []
        
        # Boundary constraints: radius must be such that circle fits entirely
        def boundary_constraint(i):
            def constraint(x):
                x_pos, y_pos, r = x[3*i], x[3*i+1], x[3*i+2]
                # Add a small safety margin to prevent numerical issues
                safe_r = max(0.001, r)
                return min(safe_r, x_pos - safe_r, 1 - x_pos - safe_r, y_pos - safe_r, 1 - y_pos - safe_r)
            return {'type': 'ineq', 'fun': constraint}
        
        # Non-overlap constraints with improved numerical handling
        def overlap_constraint(i, j):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                # Add small epsilon to avoid numerical precision issues
                dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                sum_radii_sq = (r_i + r_j)**2
                return dist_sq - sum_radii_sq
            return {'type': 'ineq', 'fun': constraint}
        
        # Add boundary constraints for all circles
        for i in range(n):
            cons.append(boundary_constraint(i))
            
        # Add non-overlap constraints for pairs - limit to reduce computational burden
        # Only check some pairs to reduce constraints but still maintain quality
        for i in range(n):
            for j in range(i+1, min(i+10, n)):  # Limit to nearby neighbors to reduce constraints
                cons.append(overlap_constraint(i, j))
        
        return cons
    
    # Enhanced optimization approach with better fallbacks
    def optimize_circles(initial_circles):
        # Flatten initial circles for optimization
        x0 = initial_circles.flatten()
        
        # Objective function: negative sum of radii (we want to maximize sum)
        def objective(x):
            total_radii = 0
            for i in range(n):
                total_radii += x[3*i+2]  # radius is third component
            return -total_radii  # Negative because we minimize
        
        # Try multiple optimization approaches for better results
        bounds = []
        for i in range(n):
            bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])
            
        # First try with constraints
        try:
            cons = get_constraints()
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                            options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6})
            
            if result.success:
                optimized_circles = result.x.reshape((n, 3))
                return optimized_circles
        except Exception as e:
            pass
            
        # If that fails, try without constraints but with stricter bounds
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6})
            
            if result.success:
                optimized_circles = result.x.reshape((n, 3))
                return optimized_circles
        except Exception as e:
            pass
            
        # Return initial circles if all optimization attempts fail
        return initial_circles
    
    # Start with hexagonal grid initialization
    initial_circles = initialize_hexagonal_grid()
    
    # Refine using optimization
    final_circles = optimize_circles(initial_circles)
    
    # Final validation and adjustment with more careful handling
    def validate_and_adjust(circles):
        adjusted = circles.copy()
        # Ensure all constraints are met and radii are reasonable
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Adjust radius to satisfy boundary constraints
            max_r = min(x, 1-x, y, 1-y)
            if r > max_r:
                adjusted[i, 2] = max_r * 0.99  # Slightly less to ensure feasibility
            
            # Make sure circles don't overlap after adjustment
            for j in range(len(circles)):
                if i != j:
                    x1, y1, r1 = adjusted[i]
                    x2, y2, r2 = adjusted[j]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist_sq = (r1 + r2)**2
                    
                    if dist_sq < min_dist_sq:
                        # Reduce radius to prevent overlap - more conservative approach
                        overlap_amount = np.sqrt(min_dist_sq) - np.sqrt(dist_sq)
                        reduction = min(overlap_amount * 0.3, r1 * 0.05)  # Smaller reduction factor
                        adjusted[i, 2] = max(0.001, r1 - reduction)
        
        return adjusted
    
    validated_circles = validate_and_adjust(final_circles)
    
    return validated_circles


# EVOLVE-BLOCK-END
