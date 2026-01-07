# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using a refined hexagonal grid pattern for better starting configuration
    def initialize_hexagonal():
        # Create a more refined hexagonal grid pattern
        circles = []
        
        # Use a more balanced approach for rows/columns
        rows = 6
        cols = 6
        
        # Grid spacing based on hexagonal packing
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Adjust spacing to allow for proper circle packing with better utilization
        max_radius = min(spacing_x, spacing_y) * 0.45
        
        # Create a more structured hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = (i % 2) * spacing_x * 0.5
                x = (j * spacing_x) + x_offset + max_radius
                y = i * spacing_y + max_radius
                
                # Ensure we're within bounds
                if x <= 1 - max_radius and y <= 1 - max_radius:
                    circles.append([x, y, max_radius])
        
        # Fill remaining positions with strategic random placements
        while len(circles) < n:
            # Place near boundaries to explore more space
            side = np.random.randint(0, 4)  # 0=top, 1=right, 2=bottom, 3=left
            if side == 0:  # top
                x = np.random.uniform(max_radius, 1 - max_radius)
                y = max_radius
            elif side == 1:  # right
                x = 1 - max_radius
                y = np.random.uniform(max_radius, 1 - max_radius)
            elif side == 2:  # bottom
                x = np.random.uniform(max_radius, 1 - max_radius)
                y = 1 - max_radius
            else:  # left
                x = max_radius
                y = np.random.uniform(max_radius, 1 - max_radius)
            circles.append([x, y, max_radius])
            
        return np.array(circles)
    
    # Generate initial configuration
    circles = initialize_hexagonal()
    
    # Define constraint functions for optimization
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must be fully contained
        def boundary_constraint(i):
            def constraint(x):
                x_c, y_c, r = x[3*i], x[3*i+1], x[3*i+2]
                return min(x_c - r, 1 - x_c - r, y_c - r, 1 - y_c - r)
            return constraint
        
        # Non-overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(i, j):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                return dist - (r_i + r_j)
            return constraint
        
        # Add boundary constraints for all circles
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Add non-overlap constraints for all pairs
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        return cons
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i + 2]  # radius is third component
        return -total_radius
    
    # Flatten initial circles for optimization
    x0 = circles.flatten()
    
    # Set bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds  
        bounds.append((0.001, 0.999))
        # r bounds (must be positive and small enough to fit in square)
        bounds.append((0.001, 0.499))
    
    # Get constraints
    constraints = get_constraints()
    
    # Try multiple optimization approaches for better results
    best_result = None
    best_sum = 0
    
    # First try with SLSQP
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            # Extract optimized solution
            optimized_circles = result.x.reshape(-1, 3)
            current_sum = np.sum(optimized_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = optimized_circles
    except Exception as e:
        pass
    
    # If first attempt didn't work well, try L-BFGS-B with different settings
    if best_result is None:
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6})
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = optimized_circles
        except Exception as e:
            pass
    
    # If still no good result, return initial configuration
    if best_result is None:
        return circles
    
    # Apply additional refinement to push toward better local optimum
    refined_circles = best_result.copy()
    
    # Run a few iterations of local refinement
    for _ in range(20):
        improved = False
        for i in range(n):
            # Try small adjustments to improve the solution
            current_x, current_y, current_r = refined_circles[i]
            
            # Try to slightly increase radius while maintaining constraints
            new_r = min(current_r * 1.02, 0.499)
            
            # Check if we can increase radius without violating constraints
            valid = True
            for j in range(n):
                if i != j:
                    x_j, y_j, r_j = refined_circles[j]
                    dist_sq = (current_x - x_j)**2 + (current_y - y_j)**2
                    if dist_sq < (new_r + r_j)**2:
                        valid = False
                        break
            
            # If valid and within bounds, update
            if valid:
                if (new_r <= current_x <= 1-new_r and 
                    new_r <= current_y <= 1-new_r):
                    refined_circles[i] = [current_x, current_y, new_r]
                    improved = True
        
        # If no improvement was made, stop early
        if not improved:
            break
    
    return refined_circles


# EVOLVE-BLOCK-END
