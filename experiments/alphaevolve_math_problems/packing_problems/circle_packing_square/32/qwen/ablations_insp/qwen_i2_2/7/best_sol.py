# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Initialize with a better hexagonal packing pattern inspired by INSPIRATION 2
    def initialize_hexagonal_layout():
        # Create a more optimized hexagonal grid
        rows = 6
        cols = 6
        spacing_x = 1.0 / (cols + 1)
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        circles = []
        
        # Create hexagonal pattern with offset rows
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = (i % 2) * spacing_x / 2
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                
                # Add small random perturbation to avoid perfect patterns
                x += random.uniform(-spacing_x * 0.1, spacing_x * 0.1)
                y += random.uniform(-spacing_y * 0.1, spacing_y * 0.1)
                
                # Ensure we're within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Initial radius based on spacing
                radius = min(spacing_x, spacing_y) * 0.3
                
                # Adjust radius to fit within bounds
                radius = min(radius, x, 1-x, y, 1-y)
                
                if radius > 0.001:
                    circles.append([x, y, radius])
        
        # Fill remaining positions with random circles
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            radius = min(x, 1-x, y, 1-y) * 0.2
            circles.append([x, y, radius])
            
        return np.array(circles[:n])
    
    # Generate constraints for scipy optimization (inspired by INSPIRATION 2)
    def generate_constraints():
        cons = []
        
        # Boundary constraints for each circle
        for i in range(n):
            def boundary_constraint(i):
                def func(x):
                    x_pos, y_pos, r = x[3*i], x[3*i+1], x[3*i+2]
                    # Ensure circle fits entirely in the unit square
                    return min(x_pos - r, 1 - x_pos - r, y_pos - r, 1 - y_pos - r)
                return func
            
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Non-overlap constraints for all pairs
        for i in range(n):
            for j in range(i+1, n):
                def overlap_constraint(i, j):
                    def func(x):
                        x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                        x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                        # Distance squared between centers
                        dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                        # Minimum distance squared for non-overlap
                        min_dist_sq = (r_i + r_j)**2
                        return dist_sq - min_dist_sq
                    return func
                
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        return cons
    
    # Objective function to maximize sum of radii (negative for minimization)
    def objective(x):
        return -np.sum(x[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Initialize circles
    circles = initialize_hexagonal_layout()
    
    # Prepare initial guess and bounds
    x0 = circles.flatten()
    
    # Set up bounds: (x, y, r) for each circle
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Generate constraints
    constraints = generate_constraints()
    
    # Optimization parameters
    options = {'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Use SLSQP optimizer which works well with constraints
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            constraints=constraints, 
            bounds=bounds,
            options=options
        )
        
        # Extract final solution
        final_circles = result.x.reshape(-1, 3)
        
        # Apply local refinement to improve the solution
        # This is inspired by INSPIRATION 2's local search approach
        def local_refinement(refined_circles):
            max_iterations = 100
            for _ in range(max_iterations):
                improved = False
                for i in range(n):
                    best_params = refined_circles[i].copy()
                    best_sum = np.sum(refined_circles[:, 2])
                    
                    # Try different small perturbations
                    steps = [0.001, 0.005, 0.01]
                    for step in steps:
                        for dx in [-step, 0, step]:
                            for dy in [-step, 0, step]:
                                for dr in [-step/2, 0, step/2]:
                                    if abs(dx) + abs(dy) + abs(dr) == 0:
                                        continue
                                    
                                    new_x = refined_circles[i][0] + dx
                                    new_y = refined_circles[i][1] + dy
                                    new_r = refined_circles[i][2] + dr
                                    
                                    # Check bounds
                                    if (0 <= new_x - new_r and new_x + new_r <= 1 and 
                                        0 <= new_y - new_r and new_y + new_r <= 1 and 
                                        new_r > 0):
                                        
                                        # Check overlaps with others
                                        valid = True
                                        for j in range(n):
                                            if i != j:
                                                dist = math.sqrt((new_x - refined_circles[j][0])**2 + 
                                                                (new_y - refined_circles[j][1])**2)
                                                if dist < (new_r + refined_circles[j][2]):
                                                    valid = False
                                                    break
                                        
                                        if valid:
                                            # Temporarily update
                                            old_x, old_y, old_r = refined_circles[i]
                                            refined_circles[i][0], refined_circles[i][1], refined_circles[i][2] = new_x, new_y, new_r
                                            
                                            # Check if this improves the total sum
                                            new_sum = np.sum(refined_circles[:, 2])
                                            if new_sum > best_sum:
                                                best_sum = new_sum
                                                best_params = refined_circles[i].copy()
                                                improved = True
                                            else:
                                                # Revert
                                                refined_circles[i][0], refined_circles[i][1], refined_circles[i][2] = old_x, old_y, old_r
                                
                # Apply best improvement found
                if not improved:
                    break
                    
            return refined_circles
        
        # Apply local refinement
        final_circles = local_refinement(final_circles)
        
        return final_circles
        
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        print(f"Optimization failed: {e}")
        return circles


# EVOLVE-BLOCK-END
