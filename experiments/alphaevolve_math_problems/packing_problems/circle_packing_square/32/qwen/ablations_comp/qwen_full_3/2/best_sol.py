# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a structured grid pattern as starting point
    def initialize_structured_grid():
        circles = np.zeros((n, 3))
        
        # Create a grid layout for initial placement
        grid_size = int(math.ceil(math.sqrt(n)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                # Initial radius - small enough to fit in the grid cell
                r = min(spacing_x, spacing_y) * 0.4
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
        
        # Ensure we have exactly 32 circles
        while idx < n:
            # Fill remaining positions with small circles at random locations
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = 0.02
            circles[idx] = [x, y, r]
            idx += 1
            
        return circles
    
    # Physics-inspired refinement to improve solution quality
    def refine_with_physics(circles, max_iterations=200):
        """Apply physics-inspired refinement to resolve overlaps and improve radii."""
        for iteration in range(max_iterations):
            # Simple relaxation step to resolve overlaps
            for i in range(n):
                # For each circle, find neighbors and adjust positions to resolve overlaps
                for j in range(n):
                    if i != j:
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = math.sqrt(dx*dx + dy*dy)
                        
                        # If overlapping, push them apart
                        if dist > 0 and dist < (r1 + r2):
                            # Move them apart
                            move_amount = (r1 + r2 - dist) * 0.5
                            dx_norm = dx / dist
                            dy_norm = dy / dist
                            
                            circles[i, 0] -= dx_norm * move_amount * 0.3
                            circles[i, 1] -= dy_norm * move_amount * 0.3
                            circles[j, 0] += dx_norm * move_amount * 0.3
                            circles[j, 1] += dy_norm * move_amount * 0.3
            
            # Boundary corrections
            for i in range(n):
                x, y, r = circles[i]
                # Keep within bounds
                circles[i, 0] = max(r, min(1-r, x))
                circles[i, 1] = max(r, min(1-r, y))
            
            # Try to increase radii where possible
            for i in range(n):
                # Compute maximum possible radius without violating constraints
                max_radius = min(
                    circles[i, 0], 1 - circles[i, 0],
                    circles[i, 1], 1 - circles[i, 1]
                )
                
                # Check overlap constraints with neighbors
                for j in range(n):
                    if i != j:
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist > 0:
                            max_radius = min(max_radius, dist - r2 - 1e-6)
                
                # Increase radius if beneficial and safe
                max_radius = min(max_radius, 0.49)  # Cap at reasonable value
                if max_radius > circles[i, 2]:
                    circles[i, 2] = min(max_radius, circles[i, 2] + 0.002)
        
        return circles
    
    # More efficient constraint handling using vectorized operations
    def constraint_violations(params):
        # Reshape parameters back into circles array
        pos_radii = params.reshape(-1, 3)
        x = pos_radii[:, 0]
        y = pos_radii[:, 1]
        r = pos_radii[:, 2]
        
        # Containment constraints: each circle must be fully inside the unit square
        containment_violations = []
        for i in range(n):
            containment_violations.extend([
                x[i] - r[i],           # x - r >= 0
                y[i] - r[i],           # y - r >= 0
                1 - x[i] - r[i],       # 1 - x - r >= 0
                1 - y[i] - r[i]        # 1 - y - r >= 0
            ])
        
        # Non-overlap constraints: distance between centers >= sum of radii
        non_overlap_violations = []
        for i in range(n):
            for j in range(i+1, n):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                sum_radii = r[i] + r[j]
                # Violation is negative when circles overlap
                non_overlap_violations.append(dist_sq - sum_radii**2)
        
        return np.array(containment_violations + non_overlap_violations)
    
    # Objective function to maximize sum of radii
    def objective(params):
        # Reshape parameters back into circles array
        pos_radii = params.reshape(-1, 3)
        radii = pos_radii[:, 2]
        return -np.sum(radii)  # Negative because we want to maximize
    
    # Constraint function for scipy.optimize
    def constraint_func(params):
        violations = constraint_violations(params)
        return violations
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # Bounds for x and y: [0.001, 0.999] to ensure containment with margin
        # Bounds for radius: [0.001, 0.499] to prevent numerical issues
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Try multiple starting points to improve chances of finding better solution
    best_solution = None
    best_sum = 0
    
    # Try 10 different initializations to get better results
    for attempt in range(10):
        # Initialize with structured grid
        circles = initialize_structured_grid()
        
        # Apply physics-based refinement to get a better starting point
        circles = refine_with_physics(circles, max_iterations=100)
        
        # Flatten initial guess
        initial_guess = circles.flatten()
        
        # Define constraints dictionary
        cons = {
            'type': 'ineq',
            'fun': constraint_func
        }
        
        # Perform optimization with higher precision
        try:
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_solution = optimized_circles.copy()
        except Exception as e:
            continue
    
    # If we found a good solution, return it; otherwise return the initial configuration
    if best_solution is not None:
        # Apply final physics refinement
        best_solution = refine_with_physics(best_solution, max_iterations=50)
        return best_solution
    else:
        # Fallback to the structured grid initialization
        circles = initialize_structured_grid()
        circles = refine_with_physics(circles, max_iterations=100)
        return circles


# EVOLVE-BLOCK-END
