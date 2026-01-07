# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal grid initialization with optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Improved hexagonal grid initialization with better distribution
    def initialize_hexagonal_layout():
        # Create a refined hexagonal grid pattern
        rows = 6
        cols = 6
        
        # Make sure we have enough points
        if rows * cols < n:
            rows = 5
            cols = 7
            
        circles = []
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Offset every other row for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x / 2
                    
                # Ensure we're within bounds and not too close to edges
                if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                    # Initial radius estimate based on available space
                    max_radius = min(x, 1-x, y, 1-y) * 0.4
                    circles.append([x, y, max_radius])
        
        # Fill remaining positions if needed with better spread
        while len(circles) < n:
            # Use a more strategic placement approach
            if len(circles) < 16:
                # Place in upper left quadrant
                x = np.random.uniform(0.05, 0.45)
                y = np.random.uniform(0.05, 0.45)
            elif len(circles) < 24:
                # Place in lower right quadrant  
                x = np.random.uniform(0.55, 0.95)
                y = np.random.uniform(0.55, 0.95)
            else:
                # Place in remaining quadrants
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        return np.array(circles[:n])
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Sum of all radii (negative for maximization)
    
    # Constraints for scipy optimizer - optimized version
    def constraint_containment(circles_flat):
        """Ensure all circles are within the unit square"""
        n = len(circles_flat) // 3
        result = []
        for i in range(n):
            x, y, r = circles_flat[3*i:3*i+3]
            # Constraints: x >= r, y >= r, x <= 1-r, y <= 1-r
            # So: x-r >= 0, y-r >= 0, 1-x-r >= 0, 1-y-r >= 0
            result.extend([x - r, y - r, 1 - x - r, 1 - y - r])
        return np.array(result)
    
    def constraint_nonoverlap(circles_flat):
        """Ensure no two circles overlap using efficient pairwise checking"""
        n = len(circles_flat) // 3
        result = []
        circles = circles_flat.reshape(-1, 3)
        
        # Efficiently compute non-overlap constraints
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                # Distance squared between centers
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                # Minimum distance squared for no overlap
                min_dist_sq = (r1 + r2) * (r1 + r2)
                # We want dist >= r1 + r2, so dist_sq >= min_dist_sq
                # Constraint: dist_sq - min_dist_sq >= 0
                result.append(dist_sq - min_dist_sq)
        return np.array(result)
    
    # Initial guess
    initial_circles = initialize_hexagonal_layout()
    initial_guess = initial_circles.flatten()
    
    # Define bounds for each variable (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x bounds: r <= x <= 1-r, so r <= 0.5 (to allow valid x)
        # y bounds: r <= y <= 1-r, so r <= 0.5 (to allow valid y)  
        # r bounds: r > 0, r <= 0.499 (slightly less than 0.5 for safety)
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.499)])
    
    # Set up constraints
    constraints = [
        {'type': 'ineq', 'fun': constraint_containment},
        {'type': 'ineq', 'fun': constraint_nonoverlap}
    ]
    
    # Optimization options with even better settings
    options = {'maxiter': 2000, 'ftol': 1e-8, 'gtol': 1e-8}
    
    # Run optimization with multiple restarts for better results
    best_solution = initial_guess.copy()
    best_sum = -objective(initial_guess)  # Negative because we minimize
    
    # Try several restarts with small random perturbations
    for restart in range(15):  # Increase restarts for better chance
        # Start with the initial solution
        current_solution = initial_guess.copy()
        
        # Apply small random perturbations to positions only (not radii) to escape local minima
        for i in range(n):
            # Slightly perturb positions with more variation
            current_solution[3*i] += random.uniform(-0.02, 0.02)
            current_solution[3*i+1] += random.uniform(-0.02, 0.02)
            # Keep within bounds
            current_solution[3*i] = np.clip(current_solution[3*i], 1e-6, 1-1e-6)
            current_solution[3*i+1] = np.clip(current_solution[3*i+1], 1e-6, 1-1e-6)
        
        try:
            # Run optimization with better tolerance and more iterations
            result = minimize(
                objective,
                current_solution,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options=options,
                tol=1e-8
            )
            
            if result.success:
                current_sum = -objective(result.x)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_solution = result.x.copy()
            else:
                # Even if not successful, still evaluate the current solution
                current_sum = -objective(current_solution)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_solution = current_solution.copy()
                    
        except Exception:
            # In case of any error, continue with current best
            continue
    
    # Convert back to circles array
    circles = best_solution.reshape(-1, 3)
    return circles


# EVOLVE-BLOCK-END
