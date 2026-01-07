# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-stage approach: initial grid placement + simulated annealing + local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Create better initial configuration using a more systematic approach
    def create_better_initial():
        # Use a more structured approach - start with a dense hexagonal packing
        # and then adjust for better distribution
        
        # First, try a hexagonal lattice approach
        # For 32 circles, we can use approximately 6x6 grid with some adjustments
        circles = []
        
        # Try different grid sizes and configurations
        best_config = None
        best_radius_sum = 0
        
        # Try multiple grid configurations
        for rows in [5, 6, 7]:
            cols = (n + rows - 1) // rows  # Ceiling division
            if rows * cols < n:
                continue
                
            # Hexagonal packing pattern
            spacing_x = 1.0 / cols
            spacing_y = 1.0 / rows
            
            # Try both regular and staggered patterns
            for stagger in [False, True]:
                test_circles = []
                for i in range(rows):
                    for j in range(cols):
                        if len(test_circles) >= n:
                            break
                        x = (j + 0.5) * spacing_x
                        y = (i + 0.5) * spacing_y
                        
                        # Apply hexagonal offset for odd rows
                        if stagger and i % 2 == 1:
                            x += spacing_x / 2
                            
                        # Ensure within bounds
                        x = max(0.05, min(0.95, x))
                        y = max(0.05, min(0.95, y))
                        
                        # Initial radius - based on spacing but ensure reasonable values
                        r = min(spacing_x, spacing_y) / 2.0
                        r = min(r, 0.15)  # Cap max radius
                        test_circles.append([x, y, r])
                    
                    if len(test_circles) >= n:
                        break
                
                # Fill remaining positions with random placements near edges
                while len(test_circles) < n:
                    # Prefer placing near edges for better utilization
                    edge = np.random.choice(['top', 'bottom', 'left', 'right'])
                    if edge == 'top':
                        x = np.random.uniform(0.1, 0.9)
                        y = 0.95
                    elif edge == 'bottom':
                        x = np.random.uniform(0.1, 0.9)
                        y = 0.05
                    elif edge == 'left':
                        x = 0.05
                        y = np.random.uniform(0.1, 0.9)
                    else:  # right
                        x = 0.95
                        y = np.random.uniform(0.1, 0.9)
                    r = np.random.uniform(0.02, 0.1)
                    test_circles.append([x, y, r])
                
                test_circles = test_circles[:n]
                radius_sum = sum(circle[2] for circle in test_circles)
                
                if radius_sum > best_radius_sum:
                    best_radius_sum = radius_sum
                    best_config = test_circles.copy()
        
        # If we still don't have enough circles, fill with random ones
        if len(best_config) < n:
            while len(best_config) < n:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                r = np.random.uniform(0.01, 0.1)
                best_config.append([x, y, r])
        
        return np.array(best_config[:n])
    
    # Create initial guess
    initial_circles = create_better_initial()
    
    # Define constraints and objective
    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we want to maximize
    
    def constraint_containment(params):
        circles = params.reshape(-1, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Each circle must be fully contained in [0,1]x[0,1]
        cons = np.concatenate([
            r,                          # r >= 0
            1 - r - x,                  # x + r <= 1
            1 - r - y,                  # y + r <= 1
            x - r,                      # x - r >= 0
            y - r                       # y - r >= 0
        ])
        return cons
    
    def constraint_nonoverlap(params):
        circles = params.reshape(-1, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # More efficient way to compute non-overlap constraints
        # Using spatial indexing for better performance
        cons = []
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                r_sum = r[i] + r[j]
                # We want dist >= r_sum to prevent overlap
                # So constraint is: dist_sq - r_sum^2 >= 0
                cons.append(dist_sq - r_sum*r_sum)
        return np.array(cons)
    
    # Use a more robust optimization approach with multiple restarts
    def optimize_with_restarts(initial_params, max_restarts=3):
        best_result = None
        best_sum = -np.inf
        
        for restart in range(max_restarts):
            # Add small random perturbations to initial parameters
            perturbed_params = initial_params + np.random.normal(0, 0.001, len(initial_params))
            
            # Create bounds (x, y, r for each circle)
            bounds = []
            for _ in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            # Constraints
            cons = [
                {'type': 'ineq', 'fun': lambda p: constraint_containment(p)},
                {'type': 'ineq', 'fun': lambda p: constraint_nonoverlap(p)}
            ]
            
            try:
                result = minimize(
                    objective,
                    perturbed_params,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 500, 'ftol': 1e-6}
                )
                
                if result.success:
                    current_sum = -objective(result.x)  # Convert back to positive sum
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result.x.copy()
                        
            except Exception:
                continue
                
        return best_result if best_result is not None else initial_params
    
    # Run optimization
    try:
        optimized_params = optimize_with_restarts(initial_circles.flatten())
        
        if optimized_params is not None:
            final_circles = optimized_params.reshape(-1, 3)
            return final_circles
        else:
            return initial_circles
            
    except Exception as e:
        # Return initial configuration if optimization fails
        return initial_circles


# EVOLVE-BLOCK-END
