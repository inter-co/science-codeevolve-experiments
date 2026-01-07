# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a robust optimization approach inspired by successful implementations with proper 
    constraint handling and multiple restarts.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    best_sum = 0
    best_circles = None
    
    # Multiple restarts to avoid local optima - optimized for best balance of quality vs time
    for restart in range(12):  # Slightly more restarts for better exploration
        np.random.seed(42 + restart)
        
        # Better initialization using hexagonal grid with some randomness
        circles = np.zeros((n, 3))
        
        # Arrange in a hexagonal-like pattern with careful spacing
        rows = 5
        cols = 5
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        max_radius = min(spacing_x, spacing_y) * 0.365  # Slightly larger radius
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Offset every other row for hexagonal pattern
                if i % 2 == 1:
                    x += spacing_x * 0.5
                # Add moderate random perturbation for diversity
                x += np.random.uniform(-spacing_x*0.12, spacing_x*0.12)
                y += np.random.uniform(-spacing_y*0.12, spacing_y*0.12)
                circles[idx] = [x, y, max_radius]
                idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with random circles
        for i in range(idx, n):
            x = np.random.uniform(max_radius, 1-max_radius)
            y = np.random.uniform(max_radius, 1-max_radius)
            r = np.random.uniform(0.01, max_radius * 0.75)  # Slightly wider range
            circles[i] = [x, y, r]
        
        # Ensure initial constraints are satisfied
        for i in range(n):
            x, y, r = circles[i]
            # Clip to bounds and adjust radius if necessary
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            circles[i] = [x, y, r]
        
        # Define optimization bounds: [x1, y1, r1, x2, y2, r2, ...]
        bounds = []
        for i in range(n):
            bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.499)])
        
        # Flatten initial circles for optimization
        initial_params = circles.flatten()
        
        # Objective function: minimize negative sum of radii (to maximize sum)
        def objective(params):
            circles_local = params.reshape(-1, 3)
            return -np.sum(circles_local[:, 2])
        
        # Constraint functions for scipy.optimize (clean and robust)
        def containment_constraints(params):
            circles_local = params.reshape(-1, 3)
            constraints = []
            for i in range(len(circles_local)):
                x, y, r = circles_local[i]
                # Left boundary: x >= r
                constraints.append(x - r)
                # Right boundary: 1 - x >= r
                constraints.append(1 - x - r)
                # Bottom boundary: y >= r
                constraints.append(y - r)
                # Top boundary: 1 - y >= r
                constraints.append(1 - y - r)
            return np.array(constraints)
        
        def overlap_constraints(params):
            circles_local = params.reshape(-1, 3)
            constraints = []
            for i in range(len(circles_local)):
                for j in range(i+1, len(circles_local)):
                    x1, y1, r1 = circles_local[i]
                    x2, y2, r2 = circles_local[j]
                    # Distance constraint: distance >= r1 + r2
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist_sq = (r1 + r2)**2
                    constraints.append(dist_sq - min_dist_sq)
            return np.array(constraints)
        
        # Set up constraints for scipy.optimize
        constraints = [
            {'type': 'ineq', 'fun': lambda x: containment_constraints(x)},
            {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
        ]
        
        try:
            # Use SLSQP optimizer with slightly tighter tolerances
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 450, 'ftol': 1e-9, 'eps': 1e-7},  # Slight improvement
                callback=lambda x: None
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                # Validate and clean up the result
                valid_circles = []
                for i in range(len(optimized_circles)):
                    x, y, r = optimized_circles[i]
                    # Ensure valid positioning
                    x = np.clip(x, r, 1-r)
                    y = np.clip(y, r, 1-r)
                    r = np.clip(r, 1e-6, 0.499)
                    valid_circles.append([x, y, r])
                optimized_circles = np.array(valid_circles)
                
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized_circles.copy()
            else:
                # Even if optimization fails, keep the best configuration so far
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
                    
        except Exception as e:
            # If optimization fails due to numerical issues, keep current configuration
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
    
    # Final validation and cleanup
    if best_circles is None:
        # Fallback to initial configuration
        circles = np.zeros((n, 3))
        rows = 5
        cols = 5
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        max_radius = min(spacing_x, spacing_y) * 0.365
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                if i % 2 == 1:
                    x += spacing_x * 0.5
                circles[idx] = [x, y, max_radius]
                idx += 1
            if idx >= n:
                break
        
        for i in range(idx, n):
            x = np.random.uniform(max_radius, 1-max_radius)
            y = np.random.uniform(max_radius, 1-max_radius)
            r = np.random.uniform(0.01, max_radius * 0.75)
            circles[i] = [x, y, r]
            
        best_circles = circles
    
    return best_circles


# EVOLVE-BLOCK-END
