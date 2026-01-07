# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining mathematical programming and spatial partitioning.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a symmetric configuration that's known to work well
    def initialize_symmetric_config():
        # Create a grid-like pattern with some jitter
        circles = np.zeros((n, 3))
        
        # Generate points in a grid pattern with some jitter
        rows_cols = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (rows_cols + 1)
        
        idx = 0
        for i in range(rows_cols):
            for j in range(rows_cols):
                if idx >= n:
                    break
                x = (i + 1) * spacing + random.uniform(-spacing/4, spacing/4)
                y = (j + 1) * spacing + random.uniform(-spacing/4, spacing/4)
                # Initial radius - small enough to fit in square
                r = min(x, 1-x, y, 1-y) * 0.4
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
        return circles
    
    # Constraint checking
    def check_constraints(circles):
        """Check if all constraints are satisfied"""
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Check containment
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
            # Check non-overlap with all other circles
            for j in range(i+1, len(circles)):
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                if dist < r + r2:
                    return False
        return True
    
    # Objective function to maximize (negative because we minimize)
    def objective(params):
        # Reshape params into circles array
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we want to maximize
    
    # Constraint functions (inspired by INSPIRATION 2)
    def contain_constraint(params):
        circles = params.reshape(-1, 3)
        # For each circle, check that it's fully contained
        cons = []
        for i in range(len(circles)):
            x, y, r = circles[i]
            # r <= x <= 1-r
            cons.append(x - r)      # Should be >= 0
            cons.append(1 - x - r)  # Should be >= 0
            # r <= y <= 1-r  
            cons.append(y - r)      # Should be >= 0
            cons.append(1 - y - r)  # Should be >= 0
        return np.array(cons)
    
    def overlap_constraint(params):
        circles = params.reshape(-1, 3)
        # For each pair of circles, check they don't overlap
        cons = []
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                # dist >= r1 + r2 (non-overlapping)
                cons.append(dist - r1 - r2)  # Should be >= 0
        return np.array(cons)
    
    # Multi-start optimization with different initial configurations (inspired by INSPIRATION 2)
    best_result = None
    best_sum = -float('inf')
    
    # Try several different initial configurations
    for attempt in range(10):  # Reduced attempts to stay within time limit
        # Initialize with different strategies
        if attempt == 0:
            # Grid-based initialization
            circles = initialize_symmetric_config()
        else:
            # Random initialization with some structure
            circles = np.zeros((n, 3))
            for i in range(n):
                # Place circles with some minimum distance from boundaries
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                # Start with small radius and increase if possible
                r = random.uniform(0.01, 0.1)
                circles[i] = [x, y, r]
        
        # Flatten for optimization
        initial_params = circles.flatten()
        
        # Optimization bounds
        bounds = []
        for i in range(n):
            # x, y, r bounds
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        try:
            # Use SLSQP method which handles constraints well (as in INSPIRATION 2)
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: contain_constraint(p)},
                    {'type': 'ineq', 'fun': lambda p: overlap_constraint(p)}
                ],
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6}  # Reduced iterations for speed
            )
            
            if result.success:
                final_circles = result.x.reshape(-1, 3)
                total_radius = np.sum(final_circles[:, 2])
                if total_radius > best_sum:
                    best_sum = total_radius
                    best_result = final_circles.copy()
                    
        except Exception as e:
            continue
    
    # If we still have no good result, return a basic configuration
    if best_result is None:
        best_result = initialize_symmetric_config()
    
    # Final refinement with local optimization
    try:
        # Apply bounds to ensure valid ranges
        refined_circles = best_result.copy()
        for i in range(n):
            x, y, r = refined_circles[i]
            # Ensure proper bounds
            refined_circles[i] = [
                max(0.001, min(0.999, x)),
                max(0.001, min(0.999, y)),
                max(0.001, min(0.499, r))
            ]
        
        # Final optimization with L-BFGS-B for fine-tuning
        final_params = refined_circles.flatten()
        result = minimize(
            objective,
            final_params,
            method='L-BFGS-B',
            bounds=[(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n,
            options={'maxiter': 200, 'ftol': 1e-6}  # Reduced iterations for speed
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            # Verify constraints and adjust if needed
            if check_constraints(final_circles):
                return final_circles
    except Exception:
        pass
    
    return best_result


# EVOLVE-BLOCK-END
