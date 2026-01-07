# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Better initialization using hexagonal packing with refinement
    def initialize_hexagonal():
        # Create a hexagonal grid pattern with better spacing
        rows = 6
        cols = 6
        while rows * cols < n:
            cols += 1
            
        spacing_x = 0.9 / cols  # Leave some margin
        spacing_y = 0.9 / rows
        
        circles = []
        for i in range(rows):
            y = 0.05 + (i + 0.5) * spacing_y
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                if i % 2 == 1:  # Offset every other row
                    x += spacing_x * 0.5
                    
                # Start with a reasonable radius
                r = min(spacing_x, spacing_y) * 0.35
                
                # Ensure circle fits in square
                if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                    circles.append([x, y, r])
        
        # Fill remaining spots with strategic random placement
        while len(circles) < n:
            # Use a more intelligent placement strategy
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Start with a reasonable initial radius that allows room for growth
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    # Phase 2: Efficient constraint handling using vectorized operations
    def create_constraints_vectorized():
        """Create constraints more efficiently using vectorized operations"""
        # Pre-compute all constraint pairs
        constraint_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                constraint_pairs.append((i, j))
        
        # Create constraint functions
        def contain_constraints(vars):
            # Each circle must satisfy containment constraints
            constraints = []
            for i in range(n):
                x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
                # x >= r, y >= r, 1-x >= r, 1-y >= r
                constraints.extend([
                    x - r,      # x >= r
                    y - r,      # y >= r
                    1 - x - r,  # 1-x >= r
                    1 - y - r   # 1-y >= r
                ])
            return np.array(constraints)
        
        def overlap_constraints(vars):
            # No overlaps between any pair of circles
            constraints = []
            for i, j in constraint_pairs:
                x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                # We want distance_sq >= (r1 + r2)^2
                # So: distance_sq - (r1 + r2)^2 >= 0
                constraints.append(distance_sq - (r1 + r2)**2)
            return np.array(constraints)
        
        return contain_constraints, overlap_constraints
    
    # Phase 3: Optimized optimization approach
    def optimize_with_constraints(initial_config):
        # Flatten initial configuration
        x0 = initial_config.flatten()
        
        # Create constraint functions
        contain_cons, overlap_cons = create_constraints_vectorized()
        
        # Objective function (negative because we want to maximize sum of radii)
        def objective(vars):
            return -np.sum(vars[2::3])  # Sum of all radii (indices 2,5,8,...)
        
        # Constraint functions
        def contain_func(vars):
            return contain_cons(vars)
            
        def overlap_func(vars):
            return overlap_cons(vars)
        
        # Bounds for x, y, r coordinates
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': contain_func},
            {'type': 'ineq', 'fun': overlap_func}
        ]
        
        # Try multiple optimization runs with different tolerances
        best_result = None
        best_sum = -np.inf
        
        try:
            # Run with standard settings
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                current_sum = -result.fun  # Convert back to sum of radii
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception:
            pass
        
        # If optimization failed, return initial config
        if best_result is None:
            return initial_config
        else:
            return best_result.x.reshape((n, 3))
    
    # Phase 4: Main execution
    try:
        # Initialize with better layout
        circles = initialize_hexagonal()
        
        # Optimize the configuration
        optimized_circles = optimize_with_constraints(circles)
        
        # Final validation and cleanup
        final_circles = np.copy(optimized_circles)
        
        # Ensure all circles are valid with proper bounds
        for i in range(n):
            x, y, r = final_circles[i]
            # Make sure the circle fits within the square
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            r = np.clip(r, 0.001, min(x, 1-x, y, 1-y))
            final_circles[i] = [x, y, r]
            
        return final_circles
        
    except Exception as e:
        # Fallback to basic initialization if anything goes wrong
        return initialize_hexagonal()


# EVOLVE-BLOCK-END
