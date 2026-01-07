# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: improved initialization + robust constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Improved initialization using a more systematic approach
    def initialize_better_layout():
        # Start with a regular grid pattern but adjust for better packing
        # We'll use a more sophisticated approach based on known good packings
        
        # Try a rectangular grid first, then optimize
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # Make sure we have enough space
        while cols * rows < n:
            cols += 1
            
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Add slight offset to create a more even distribution
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Adjust for better packing
                if i % 2 == 1:  # Odd rows
                    x += spacing_x * 0.25
                elif i % 2 == 0 and j % 2 == 1:  # Even rows, odd columns
                    x -= spacing_x * 0.25
                    
                # Ensure we're within bounds
                if x <= 1 and y <= 1:
                    # Initial radius - start with smaller value to allow room for optimization
                    radius = min(spacing_x, spacing_y) * 0.3
                    circles.append([x, y, radius])
                    
        # If we don't have enough circles, add more with random placement
        while len(circles) < n:
            # Place randomly but ensure they're not too close to boundaries
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Radius should be reasonable given available space
            max_radius = min(x, 1-x, y, 1-y) * 0.8
            radius = np.random.uniform(0.01, max_radius)
            circles.append([x, y, radius])
            
        return np.array(circles[:n])
    
    # Initialize with better layout
    circles = initialize_better_layout()
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        # Reshape params back into circles array
        circles_flat = params.reshape(-1, 3)
        # Maximize sum of radii (minimize negative)
        return -np.sum(circles_flat[:, 2])
    
    # Constraint functions for optimization - more robust formulation
    def constraint_func(params):
        circles_flat = params.reshape(-1, 3)
        constraints = []
        
        # Boundary constraints: each circle must fit within unit square
        for i in range(n):
            x, y, r = circles_flat[i]
            # Circle must be within bounds with margin
            constraints.append(x - r - 1e-6)  # x >= r + small epsilon
            constraints.append(1 - x - r - 1e-6)  # x <= 1-r - small epsilon
            constraints.append(y - r - 1e-6)  # y >= r + small epsilon
            constraints.append(1 - y - r - 1e-6)  # y <= 1-r - small epsilon
            
        # Overlap constraints - ensure no overlap between circles
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_flat[i]
                x2, y2, r2 = circles_flat[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # Distance squared should be >= (r1+r2)^2 for non-overlap
                # Add small epsilon to avoid numerical issues
                constraints.append(dist_sq - (r1 + r2)**2 + 1e-12)
                
        return np.array(constraints)
    
    # Flatten initial circles for optimization
    initial_params = circles.flatten()
    
    # Set up bounds for optimization (more reasonable bounds)
    bounds = []
    for i in range(n):
        # Bounds for positions (slightly inside square to prevent boundary issues)
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.499)])
    
    # Try multiple optimization methods for better robustness
    methods_to_try = ['SLSQP', 'trust-constr']
    
    for method in methods_to_try:
        try:
            # Optimization parameters
            options = {'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
            
            # Perform optimization using different methods
            result = minimize(
                objective,
                initial_params,
                method=method,
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': constraint_func}],
                options=options,
                tol=1e-6
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                # Ensure final constraints are met by clamping values
                final_circles = []
                for i in range(n):
                    x, y, r = optimized_circles[i]
                    # Clamp values to valid ranges
                    x = max(1e-6, min(1-1e-6, x))
                    y = max(1e-6, min(1-1e-6, y))
                    r = max(1e-6, min(0.499, r))
                    final_circles.append([x, y, r])
                
                return np.array(final_circles)
        except Exception:
            # Continue to next method if this one fails
            continue
    
    # If all optimization attempts fail, return the initial configuration
    return circles


# EVOLVE-BLOCK-END
