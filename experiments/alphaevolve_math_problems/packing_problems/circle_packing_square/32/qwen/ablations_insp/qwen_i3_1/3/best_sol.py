# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: improved initialization + robust constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Improved initialization using a more systematic approach based on known good packings
    def initialize_better_layout():
        # Use a more sophisticated approach inspired by hexagonal packing
        # Create a grid with more careful spacing to achieve better packing density
        
        # For 32 circles, we can try a 6x6 grid (36 cells) and remove 4
        cols = 6
        rows = 6
        
        # Calculate spacing that allows for better packing
        spacing_x = 0.9 / cols  # leave some margin
        spacing_y = 0.9 / rows
        
        circles = []
        
        # Create a hexagonal-like arrangement
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Offset every other row for better packing
                offset = 0.5 if i % 2 == 1 else 0.0
                
                x = 0.05 + (j + offset) * spacing_x
                y = 0.05 + i * spacing_y
                
                # Ensure we're within bounds and not too close to edges
                if x <= 0.95 and y <= 0.95:
                    # Initial radius - start with larger value to allow optimization
                    radius = min(spacing_x, spacing_y) * 0.4
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
            constraints.append(x - r)  # x >= r 
            constraints.append(1 - x - r)  # x <= 1-r
            constraints.append(y - r)  # y >= r
            constraints.append(1 - y - r)  # y <= 1-r
            
        # Overlap constraints - ensure no overlap between circles
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_flat[i]
                x2, y2, r2 = circles_flat[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # Distance squared should be >= (r1+r2)^2 for non-overlap
                constraints.append(dist_sq - (r1 + r2)**2)
                
        return np.array(constraints)
    
    # Flatten initial circles for optimization
    initial_params = circles.flatten()
    
    # Set up bounds for optimization (more reasonable bounds)
    bounds = []
    for i in range(n):
        # Bounds for positions (slightly inside square to prevent boundary issues)
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Try multiple optimization methods for better robustness
    methods_to_try = ['SLSQP', 'trust-constr', 'COBYLA']
    
    for method in methods_to_try:
        try:
            # Optimization parameters - increased iterations and tighter tolerances
            options = {'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            
            # Perform optimization using different methods
            result = minimize(
                objective,
                initial_params,
                method=method,
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': constraint_func}],
                options=options,
                tol=1e-8
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                # Ensure final constraints are met by clamping values
                final_circles = []
                for i in range(n):
                    x, y, r = optimized_circles[i]
                    # Clamp values to valid ranges
                    x = max(0.001, min(0.999, x))
                    y = max(0.001, min(0.999, y))
                    r = max(0.001, min(0.499, r))
                    final_circles.append([x, y, r])
                
                return np.array(final_circles)
        except Exception:
            # Continue to next method if this one fails
            continue
    
    # If all optimization attempts fail, return the initial configuration
    return circles


# EVOLVE-BLOCK-END
