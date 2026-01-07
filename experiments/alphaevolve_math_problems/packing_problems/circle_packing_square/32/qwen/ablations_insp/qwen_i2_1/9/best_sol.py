# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and scipy optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Use hexagonal initialization like INSPIRATION 2 but with better parameters
    def initialize_hexagonal():
        # Create hexagonal grid pattern for good initial configuration
        rows = 6
        cols = 6
        circles = []
        
        # Hexagonal packing parameters with better spacing
        spacing_x = 0.15
        spacing_y = 0.15 * math.sqrt(3)/2
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Adjust for hexagonal offset
                if i % 2 == 1:
                    x += spacing_x / 2
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, 0.05])
        
        # Fill remaining positions with random placements
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        return np.array(circles)
    
    # Objective function to maximize sum of radii
    def objective(params):
        circles = params.reshape(-1, 3)
        # Negative because we want to maximize (scipy minimizes)
        return -np.sum(circles[:, 2])
    
    # Better constraint handling - use scipy-friendly constraint format
    def get_constraints(params):
        """Generate constraints for scipy optimization"""
        circles = params.reshape(-1, 3)
        constraints = []
        
        # Containment constraints: r <= x, r <= y, r <= 1-x, r <= 1-y
        for i in range(len(circles)):
            x, y, r = circles[i]
            constraints.extend([
                x - r,      # x >= r
                y - r,      # y >= r  
                1 - x - r,  # 1-x >= r
                1 - y - r   # 1-y >= r
            ])
        
        # Non-overlap constraints: distance^2 >= (r1+r2)^2
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                constraints.append(dist_sq - (r1+r2)**2)
        
        return np.array(constraints)
    
    # Initial configuration
    circles = initialize_hexagonal()
    initial_params = circles.flatten()
    
    # Define bounds for parameters: x, y, r for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Define constraints for scipy
    cons = {
        'type': 'ineq',
        'fun': lambda p: get_constraints(p)
    }
    
    # Try optimization with scipy
    try:
        # Use SLSQP method which works well for this type of problem
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        pass
    
    # Fallback to simple local optimization if scipy fails
    def simple_local_search(initial_circles):
        circles = initial_circles.copy()
        max_iter = 500
        
        for iteration in range(max_iter):
            improved = False
            
            # Try to increase radii
            for i in range(n):
                old_r = circles[i, 2]
                old_x, old_y = circles[i, 0], circles[i, 1]
                
                # Calculate maximum possible radius
                max_radius = min(old_x, old_y, 1-old_x, 1-old_y)
                
                # Find minimum distance to other circles
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                        min_dist = min(min_dist, dist)
                
                # Safe radius considering overlap constraints
                safe_radius = min(max_radius, min_dist/2) if min_dist != float('inf') else max_radius
                
                if safe_radius > old_r + 1e-6:
                    circles[i, 2] = safe_radius
                    improved = True
            
            if not improved:
                break
                
        return circles
    
    # Apply simple local search as fallback
    circles = simple_local_search(circles)
    return circles


# EVOLVE-BLOCK-END
