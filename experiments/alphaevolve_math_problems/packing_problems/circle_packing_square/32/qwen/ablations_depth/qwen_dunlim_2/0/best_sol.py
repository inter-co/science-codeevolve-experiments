# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a direct mathematical optimization approach with geometric constraints.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    # Initialize with a structured pattern
    def initialize_circles():
        """Initialize circles using a hexagonal-like arrangement"""
        circles = np.zeros((n, 3))
        
        # Create a grid-based starting configuration
        grid_size = int(math.ceil(math.sqrt(n)))
        spacing = 1.0 / grid_size
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n:
                    break
                x = (j + 0.5) * spacing
                y = (i + 0.5) * spacing
                # Adjust positions to avoid edge issues
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                # Initial small radius
                r = spacing / 4
                circles[count] = [x, y, r]
                count += 1
            if count >= n:
                break
        
        # Fill remaining slots with random positions
        for i in range(count, n):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = random.uniform(0.01, 0.1)
            circles[i] = [x, y, r]
            
        return circles
    
    # Create constraint functions for scipy optimization
    def create_constraints():
        """Create constraint functions for optimization"""
        cons = []
        
        # Boundary constraints: each circle must fit in unit square
        for i in range(n):
            def bound_constraint(x, i=i):
                # x[3*i:3*i+3] contains [x, y, r] for circle i
                x_pos, y_pos, r = x[3*i], x[3*i+1], x[3*i+2]
                return min(x_pos - r, 1 - x_pos - r, y_pos - r, 1 - y_pos - r)
            cons.append({'type': 'ineq', 'fun': bound_constraint})
        
        # Overlap constraints: distance between centers >= sum of radii
        for i in range(n):
            for j in range(i + 1, n):
                def overlap_constraint(x, i=i, j=j):
                    x1, y1, r1 = x[3*i], x[3*i+1], x[3*i+2]
                    x2, y2, r2 = x[3*j], x[3*j+1], x[3*j+2]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    return dist - (r1 + r2)
                cons.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return cons
    
    # Optimization approach using scipy minimize
    def optimize_circles(initial_circles):
        """Optimize circle positions and radii using scipy"""
        # Flatten initial circles into parameter vector [x0, y0, r0, x1, y1, r1, ...]
        x0 = initial_circles.flatten()
        
        # Objective function (negative because we want to maximize sum of radii)
        def objective(x):
            total_radius = 0
            for i in range(n):
                total_radius += x[3*i + 2]  # radius is at index 3*i + 2
            return -total_radius  # Negative because minimize
        
        # Create constraints
        cons = create_constraints()
        
        # Bounds: x,y in [0,1], r > 0
        bounds = []
        for i in range(n):
            bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])  # x, y, r bounds
        
        # Run optimization
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                            options={'maxiter': 1000, 'ftol': 1e-6})
            if result.success:
                optimized = result.x.reshape((n, 3))
                return optimized
        except Exception:
            pass
            
        # If optimization fails, return original
        return initial_circles
    
    # Alternative: Local improvement with geometric constraints
    def geometric_improvement(circles):
        """Apply geometric optimization to improve the configuration"""
        improved = circles.copy()
        
        # Iteratively try to increase radii while maintaining constraints
        for iteration in range(100):
            improved = improved.copy()
            any_improved = False
            
            # For each circle, compute maximum allowable radius
            for i in range(n):
                x, y, r = improved[i]
                
                # Maximum radius allowed by boundaries
                max_r = min(x, 1-x, y, 1-y)
                
                # Maximum radius allowed by overlapping constraints
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = improved[j]
                        dist = math.sqrt((x - x2)**2 + (y - y2)**2)
                        if dist > 0:  # Avoid division by zero
                            max_r = min(max_r, dist - r2)
                
                # Increase radius if beneficial
                if max_r > r and max_r > 0:
                    improved[i, 2] = max_r
                    any_improved = True
            
            if not any_improved:
                break
                
        return improved
    
    # Start with structured initialization
    circles = initialize_circles()
    
    # Apply geometric improvement first
    circles = geometric_improvement(circles)
    
    # Then apply optimization
    circles = optimize_circles(circles)
    
    # Final geometric improvement
    circles = geometric_improvement(circles)
    
    return circles


# EVOLVE-BLOCK-END
