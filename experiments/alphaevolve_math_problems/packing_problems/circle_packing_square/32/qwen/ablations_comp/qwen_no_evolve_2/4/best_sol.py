# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles in a structured pattern (hexagonal lattice)
    def initialize_hexagonal_pattern():
        # Create a hexagonal grid pattern
        rows = 6
        cols = 6
        circles = []
        
        # Hexagon packing parameters
        spacing_x = 0.15
        spacing_y = 0.15 * math.sqrt(3)/2
        
        # Generate initial positions
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Adjust for hexagonal offset
                if i % 2 == 1:
                    x += spacing_x / 2
                
                # Ensure we're within bounds
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, 0.05])
        
        # Fill remaining slots with random positions
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        return np.array(circles)
    
    # Initialize
    circles = initialize_hexagonal_pattern()
    
    # Define constraint functions
    def radius_constraint(i, circles):
        """Ensure circle i stays within bounds"""
        x, y, r = circles[i]
        return min(r, 1-r, 1-y, y)
    
    def non_overlap_constraints(circles):
        """Generate all pairwise non-overlap constraints"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                # Distance between centers should be at least sum of radii
                constraints.append(dist_sq - (r1+r2)**2)
        return constraints
    
    # Objective function to maximize (negative because minimize)
    def objective(circles_flat):
        # Extract radii
        radii = circles_flat[2::3]
        return -np.sum(radii)
    
    # Constraint functions for scipy optimizer
    def bound_constraints(circles_flat):
        # Each circle's radius must be within bounds
        constraints = []
        for i in range(n):
            x, y, r = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
            # r must be <= x, 1-x, y, 1-y
            constraints.extend([
                x - r,           # x >= r
                1 - x - r,       # 1-x >= r
                y - r,           # y >= r
                1 - y - r        # 1-y >= r
            ])
        return np.array(constraints)
    
    def overlap_constraints(circles_flat):
        # Non-overlap constraints
        constraints = []
        circles = circles_flat.reshape(-1, 3)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                # Distance between centers should be at least sum of radii
                constraints.append(dist_sq - (r1+r2)**2)
        return np.array(constraints)
    
    # Flatten initial circles for optimization
    initial_flat = circles.flatten()
    
    # Set up bounds for optimization (x, y, r) for each circle
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
    
    # Define constraints for scipy optimizer
    cons = [
        {'type': 'ineq', 'fun': lambda x: bound_constraints(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
    ]
    
    # Perform optimization
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure valid radii (not negative)
            optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], 0.001)
            return optimized_circles
    except Exception as e:
        pass
    
    # Return initial configuration if optimization fails
    return circles


# EVOLVE-BLOCK-END
