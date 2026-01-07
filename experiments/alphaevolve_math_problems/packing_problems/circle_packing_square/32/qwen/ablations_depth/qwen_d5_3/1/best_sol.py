# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using hexagonal packing heuristic
    def initialize_hexagonal():
        circles = []
        # Try to arrange in hexagonal pattern
        rows = int(math.sqrt(n))
        cols = int(math.ceil(n / rows))
        
        # Hexagonal packing parameters
        spacing_x = 1.0 / cols
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        # Adjust spacing to leave room for radii
        max_radius = min(spacing_x, spacing_y) / 2.5
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                circles.append([x, y, max_radius])
            if len(circles) >= n:
                break
        
        # Fill remaining positions with random placements near edges
        while len(circles) < n:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            # Small random radius
            r = np.random.uniform(0.01, 0.1)
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Create initial configuration
    circles = initialize_hexagonal()
    
    # Define constraint functions
    def containement_constraints(circles_flat):
        """Ensure all circles are contained within unit square"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Each circle must be fully inside the unit square
        for i in range(len(circles)):
            x, y, r = circles[i]
            # r <= x <= 1-r and r <= y <= 1-r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1-x-r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})  # 1-y-r >= 0
            
        return constraints
    
    def overlap_constraints(circles_flat):
        """Ensure no overlaps between circles"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Check all pairs of circles
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                def overlap_constraint(c, i=i, j=j):
                    x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                    x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    # Distance squared should be >= (r1+r2)^2
                    return dist_sq - (r1 + r2)**2
                    
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
                
        return constraints
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Flatten initial circles for optimization
    x0 = circles.flatten()
    
    # Set up bounds (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # r < 0.5 to prevent overlap issues
    
    # Create constraints
    cons = []
    # Add containment constraints
    for i in range(n):
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})
    
    # Add overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(c, i=i, j=j):
                x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                return dist_sq - (r1 + r2)**2
            cons.append({'type': 'ineq', 'fun': overlap_constraint})
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure all circles are valid
            for i in range(len(optimized_circles)):
                x, y, r = optimized_circles[i]
                # Clip values to valid ranges
                optimized_circles[i] = [max(0.001, min(0.999, x)), 
                                      max(0.001, min(0.999, y)), 
                                      max(0.001, min(0.499, r))]
            return optimized_circles
    except Exception as e:
        pass
    
    # If optimization fails, return the initial configuration
    return circles


# EVOLVE-BLOCK-END
