# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial hexagonal packing with optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Create initial configuration using hexagonal packing pattern
    def create_hexagonal_initial():
        # Try to place circles in a hexagonal pattern
        circles = []
        radius_guess = 0.1
        
        # Place in a roughly hexagonal pattern
        rows = 5
        cols = 5
        spacing_x = 2 * radius_guess
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = spacing_x * j + radius_guess
                y = spacing_y * i + radius_guess
                # Adjust for hexagonal offset
                if i % 2 == 1:
                    x += spacing_x / 2
                circles.append([x, y, radius_guess])
        
        # Ensure we have exactly n circles
        while len(circles) < n:
            circles.append([0.5, 0.5, radius_guess])
            
        return np.array(circles[:n])
    
    # Constraint functions for optimization
    def contain_constraints(circles_flat):
        """Ensure all circles are contained within the unit square"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Each circle must satisfy containment constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            # r <= x <= 1-r and r <= y <= 1-r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1 - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})  # 1 - y - r >= 0
            
        return constraints
    
    def overlap_constraints(circles_flat):
        """Ensure no overlapping circles"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Check pairwise non-overlap constraints
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                # Distance between centers >= sum of radii
                def constraint_func(c):
                    x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                    x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    return dist_sq - (r1 + r2)**2
                
                constraints.append({'type': 'ineq', 'fun': constraint_func})
                
        return constraints
    
    # Create initial guess
    initial_circles = create_hexagonal_initial()
    
    # Flatten for optimization
    initial_flat = initial_circles.flatten()
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we're minimizing
    
    # Create constraints
    cons = []
    # Add containment constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})  # 1 - y - r >= 0
    
    # Add overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(c, i=i, j=j):
                x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                return dist_sq - (r1 + r2)**2
            cons.append({'type': 'ineq', 'fun': overlap_constraint})
    
    # Bounds for variables: x, y, r for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r ranges
    
    # Run optimization
    try:
        result = minimize(objective, initial_flat, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial guess if optimization fails
            return initial_circles
    except Exception as e:
        # Return initial guess if optimization fails
        return initial_circles


# EVOLVE-BLOCK-END
