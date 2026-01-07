# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining spatial initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores 
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize circles in a grid pattern with small random perturbations
    def initialize_circles():
        circles = np.zeros((n, 3))
        # Grid layout with some randomness
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Create grid points
        x_positions = np.linspace(0.1, 0.9, cols)
        y_positions = np.linspace(0.1, 0.9, rows)
        
        # Fill circles with initial positions and radii
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Add small random perturbation
                x = max(0.05, min(0.95, x_positions[j] + np.random.normal(0, 0.02)))
                y = max(0.05, min(0.95, y_positions[i] + np.random.normal(0, 0.02)))
                # Initial radius - small enough to fit in unit square
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
                
        return circles
    
    # Constraint functions for optimization
    def constraint_radius(circles_flat):
        """Ensure all radii are positive and within bounds"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Radius constraint: radius must be positive and not exceed boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Radius cannot be negative
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+2]})
            # Radius cannot exceed distance to boundaries
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1-x >= r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})  # 1-y >= r
            
        return constraints
    
    def constraint_overlap(circles_flat):
        """Ensure no overlap between circles"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # For each pair of circles, add constraint that distance >= sum of radii
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                def overlap_constraint(c):
                    x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                    x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                    dist_sq = (x1-x2)**2 + (y1-y2)**2
                    # Distance squared should be >= (r1+r2)^2
                    return dist_sq - (r1+r2)**2
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
                
        return constraints
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    # Initialize
    circles = initialize_circles()
    
    # Flatten for optimization
    circles_flat = circles.flatten()
    
    # Set up constraints
    cons = []
    # Add boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x >= r
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y >= r
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1-x >= r
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})  # 1-y >= r
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+2]})  # r >= 0
    
    # Add overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(c, i=i, j=j):
                x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                return dist_sq - (r1+r2)**2
            cons.append({'type': 'ineq', 'fun': overlap_constraint})
    
    # Optimization bounds (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r ranges
    
    # Optimize
    try:
        result = minimize(objective, circles_flat, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6})
        if result.success:
            circles = result.x.reshape(-1, 3)
        else:
            # If optimization fails, return the initial configuration
            pass
    except Exception as e:
        # In case of optimization failure, use initial configuration
        pass
    
    return circles


# EVOLVE-BLOCK-END
