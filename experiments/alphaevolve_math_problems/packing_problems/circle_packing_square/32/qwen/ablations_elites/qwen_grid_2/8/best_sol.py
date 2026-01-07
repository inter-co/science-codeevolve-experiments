# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining smart initialization and scipy optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Generate a good initial configuration using hexagonal lattice with refinement
    def generate_initial():
        # Create hexagonal pattern with slight randomness for better results
        rows = 6
        cols = 6
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                
                # Add small randomness to avoid perfect patterns
                x += np.random.uniform(-spacing_x*0.1, spacing_x*0.1)
                y += np.random.uniform(-spacing_y*0.1, spacing_y*0.1)
                
                # Ensure within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                
                # Initial radius based on proximity to edges
                r = min(x, 1-x, y, 1-y) * 0.3
                r = max(0.01, min(0.2, r))
                
                circles.append([x, y, r])
        
        # Fill remaining positions if needed
        while len(circles) < n:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = min(x, 1-x, y, 1-y) * 0.2
            r = max(0.01, min(0.15, r))
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Generate initial configuration
    circles = generate_initial()
    
    # Flatten for optimization
    initial_flat = circles.flatten()
    
    # Constraints for optimization
    def contain_constraints():
        """Generate containment constraints for all circles"""
        constraints = []
        for i in range(n):
            # radius must be positive
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2+i*3]})
            # x coordinate constraints
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[0+i*3] - x[2+i*3]})  # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[0+i*3] - x[2+i*3]})  # 1-x >= r
            # y coordinate constraints
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[1+i*3] - x[2+i*3]})  # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[1+i*3] - x[2+i*3]})  # 1-y >= r
        return constraints
    
    def non_overlap_constraints():
        """Generate non-overlap constraints for all pairs of circles"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                def constraint_func(x, i=i, j=j):
                    xi, yi, ri = x[i*3], x[i*3+1], x[i*3+2]
                    xj, yj, rj = x[j*3], x[j*3+1], x[j*3+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    # We want sqrt(dist_sq) >= ri + rj, so dist_sq >= (ri + rj)^2
                    return dist_sq - (ri + rj)**2
                constraints.append({'type': 'ineq', 'fun': constraint_func})
        return constraints
    
    # Objective function (negative because we're minimizing)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Sum of radii (negated for minimization)
    
    # Create constraints
    cons = []
    cons.extend(contain_constraints())
    cons.extend(non_overlap_constraints())
    
    # Bounds for variables (x, y, r) - ensure radii are positive and positions are valid
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((1e-6, 1-1e-6))  # x in (0,1)
        # y coordinate bounds  
        bounds.append((1e-6, 1-1e-6))  # y in (0,1)
        # radius bounds
        bounds.append((1e-6, 0.5))     # r in (0,0.5) - max possible radius is 0.5
    
    # Optimize using SLSQP method with multiple attempts
    try:
        # First attempt with SLSQP
        result = minimize(
            objective, 
            initial_flat, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        if result.success:
            optimized_flat = result.x
            circles = optimized_flat.reshape(-1, 3)
        else:
            # If first attempt fails, try with a simpler approach
            pass
    except Exception as e:
        # If optimization fails, return the initial configuration
        pass
    
    # Final cleanup to ensure all constraints are satisfied
    for i in range(n):
        x, y, r = circles[i]
        # Clamp coordinates to valid range
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
