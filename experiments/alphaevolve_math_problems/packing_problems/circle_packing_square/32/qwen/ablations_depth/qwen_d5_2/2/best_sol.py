# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial hexagonal grid placement with scipy optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Create initial configuration using hexagonal packing pattern
    # This provides a good starting point that's likely to be feasible
    def create_hexagonal_initial():
        # Arrange circles in a hexagonal pattern
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset odd rows
                x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Ensure we're within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius based on available space
                    r = min(x, 1-x, y, 1-y) * 0.4
                    if r > 0:
                        circles.append([x, y, r])
        
        # Fill remaining positions if needed
        while len(circles) < n:
            # Place randomly in valid positions
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = min(x, 1-x, y, 1-y) * 0.3
            if r > 0:
                circles.append([x, y, r])
                
        return np.array(circles[:n])
    
    # Constraint functions for optimization
    def constraint_radius(circle_data):
        """Ensure all circles fit within the unit square"""
        x, y, r = circle_data
        return min(r, x-r, 1-x-r, y-r, 1-y-r)
    
    def constraint_overlap(circle_i, circle_j):
        """Ensure two circles don't overlap"""
        x1, y1, r1 = circle_i
        x2, y2, r2 = circle_j
        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        return distance - (r1 + r2)
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        total_radius = sum(params[2::3])  # Extract all radii
        return -total_radius
    
    # Constraints for optimization
    def radius_constraint(i):
        def constraint(params):
            x = params[i*3]
            y = params[i*3+1]
            r = params[i*3+2]
            return min(r, x-r, 1-x-r, y-r, 1-y-r)
        return constraint
    
    def overlap_constraint(i, j):
        def constraint(params):
            x1, y1, r1 = params[i*3], params[i*3+1], params[i*3+2]
            x2, y2, r2 = params[j*3], params[j*3+1], params[j*3+2]
            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            return distance - (r1 + r2)
        return constraint
    
    # Get initial configuration
    circles = create_hexagonal_initial()
    
    # Flatten parameters for optimization: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = circles.flatten()
    
    # Build constraints
    constraints = []
    
    # Add radius constraints for each circle
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': radius_constraint(i)})
    
    # Add overlap constraints for each pair of circles
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
    
    # Bounds for parameters (x, y, r) - ensure radii are positive
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Optimize using SLSQP method
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_params = result.x
            circles = optimized_params.reshape(-1, 3)
        else:
            # If optimization fails, return initial configuration
            pass
            
    except Exception as e:
        # If optimization fails, return initial configuration
        pass
    
    # Final refinement step: ensure all constraints are met and slightly adjust
    # Check final configuration and make small adjustments if needed
    final_circles = circles.copy()
    
    # Apply final constraint enforcement
    for i in range(n):
        x, y, r = final_circles[i]
        # Enforce containment constraints
        r = min(r, x, 1-x, y, 1-y)
        if r <= 0:
            # Revert to safe value
            r = 0.01
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
        final_circles[i] = [x, y, r]
    
    return final_circles


# EVOLVE-BLOCK-END
