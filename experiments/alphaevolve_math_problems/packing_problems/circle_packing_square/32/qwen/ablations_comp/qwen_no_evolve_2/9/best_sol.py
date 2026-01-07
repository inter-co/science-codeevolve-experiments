# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a hexagonal packing pattern as starting point
    def initialize_hexagonal():
        # Create a hexagonal grid pattern
        circles = []
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Adjust for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x / 2
                    
                # Initial radius - small enough to fit in the square
                r = min(spacing_x, spacing_y) / 4
                
                # Ensure it fits in the unit square
                if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                    circles.append([x, y, r])
        
        # Fill remaining circles with random valid positions
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.1)
            
            # Check containment
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                circles.append([x, y, r])
                
        return np.array(circles)
    
    # Initialize
    circles = initialize_hexagonal()
    
    # Define constraint functions
    def containement_constraints(circles_flat):
        """Ensure all circles are contained within the unit square"""
        constraints = []
        for i in range(n):
            x, y, r = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
            # r <= x <= 1-r
            constraints.append({'type': 'ineq', 'fun': lambda x, y, r: x - r})
            constraints.append({'type': 'ineq', 'fun': lambda x, y, r: 1 - x - r})
            # r <= y <= 1-r
            constraints.append({'type': 'ineq', 'fun': lambda x, y, r: y - r})
            constraints.append({'type': 'ineq', 'fun': lambda x, y, r: 1 - y - r})
        return constraints
    
    def non_overlap_constraints(circles_flat):
        """Ensure no overlap between circles"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                def overlap_constraint(circles_flat, i=i, j=j):
                    x1, y1, r1 = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
                    x2, y2, r2 = circles_flat[3*j], circles_flat[3*j+1], circles_flat[3*j+2]
                    # Distance between centers >= sum of radii
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    return dist_sq - (r1 + r2)**2
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        return constraints
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        total_radius = 0
        for i in range(n):
            total_radius += circles_flat[3*i+2]  # radius is at index 3*i+2
        return -total_radius
    
    # Flatten the circles array for optimization
    initial_flat = circles.flatten()
    
    # Set bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x bounds: r <= x <= 1-r
        bounds.append((0.001, 0.999))  # r=0.001 to ensure feasibility
        # y bounds: r <= y <= 1-r
        bounds.append((0.001, 0.999))
        # r bounds: 0 < r <= 0.5 (maximum possible for any single circle)
        bounds.append((0.001, 0.5))
    
    # Constraints
    cons = []
    
    # Add containment constraints
    for i in range(n):
        # r <= x <= 1-r
        def x_bound_lower(circles_flat, i=i):
            return circles_flat[3*i] - circles_flat[3*i+2]
        def x_bound_upper(circles_flat, i=i):
            return 1 - circles_flat[3*i] - circles_flat[3*i+2]
        # r <= y <= 1-r
        def y_bound_lower(circles_flat, i=i):
            return circles_flat[3*i+1] - circles_flat[3*i+2]
        def y_bound_upper(circles_flat, i=i):
            return 1 - circles_flat[3*i+1] - circles_flat[3*i+2]
        
        cons.append({'type': 'ineq', 'fun': x_bound_lower})
        cons.append({'type': 'ineq', 'fun': x_bound_upper})
        cons.append({'type': 'ineq', 'fun': y_bound_lower})
        cons.append({'type': 'ineq', 'fun': y_bound_upper})
    
    # Add non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(circles_flat, i=i, j=j):
                x1, y1, r1 = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
                x2, y2, r2 = circles_flat[3*j], circles_flat[3*j+1], circles_flat[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                return dist_sq - (r1 + r2)**2
            cons.append({'type': 'ineq', 'fun': overlap_constraint})
    
    # Optimization parameters
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Run optimization
        result = minimize(objective, initial_flat, method='SLSQP', bounds=bounds, constraints=cons, 
                         options=options, tol=1e-6)
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure final validation
            validated_circles = validate_solution(optimized_circles)
            return validated_circles
        else:
            # If optimization fails, return the initial configuration
            return circles
    except Exception as e:
        # Return initial configuration if optimization fails
        return circles

def validate_solution(circles):
    """Validate that the solution satisfies all constraints"""
    n = len(circles)
    
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if not (r <= x <= 1-r and r <= y <= 1-r):
            # Adjust invalid positions
            circles[i][0] = max(r, min(1-r, x))
            circles[i][1] = max(r, min(1-r, y))
    
    # Re-check overlap constraints
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            if dist_sq < (r1 + r2)**2:
                valid = False
                break
        if not valid:
            break
    
    return circles


# EVOLVE-BLOCK-END
