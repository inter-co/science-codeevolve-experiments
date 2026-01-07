# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial grid placement + constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Create initial configuration using a hexagonal grid pattern
    def create_initial_layout():
        # Arrange circles in a roughly hexagonal pattern to get good initial placement
        rows = 6
        cols = 6
        circles = []
        
        # Calculate spacing based on number of circles
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Adjust for boundary constraints
                if x < 0.1: x = 0.1
                if x > 0.9: x = 0.9
                if y < 0.1: y = 0.1
                if y > 0.9: y = 0.9
                
                # Initial radius - small but feasible
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles.append([x, y, r])
                
            if len(circles) >= n:
                break
        
        # Fill remaining circles with smaller radii near boundaries
        while len(circles) < n:
            # Place near corners or edges
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = min(0.03, 0.5 * min(x, 1-x, y, 1-y))
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Constraint functions
    def constraint_radius(circle_params, idx):
        """Ensure circle stays within bounds"""
        x, y, r = circle_params
        return min(r, x - r, 1 - x - r, y - r, 1 - y - r)
    
    def constraint_overlap(circle_params1, circle_params2):
        """Ensure two circles don't overlap"""
        x1, y1, r1 = circle_params1
        x2, y2, r2 = circle_params2
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance - (r1 + r2)
    
    # Create initial configuration
    circles = create_initial_layout()
    
    # Flatten parameters for optimization: [x1, y1, r1, x2, y2, r2, ...]
    def flatten_circles(circles_array):
        return circles_array.flatten()
    
    def unflatten_circles(params):
        return params.reshape(-1, 3)
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        circles_array = unflatten_circles(params)
        return -np.sum(circles_array[:, 2])  # Negative because we want to maximize sum of radii
    
    # Constraint functions for scipy optimizer
    def constraint_func(params):
        circles_array = unflatten_circles(params)
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            x, y, r = circles_array[i]
            # Each circle must stay within unit square with its radius
            constraints.extend([
                x - r,           # x >= r
                1 - x - r,       # x <= 1 - r
                y - r,           # y >= r
                1 - y - r        # y <= 1 - r
            ])
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                constraints.append(dist - (r1 + r2))  # Distance >= sum of radii
                
        return np.array(constraints)
    
    # Set up bounds for each parameter
    bounds = []
    for i in range(n):
        # x, y, r bounds
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Convert initial guess to flat array
    initial_guess = flatten_circles(circles)
    
    # Use scipy's SLSQP optimizer which handles constraints well
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            final_circles = unflatten_circles(result.x)
            return final_circles
        else:
            # Return the initial configuration if optimization fails
            return circles
            
    except Exception as e:
        # Return the initial configuration if anything goes wrong
        return circles


# EVOLVE-BLOCK-END
