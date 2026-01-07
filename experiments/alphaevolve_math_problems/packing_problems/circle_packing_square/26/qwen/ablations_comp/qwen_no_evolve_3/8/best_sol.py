# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining initial heuristic placement with constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a good heuristic placement based on hexagonal packing
    def initialize_placement():
        # Create a hexagonal lattice pattern
        circles = []
        # Try to arrange in approximately 5 rows and 5 columns
        rows = 5
        cols = 5
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Hexagonal offset for even rows
        for i in range(rows):
            y = (i + 1) * spacing_y
            for j in range(cols):
                if i % 2 == 0:
                    x = (j + 1) * spacing_x
                else:
                    x = (j + 1.5) * spacing_x
                
                # Ensure we don't exceed 26 circles
                if len(circles) >= n:
                    break
                    
                # Initial radius based on available space
                max_radius = min(x, 1-x, y, 1-y)
                # Reduce slightly to allow for optimization
                radius = max_radius * 0.8
                
                circles.append([x, y, radius])
                
        # Fill remaining slots with random placements near edges
        while len(circles) < n:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            # Radius based on distance to boundaries
            max_radius = min(x, 1-x, y, 1-y)
            radius = max_radius * np.random.uniform(0.5, 0.9)
            circles.append([x, y, radius])
            
        return np.array(circles)
    
    # Define the objective function to maximize sum of radii
    def objective(params):
        # params contains [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
        total_radius = 0
        circles = params.reshape(-1, 3)
        
        # Sum of all radii (we want to maximize this)
        for i in range(n):
            total_radius += circles[i, 2]
        
        return -total_radius  # Negative because we're minimizing
    
    # Define constraints
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must fit entirely in the unit square
        def boundary_constraint(params):
            circles = params.reshape(-1, 3)
            result = []
            for i in range(n):
                x, y, r = circles[i]
                # r <= x <= 1-r and r <= y <= 1-r
                result.append(x - r)  # x - r >= 0
                result.append(1 - x - r)  # 1 - x - r >= 0
                result.append(y - r)  # y - r >= 0
                result.append(1 - y - r)  # 1 - y - r >= 0
            return np.array(result)
        
        cons.append({'type': 'ineq', 'fun': boundary_constraint})
        
        # Non-overlap constraints
        def overlap_constraint(params):
            circles = params.reshape(-1, 3)
            result = []
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    # Distance between centers >= sum of radii
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist_sq = (r1 + r2)**2
                    # We want dist_sq >= min_dist_sq, so we return dist_sq - min_dist_sq
                    result.append(dist_sq - min_dist_sq)
            return np.array(result)
        
        cons.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return cons
    
    # Set up initial guess
    initial_circles = initialize_placement()
    initial_params = initial_circles.flatten()
    
    # Set bounds for parameters (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))  # Avoid exact boundaries for numerical stability
        # y bounds  
        bounds.append((0.001, 0.999))
        # r bounds (must be positive and small enough to fit)
        bounds.append((0.001, 0.499))
    
    # Get constraints
    constraints = get_constraints()
    
    # Run optimization
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
            final_circles = result.x.reshape(-1, 3)
        else:
            # Fallback to initial placement if optimization fails
            final_circles = initial_circles
            
    except Exception as e:
        # Fallback to initial placement if optimization errors
        final_circles = initial_circles
    
    # Final adjustment to ensure all constraints are met
    # Make sure all circles fit within the unit square
    for i in range(n):
        x, y, r = final_circles[i]
        # Clamp values to valid ranges
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        final_circles[i] = [x, y, r]
    
    return final_circles


# EVOLVE-BLOCK-END
