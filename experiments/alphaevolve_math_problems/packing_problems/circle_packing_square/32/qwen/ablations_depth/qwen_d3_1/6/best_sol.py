# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from scipy.spatial import KDTree
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Improved initialization using a more sophisticated approach
    def initialize_better_placement():
        # Start with a greedy approach to place circles
        circles = []
        
        # Place some circles near corners and edges to utilize space efficiently
        corner_positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        edge_positions = [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]
        
        # Place initial circles near corners and edges
        for x, y in corner_positions + edge_positions:
            if len(circles) < n:
                r = min(x, 1-x, y, 1-y) * 0.3
                circles.append([x, y, r])
        
        # Fill remaining positions with a more systematic approach
        attempts = 0
        max_attempts = 1000
        while len(circles) < n and attempts < max_attempts:
            # Try to place a circle at a random valid location
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            
            # Calculate max possible radius at this position
            max_r = min(x, 1-x, y, 1-y)
            
            if max_r <= 0:
                attempts += 1
                continue
                
            # Find the minimum distance to existing circles
            min_dist = float('inf')
            for cx, cy, cr in circles:
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                min_dist = min(min_dist, dist)
            
            # If we have at least one circle, use the minimum distance to determine radius
            if len(circles) > 0:
                r = min(max_r, min_dist * 0.4)  # Allow some overlap tolerance
            else:
                r = max_r * 0.3  # First circle gets smaller radius to allow room
            
            # Only accept valid circles
            if r > 0.001 and r <= max_r:
                circles.append([x, y, r])
            
            attempts += 1
            
        # If we still don't have enough circles, fill with random valid ones
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            max_r = min(x, 1-x, y, 1-y)
            r = max_r * 0.2 if max_r > 0 else 0.01
            if r > 0.001:
                circles.append([x, y, r])
                
        return np.array(circles)
    
    # Generate initial configuration
    circles = initialize_better_placement()
    
    # Define constraint functions with improved performance
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must be fully inside unit square
        def boundary_constraint(i):
            def con(x):
                idx = i * 3
                x_c, y_c, r = x[idx], x[idx+1], x[idx+2]
                # Return positive when constraint is satisfied
                return min(r, x_c - r, 1 - x_c - r, y_c - r, 1 - y_c - r)
            return con
        
        # Non-overlap constraints
        def overlap_constraint(i, j):
            def con(x):
                idx_i = i * 3
                idx_j = j * 3
                x_i, y_i, r_i = x[idx_i], x[idx_i+1], x[idx_i+2]
                x_j, y_j, r_j = x[idx_j], x[idx_j+1], x[idx_j+2]
                dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                # Return positive when constraint is satisfied (distance >= radii sum)
                return dist - (r_i + r_j)
            return con
            
        # Add boundary constraints for all circles
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
            
        # Add non-overlap constraints - only check with nearby circles for efficiency
        # Use spatial indexing for better performance
        points = np.array([[circles[i][0], circles[i][1]] for i in range(n)])
        tree = KDTree(points)
        
        # For each circle, only consider nearby circles (within 2x max radius)
        for i in range(n):
            # Get neighbors within a reasonable distance
            neighbors = tree.query_ball_point(points[i], 2.0, p=np.inf)
            for j in neighbors:
                if i < j:  # Avoid duplicate constraints
                    cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
                
        return cons
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[i*3 + 2]  # radius is third component
        return -total_radius
    
    # Flatten initial circles for optimization
    x0 = circles.flatten()
    
    # Get constraints
    constraints = get_constraints()
    
    # Set bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Run optimization with multiple strategies for better results
    try:
        # Try different optimization methods
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_sum = -float('inf')
        
        for method in methods:
            try:
                result = minimize(objective, x0, method=method, bounds=bounds, constraints=constraints, 
                                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6})
                
                if result.success:
                    # Calculate actual sum of radii
                    current_sum = -objective(result.x)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception:
                continue
                
        # Extract final solution
        if best_result is not None and best_result.success:
            optimized_circles = best_result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial solution if optimization fails
            return circles
    except Exception:
        # Return initial solution if optimization fails
        return circles


# EVOLVE-BLOCK-END
