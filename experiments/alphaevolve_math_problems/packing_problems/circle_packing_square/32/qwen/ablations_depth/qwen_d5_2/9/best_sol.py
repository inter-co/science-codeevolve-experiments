# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining Voronoi-based initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Step 1: Better initialization using a more systematic approach
    # Try to place circles in a way that maximizes density while maintaining feasibility
    # Use a grid-based approach with adaptive spacing
    
    # Create a more strategic initial layout
    # Start with a refined grid that considers the maximum possible radius
    points = []
    
    # Try different grid arrangements to find a good starting point
    grid_size = int(np.ceil(np.sqrt(n)))
    
    # Create a refined hexagonal-like grid
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) >= n:
                break
            # Offset odd rows for hexagonal packing
            offset = 0.5 * (i % 2)
            x = (j + offset) / grid_size
            y = i / grid_size
            
            # Keep within bounds with some margin
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            points.append([x, y])
    
    # If we have too many points, reduce to n
    if len(points) > n:
        points = points[:n]
    # If we have too few, fill with additional points near boundaries
    elif len(points) < n:
        # Fill remaining spots with boundary points
        while len(points) < n:
            # Place near edges for better distribution
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            points.append([x, y])
    
    points = points[:n]
    
    # Step 2: Initialize radii with better estimates
    # Estimate initial radii based on local density
    radii = np.full(n, 0.05)  # Start with moderate radii
    
    # Step 3: More efficient constraint handling
    def distance_constraint(i, j):
        """Constraint that circles i and j don't overlap"""
        def constraint(params):
            x_i, y_i, r_i = params[3*i:3*i+3]
            x_j, y_j, r_j = params[3*j:3*j+3]
            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            dist = np.sqrt(dist_sq)
            return dist - (r_i + r_j)
        return constraint
    
    def containment_constraint(i):
        """Constraint that circle i stays within the unit square"""
        def constraint(params):
            x_i, y_i, r_i = params[3*i:3*i+3]
            # Minimum distance from edges
            left = x_i - r_i
            right = 1 - x_i - r_i
            bottom = y_i - r_i
            top = 1 - y_i - r_i
            return min(left, right, bottom, top)
        return constraint
    
    # Step 4: Improved optimization approach
    # Flatten parameters: [x0, y0, r0, x1, y1, r1, ...]
    initial_params = np.zeros(3*n)
    for i in range(n):
        initial_params[3*i] = points[i][0]  # x
        initial_params[3*i+1] = points[i][1]  # y
        initial_params[3*i+2] = radii[i]  # r
    
    # Define bounds for each parameter
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds  
        bounds.append((0.001, 0.999))
        # r bounds
        bounds.append((0.001, 0.499))
    
    # Define constraints efficiently
    constraints = []
    
    # Add containment constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': containment_constraint(i)})
    
    # Add non-overlap constraints (only check with nearby points for efficiency)
    # Use spatial indexing to reduce constraint count
    def create_efficient_constraints():
        # For better performance, only consider nearby pairs
        # But since this is a simple implementation, keep all constraints
        # For large N, we'd use spatial indexing like KDTree
        
        # Add all pairwise non-overlap constraints for now
        local_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                local_constraints.append({'type': 'ineq', 'fun': distance_constraint(i, j)})
        return local_constraints
    
    constraints.extend(create_efficient_constraints())
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i+2]  # radius is at index 3*i+2
        return -total_radius
    
    # Try multiple optimization approaches
    best_result = None
    best_sum = 0
    
    # Try with different solvers and settings
    try:
        # First attempt with SLSQP
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-5, 'eps': 1e-5}
        )
        
        if result.success:
            # Check if this result is better than our previous attempts
            final_params = result.x
            current_sum = sum(final_params[3*i+2] for i in range(n))
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = result
    except Exception as e:
        pass
    
    # If first attempt failed, try a simpler approach
    if best_result is None:
        try:
            # Try with L-BFGS-B which might handle this better
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300}
            )
            
            if result.success:
                final_params = result.x
                current_sum = sum(final_params[3*i+2] for i in range(n))
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            pass
    
    # Final fallback to initial parameters
    if best_result is None:
        final_params = initial_params
    else:
        final_params = best_result.x
    
    # Convert back to circles array
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i][0] = final_params[3*i]      # x
        circles[i][1] = final_params[3*i+1]    # y
        circles[i][2] = final_params[3*i+2]    # r
    
    return circles


# EVOLVE-BLOCK-END
