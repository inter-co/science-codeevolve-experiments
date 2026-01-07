# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Step 1: Better initialization using a more systematic approach
    # Start with a more uniform distribution that allows larger radii
    # Create a grid pattern but with optimized spacing
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Generate points in a more optimal pattern
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            # Use spiral-like pattern to avoid regular grid artifacts
            offset = 0.5 * (i % 2)  # staggered rows
            x = (j + offset) / cols
            y = i / rows
            
            # Ensure points are well within the unit square with some margin
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            points.append([x, y])
    
    # Adjust to exactly n points
    points = points[:n]
    
    # Step 2: Initialize with better starting radii
    # Start with radii that are reasonable given the initial positions
    radii = np.full(n, 0.05)
    
    # Step 3: More efficient constraint handling
    def distance_constraint(i, j, params):
        """Constraint that circles i and j don't overlap"""
        x_i, y_i, r_i = params[3*i:3*i+3]
        x_j, y_j, r_j = params[3*j:3*j+3]
        dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
        return dist - (r_i + r_j)
    
    def containment_constraint(i, params):
        """Constraint that circle i stays within the unit square"""
        x_i, y_i, r_i = params[3*i:3*i+3]
        # Minimum distance from edges
        left = x_i - r_i
        right = 1 - x_i - r_i
        bottom = y_i - r_i
        top = 1 - y_i - r_i
        return min(left, right, bottom, top)
    
    # Step 4: Optimized optimization approach
    # Flatten parameters: [x0, y0, r0, x1, y1, r1, ...]
    initial_params = np.zeros(3*n)
    for i in range(n):
        initial_params[3*i] = points[i][0]  # x
        initial_params[3*i+1] = points[i][1]  # y
        initial_params[3*i+2] = radii[i]  # r
    
    # Define bounds for each parameter
    bounds = []
    for i in range(n):
        # x bounds (slightly inside unit square)
        bounds.append((0.001, 0.999))
        # y bounds  
        bounds.append((0.001, 0.999))
        # r bounds (reasonable upper bound)
        bounds.append((0.001, 0.499))
    
    # Precompute all pairwise indices for constraints
    constraint_pairs = list(combinations(range(n), 2))
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i+2]  # radius is at index 3*i+2
        return -total_radius
    
    # Constraint function for all constraints
    def all_constraints(params):
        # Check containment constraints
        containment_violations = []
        for i in range(n):
            violation = containment_constraint(i, params)
            containment_violations.append(violation)
        
        # Check non-overlap constraints
        overlap_violations = []
        for i, j in constraint_pairs:
            violation = distance_constraint(i, j, params)
            overlap_violations.append(violation)
        
        # Return all violations (positive means constraint violated)
        return np.concatenate([containment_violations, overlap_violations])
    
    # Perform optimization with better method selection
    try:
        # Try different optimization methods
        methods_to_try = ['SLSQP', 'trust-constr']
        best_result = None
        best_sum = 0
        
        for method in methods_to_try:
            try:
                result = minimize(
                    objective,
                    initial_params.copy(),
                    method=method,
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': lambda p: all_constraints(p)}],
                    options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
                )
                
                if result.success:
                    # Calculate actual sum of radii
                    current_sum = -objective(result.x)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception:
                continue
        
        # If no method worked, use the initial configuration
        if best_result is None or not best_result.success:
            final_params = initial_params
        else:
            final_params = best_result.x
            
    except Exception as e:
        # If optimization fails completely, return initial configuration
        final_params = initial_params
    
    # Convert back to circles array
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i][0] = final_params[3*i]      # x
        circles[i][1] = final_params[3*i+1]    # y
        circles[i][2] = final_params[3*i+2]    # r
    
    return circles


# EVOLVE-BLOCK-END
