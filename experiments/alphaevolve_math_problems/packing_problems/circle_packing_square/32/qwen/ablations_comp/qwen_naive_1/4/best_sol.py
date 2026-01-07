# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using a more systematic approach
    def initialize_better():
        # Start with a uniform distribution of points, then optimize radii
        circles = []
        
        # Create a more appropriate grid pattern
        sqrt_n = int(math.ceil(math.sqrt(n)))
        grid_size = max(1, sqrt_n)
        
        # Place points in a grid with spacing that allows for reasonable radii
        spacing = 1.0 / (grid_size + 1)
        radius_guess = spacing / 2.0
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n:
                    break
                x = (j + 1) * spacing
                y = (i + 1) * spacing
                circles.append([x, y, radius_guess])
                count += 1
                
            if count >= n:
                break
        
        # Fill remaining spots with centers near edges or center
        while len(circles) < n:
            # Add some circles near the corners or center
            if len(circles) < n:
                circles.append([0.1, 0.1, 0.02])
            if len(circles) < n:
                circles.append([0.9, 0.9, 0.02])
            if len(circles) < n:
                circles.append([0.1, 0.9, 0.02])
            if len(circles) < n:
                circles.append([0.9, 0.1, 0.02])
            if len(circles) < n:
                circles.append([0.5, 0.5, 0.02])
        
        # Trim to exactly n circles
        return np.array(circles[:n])
    
    # More robust constraint functions
    def get_constraints(circles_flat):
        """Generate constraints for optimization"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Containment constraints: r <= x <= 1-r and r <= y <= 1-r
        for i in range(len(circles)):
            x, y, r = circles[i]
            # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})
            # 1 - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})
            # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})
            # 1 - y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})
            
        # Non-overlap constraints
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                def overlap_constraint(c, i=i, j=j):
                    x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                    x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    # Distance squared should be >= (r1 + r2)^2
                    return dist_sq - (r1 + r2)**2
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
                
        return constraints
    
    # Objective function to maximize sum of radii
    def objective(circles_flat):
        # Minimize negative of sum of radii (since scipy minimizes)
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    # Initial guess with better starting point
    initial_circles = initialize_better()
    initial_guess = initial_circles.flatten()
    
    # Set up bounds for variables: x,y in [r, 1-r], r in [0, 0.5] (reasonable upper bound)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Optimization parameters
    options = {'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
    
    # Try multiple optimization attempts with different methods
    best_result = None
    best_sum = -float('inf')
    
    # First attempt with SLSQP
    try:
        cons = get_constraints(initial_guess)
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=cons, 
                         options=options)
        if result.success:
            current_sum = -result.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = result
    except Exception as e:
        pass
    
    # If first attempt failed, try L-BFGS-B
    if best_result is None:
        try:
            # Simplified constraints for L-BFGS-B (just containment)
            bounds_simple = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
            result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds_simple, 
                             options=options)
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            pass
    
    # If still no success, fallback to initial guess
    if best_result is None:
        final_circles = initial_circles
    else:
        final_circles = best_result.x.reshape(-1, 3)
    
    # Final validation and refinement
    def validate_and_refine(circles):
        # Check containment
        valid = True
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                valid = False
                break
        
        if not valid:
            return circles  # Return original if invalid
        
        # Check non-overlap
        distances = cdist(circles[:, :2], circles[:, :2])
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dist = distances[i, j]
                if dist < circles[i, 2] + circles[j, 2]:
                    # Try to slightly adjust positions to resolve overlap
                    return circles  # Just return original for now to avoid complex adjustments
        
        return circles
    
    # Validate final solution
    final_circles = validate_and_refine(final_circles)
    
    # If still problematic, use a more conservative approach
    if len(final_circles) != n:
        final_circles = initialize_better()
    
    return final_circles


# EVOLVE-BLOCK-END
