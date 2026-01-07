# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import time
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a sophisticated hybrid approach combining multiple optimization strategies.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    n = 32
    
    # Strategy 1: Initialize with a better hexagonal grid
    # Create a more optimal hexagonal packing pattern
    circles = np.zeros((n, 3))
    
    # Use a 6x6 grid with offset rows for better packing
    rows = 6
    cols = 6
    
    # Calculate spacing to fit nicely in unit square
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    positions = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            # Offset every other row for hexagonal packing
            if i % 2 == 1:
                x += spacing_x * 0.5
            positions.append([x, y])
    
    # Take first n positions
    selected_positions = positions[:n]
    
    # Add small random perturbations to avoid symmetric solutions
    for i in range(n):
        x, y = selected_positions[i]
        x += random.uniform(-0.01, 0.01)
        y += random.uniform(-0.01, 0.01)
        # Ensure positions stay within bounds
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        selected_positions[i] = [x, y]
    
    # Set initial positions
    for i in range(n):
        circles[i, 0] = selected_positions[i][0]
        circles[i, 1] = selected_positions[i][1]
    
    # Initialize radii with a more informed approach
    # Start with a reasonable initial radius that allows for optimization
    initial_radius = 0.07
    for i in range(n):
        circles[i, 2] = initial_radius
    
    # Define constraint functions with improved numerical handling
    def distance_constraint(i, j):
        """Distance constraint: circles i and j must not overlap"""
        def constraint(x):
            x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
            x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
            
            # Use squared distance to avoid sqrt computation
            dx = x_i - x_j
            dy = y_i - y_j
            dist_sq = dx*dx + dy*dy
            # Return positive value when circles overlap (constraint violated)
            return dist_sq - (r_i + r_j)**2
        return constraint
    
    def containment_constraint(i):
        """Containment constraint: circle i must fit in unit square"""
        def constraint(x):
            x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
            # Distance to boundaries (positive when inside)
            left = x_i - r_i
            right = 1 - x_i - r_i
            bottom = y_i - r_i
            top = 1 - y_i - r_i
            return min(left, right, bottom, top)
        return constraint
    
    # Optimization function
    def objective(x):
        # We want to maximize sum of radii, so we minimize negative sum
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]
        return -total_radius
    
    # Constraints for optimization
    constraints = []
    
    # Add containment constraints for all circles
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': containment_constraint(i)})
    
    # Add distance constraints for all pairs
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': distance_constraint(i, j)})
    
    # Bounds for variables (x, y, r) - tighter bounds to prevent extreme values
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Initial guess vector
    x0 = np.array([circles[i][j] for i in range(n) for j in range(3)])
    
    # Try multiple optimization approaches for better convergence
    best_result = None
    best_sum = -np.inf
    
    # Method 1: SLSQP optimization
    try:
        result_slsqp = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            constraints=constraints, 
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )
        
        if result_slsqp.success:
            # Calculate sum of radii for this solution
            sum_radii = -result_slsqp.fun
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_result = result_slsqp
    except Exception as e:
        pass
    
    # Method 2: Trust-constr optimization (often better for constrained problems)
    try:
        result_trust = minimize(
            objective, 
            x0, 
            method='trust-constr', 
            constraints=constraints, 
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )
        
        if result_trust.success:
            sum_radii = -result_trust.fun
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_result = result_trust
    except Exception as e:
        pass
    
    # If optimization succeeded, use the best result
    if best_result is not None:
        final_circles = np.zeros((n, 3))
        for i in range(n):
            final_circles[i] = [best_result.x[3*i], best_result.x[3*i+1], best_result.x[3*i+2]]
        circles = final_circles
    else:
        # If all optimization failed, fall back to our initial configuration
        pass
    
    # Enhanced refinement phase with better local search
    # This is critical for pushing past local optima
    improved = True
    max_refinements = 100
    refinement_count = 0
    
    while improved and refinement_count < max_refinements:
        improved = False
        refinement_count += 1
        
        # Try to improve each circle individually
        for i in range(n):
            old_x, old_y, old_r = circles[i]
            
            # Try to increase radius while maintaining feasibility
            max_possible_radius = min(old_x, 1-old_x, old_y, 1-old_y)
            new_r = min(old_r + 0.003, max_possible_radius)
            
            # Check if we can actually increase the radius
            valid = True
            for j in range(n):
                if i != j:
                    dist_sq = (old_x - circles[j, 0])**2 + (old_y - circles[j, 1])**2
                    min_dist_sq = (new_r + circles[j, 2])**2
                    if dist_sq < min_dist_sq:
                        valid = False
                        break
            
            if valid and new_r > old_r:
                circles[i, 2] = new_r
                improved = True
                
            # Also try adjusting position to potentially increase radius further
            if not improved:
                # Try moving the circle slightly to see if we can increase radius
                best_x, best_y, best_r = old_x, old_y, old_r
                best_improvement = 0
                
                # Sample nearby positions
                for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                    for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                        test_x = max(0.001, min(0.999, old_x + dx))
                        test_y = max(0.001, min(0.999, old_y + dy))
                        
                        # Calculate max radius at this new position
                        max_radius = min(test_x, 1-test_x, test_y, 1-test_y)
                        
                        # Check constraints with other circles
                        valid_position = True
                        for j in range(n):
                            if i != j:
                                dist_sq = (test_x - circles[j, 0])**2 + (test_y - circles[j, 1])**2
                                min_dist_sq = (max_radius + circles[j, 2])**2
                                if dist_sq < min_dist_sq:
                                    valid_position = False
                                    break
                        
                        if valid_position:
                            # Try to increase radius at this position
                            test_r = max_radius
                            # Check if we can actually increase radius (this is a rough check)
                            if test_r > old_r:
                                if test_r - old_r > best_improvement:
                                    best_improvement = test_r - old_r
                                    best_x, best_y, best_r = test_x, test_y, test_r
                
                if best_improvement > 0:
                    circles[i, 0] = best_x
                    circles[i, 1] = best_y
                    circles[i, 2] = best_r
                    improved = True
    
    # Final validation and cleanup
    # Make sure all circles fit properly
    for i in range(n):
        x, y, r = circles[i]
        # Ensure containment
        r = min(r, x, 1-x, y, 1-y)
        circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
