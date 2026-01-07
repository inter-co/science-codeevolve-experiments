# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Improved initialization with better spatial distribution
    def initialize_better_layout():
        circles = []
        
        # Use a more systematic approach for grid layout
        # Try to get close to sqrt(32) ≈ 5.66 for square-like arrangement
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Adjust to ensure enough positions
        while rows * cols < n:
            rows += 1
            
        spacing_x = 0.95 / cols  # Slightly larger spacing to allow for better optimization
        spacing_y = 0.95 / rows
        
        # Create a more refined arrangement
        for i in range(rows):
            y = 0.025 + (i + 0.5) * spacing_y
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.025 + (j + 0.5) * spacing_x
                # Offset odd rows for better packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                # Start with a reasonable radius for dense packing
                r = min(spacing_x, spacing_y) * 0.4
                
                # Ensure circle fits in square
                if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                    circles.append([x, y, r])
        
        # Fill remaining spots with strategic random placement
        while len(circles) < n:
            # Use a better distribution strategy - avoid placing near edges too much
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            
            # Start with a larger range for radius to encourage growth
            r = 0.01 + np.random.random() * 0.12  # Broader range for better exploration
            
            # Check if it's valid (will be adjusted later)
            circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    # Generate initial configuration
    circles = initialize_better_layout()
    
    # More efficient constraint handling using vectorized operations where possible
    def create_constraints():
        """Create constraint functions for optimization - more efficient version"""
        constraints = []
        
        # Containment constraints: each circle must fit within [0,1]x[0,1]
        for i in range(n):
            # x >= r
            constraints.append({
                'type': 'ineq', 
                'fun': lambda c, i=i: c[3*i] - c[3*i+2]
            })
            # y >= r  
            constraints.append({
                'type': 'ineq', 
                'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]
            })
            # x <= 1-r
            constraints.append({
                'type': 'ineq', 
                'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]
            })
            # y <= 1-r
            constraints.append({
                'type': 'ineq', 
                'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]
            })
        
        # Overlap constraints: no two circles can overlap
        # Use more stable constraint formulation with small epsilon for numerical reasons
        epsilon = 1e-10
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda c, i=i, j=j: (c[3*i] - c[3*j])**2 + (c[3*i+1] - c[3*j+1])**2 - (c[3*i+2] + c[3*j+2])**2 - epsilon
                })
                
        return constraints
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Sum of all radii (indices 2,5,8,...)
    
    # Flatten initial circles for optimization
    initial_flat = circles.flatten()
    
    # Set up bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x coordinate bounds: [0.001, 0.999] 
        # y coordinate bounds: [0.001, 0.999]
        # r coordinate bounds: [0.001, 0.499]
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Collect all constraints
    cons = create_constraints()
    
    # Run optimization with multiple attempts and different methods for better results
    best_result = None
    best_sum = -float('inf')
    
    # Try multiple optimization strategies
    methods_to_try = ['SLSQP', 'trust-constr']
    attempts_per_method = 3
    
    for method in methods_to_try:
        for attempt in range(attempts_per_method):
            try:
                # Randomly perturb initial solution slightly for diversity
                perturbed_initial = initial_flat.copy()
                # Add noise to positions and radii differently
                for i in range(len(perturbed_initial)):
                    if i % 3 == 0:  # x coordinate
                        perturbed_initial[i] += np.random.normal(0, 0.01)
                        perturbed_initial[i] = max(0.001, min(0.999, perturbed_initial[i]))
                    elif i % 3 == 1:  # y coordinate
                        perturbed_initial[i] += np.random.normal(0, 0.01)
                        perturbed_initial[i] = max(0.001, min(0.999, perturbed_initial[i]))
                    else:  # radius
                        perturbed_initial[i] += np.random.normal(0, 0.005)
                        perturbed_initial[i] = max(0.001, min(0.499, perturbed_initial[i]))
                
                # Different optimization parameters for better convergence
                options = {
                    'maxiter': 500, 
                    'ftol': 1e-7, 
                    'eps': 1e-7
                }
                
                if method == 'trust-constr':
                    options['gtol'] = 1e-7
                    options['xtol'] = 1e-7
                
                result = minimize(
                    objective,
                    perturbed_initial,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options=options,
                    tol=1e-7
                )
                
                if result.success:
                    current_sum = -result.fun  # Convert back to positive sum
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception:
                continue
    
    # If optimization succeeded, use the best result; otherwise use initial
    if best_result is not None and best_result.success:
        optimized_circles = best_result.x.reshape(-1, 3)
    else:
        optimized_circles = circles
    
    # Final validation and cleanup
    final_circles = np.copy(optimized_circles)
    
    # Ensure all circles are valid
    for i in range(n):
        x, y, r = final_circles[i]
        # Clamp values to valid ranges
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        r = np.clip(r, 0.001, min(x, 1-x, y, 1-y))
        final_circles[i] = [x, y, r]
    
    return final_circles


# EVOLVE-BLOCK-END
