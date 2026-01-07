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
    
    # Initialize using a physics-inspired approach with better spatial distribution
    def initialize_better_layout():
        circles = []
        
        # Create a more refined arrangement based on inspiration program's approach
        rows = 6
        cols = 6
        while rows * cols < n:
            cols += 1
        
        spacing_x = 0.9 / cols  # Slightly smaller spacing for better utilization
        spacing_y = 0.9 / rows
        
        # Create a more refined arrangement that avoids the simple hexagonal
        for i in range(rows):
            y = 0.05 + (i + 0.5) * spacing_y
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                if i % 2 == 1:  # Offset every other row
                    x += spacing_x * 0.5
                    
                # Start with a radius that's more appropriate for dense packing
                r = min(spacing_x, spacing_y) * 0.35
                
                # Ensure circle fits in square
                if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                    circles.append([x, y, r])
        
        # Fill remaining spots with circles placed strategically
        while len(circles) < n:
            # Use a more intelligent placement strategy
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            
            # Start with a larger range for radius to encourage growth
            r = 0.02 + np.random.random() * 0.08  # Larger range for better exploration
            
            # Check if it's valid (will be adjusted later)
            circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    # Generate initial configuration
    circles = initialize_better_layout()
    
    # Define constraint functions - simplified and more robust version
    def contain_constraints(circles_flat):
        """Ensure all circles are contained within unit square"""
        constraints = []
        
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
            
        return constraints
    
    def overlap_constraints(circles_flat):
        """Ensure no overlaps between circles"""
        constraints = []
        
        for i in range(n):
            for j in range(i+1, n):
                # For better numerical stability, we'll check if distance squared 
                # is greater than or equal to (r1 + r2)^2
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda c, i=i, j=j: (c[3*i] - c[3*j])**2 + (c[3*i+1] - c[3*j+1])**2 - (c[3*i+2] + c[3*j+2])**2
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
    cons = []
    cons.extend(contain_constraints(initial_flat))
    cons.extend(overlap_constraints(initial_flat))
    
    # Run optimization with multiple attempts for better results
    best_result = None
    best_sum = -float('inf')
    
    for attempt in range(3):  # Try multiple optimization runs
        try:
            # Randomly perturb initial solution slightly for diversity
            perturbed_initial = initial_flat.copy()
            for i in range(len(perturbed_initial)):
                if i % 3 == 2:  # Only perturb radii
                    perturbed_initial[i] += np.random.normal(0, 0.005)
                    perturbed_initial[i] = max(0.001, min(0.499, perturbed_initial[i]))
            
            result = minimize(
                objective,
                perturbed_initial,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6},
                tol=1e-6
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
