# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with mathematical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Improved initialization using a more systematic approach
    def initialize_better_layout():
        # Start with a better hexagonal packing approximation
        circles = []
        
        # Try to arrange in a more optimal pattern
        # Using a grid-like approach but with some randomness to avoid local minima
        rows = 6
        cols = 6
        
        # Calculate spacing based on number of circles needed
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / cols
        
        # Create a more refined hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for better packing
                x_offset = 0.05 + (j if i % 2 == 0 else j + 0.5) * spacing_x
                y_offset = 0.05 + i * spacing_y
                
                # Add some jitter to avoid perfect patterns that might cause issues
                x_offset += np.random.normal(0, 0.005)
                y_offset += np.random.normal(0, 0.005)
                
                # Ensure within bounds
                if 0 <= x_offset <= 1 and 0 <= y_offset <= 1:
                    # Radius should be limited by available space
                    max_radius = min(x_offset, 1-x_offset, y_offset, 1-y_offset)
                    radius = min(max_radius * 0.3, 0.15)
                    circles.append([x_offset, y_offset, radius])
        
        # Fill remaining circles with random positions but proper radii
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Make sure we don't place too close to existing circles
            min_dist = 0.1
            valid = True
            for cx, cy, _ in circles:
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < min_dist:
                    valid = False
                    break
            if valid:
                max_radius = min(x, 1-x, y, 1-y)
                radius = min(max_radius * 0.3, 0.1)
                circles.append([x, y, radius])
            
        return np.array(circles[:n])
    
    # More efficient constraint generation using vectorized operations
    def create_constraints_vectorized():
        # Create constraints for boundary conditions and non-overlap
        constraints = []
        
        # Boundary constraints: each circle must stay within square with its radius
        for i in range(n):
            # x - r >= 0
            def bound_x_l(x, i=i):
                return x[3*i] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': bound_x_l})
            
            # y - r >= 0
            def bound_y_l(x, i=i):
                return x[3*i+1] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': bound_y_l})
            
            # 1 - x - r >= 0
            def bound_x_u(x, i=i):
                return 1 - x[3*i] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': bound_x_u})
            
            # 1 - y - r >= 0
            def bound_y_u(x, i=i):
                return 1 - x[3*i+1] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': bound_y_u})
        
        # Non-overlap constraints - use more efficient approach
        for i, j in combinations(range(n), 2):
            def overlap_constraint(x, i=i, j=j):
                dx = x[3*i] - x[3*j]
                dy = x[3*i+1] - x[3*j+1]
                distance = math.sqrt(dx*dx + dy*dy)
                return distance - (x[3*i+2] + x[3*j+2])
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
            
        return constraints
    
    # Objective function to maximize (negative because minimize)
    def objective(x):
        return -np.sum(x[2::3])  # Sum of all radii (indices 2, 5, 8, ...)

    # Initial guess
    initial_circles = initialize_better_layout()
    initial_guess = initial_circles.flatten()
    
    # Set bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Get constraints
    constraints = create_constraints_vectorized()
    
    # Optimization with multiple attempts for better results
    best_result = None
    best_sum = -np.inf
    
    # Try different optimization methods
    methods = ['SLSQP', 'trust-constr']
    
    for method in methods:
        try:
            # For better results, also try with different starting points
            result = minimize(
                objective,
                initial_guess,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception as e:
            continue
    
    # If we didn't get a good result, return the initial configuration
    if best_result is None or not best_result.success:
        return initial_circles
    
    # Return the best result
    final_circles = best_result.x.reshape(-1, 3)
    return final_circles


# EVOLVE-BLOCK-END
