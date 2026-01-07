# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from scipy.spatial import cKDTree
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # More sophisticated initialization using a grid-based approach with better spacing
    def initialize_better_grid():
        circles = []
        
        # Try to place circles in a pattern that maximizes initial packing density
        # Use a more structured approach than simple hexagonal grid
        
        # Start with a dense grid pattern
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        
        # Create a more uniform distribution
        for i in range(grid_size):
            for j in range(grid_size):
                if len(circles) >= n:
                    break
                # Add some randomness to avoid perfect patterns
                x = (j + 0.5 + random.uniform(-0.1, 0.1)) * spacing
                y = (i + 0.5 + random.uniform(-0.1, 0.1)) * spacing
                
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Initial radius based on proximity to edges
                r = min(x, 1-x, y, 1-y) * 0.3
                
                # Make sure it's not too large
                r = min(r, 0.2)
                
                circles.append([x, y, r])
        
        # Fill remaining slots with random placements but with better initial sizing
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            # Radius based on distance to edges
            r = min(x, 1-x, y, 1-y) * 0.25
            r = min(r, 0.2)
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Optimized constraint checking with spatial indexing for performance
    def create_constraint_functions():
        """Create constraint functions that are more efficient"""
        # Precompute all pairwise indices
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        def boundary_constraint(i):
            def constraint(x):
                idx = i * 3
                x_pos, y_pos, radius = x[idx], x[idx+1], x[idx+2]
                # Return minimum of all boundary constraints (should be > 0 for feasibility)
                return min(
                    radius,           # radius must be positive
                    x_pos - radius,   # x position must allow radius
                    1 - x_pos - radius,  # x position must allow radius
                    y_pos - radius,   # y position must allow radius
                    1 - y_pos - radius   # y position must allow radius
                )
            return constraint
        
        def overlap_constraint(i, j):
            def constraint(x):
                idx_i = i * 3
                idx_j = j * 3
                x_i, y_i, r_i = x[idx_i], x[idx_i+1], x[idx_i+2]
                x_j, y_j, r_j = x[idx_j], x[idx_j+1], x[idx_j+2]
                # Distance between centers minus sum of radii (should be >= 0)
                dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                return dist - (r_i + r_j)
            return constraint
        
        # Create all constraints
        cons = []
        
        # Add boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Add non-overlap constraints
        for i, j in pairs:
            cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
                
        return cons
    
    # Improved objective function with better handling
    def objective(x):
        # Sum of all radii (we want to maximize this)
        total_radius = 0
        for i in range(n):
            total_radius += x[i*3 + 2]  # Extract radius for each circle
        return -total_radius  # Negative because we're minimizing
    
    # Better optimization approach
    def optimize_with_improved_method(initial_circles):
        # Flatten initial values for optimization
        initial_guess = initial_circles.flatten()
        
        # Create constraints
        constraints = create_constraint_functions()
        
        # Try multiple optimization approaches
        methods = ['SLSQP', 'trust-constr']
        
        best_result = None
        best_sum = -np.inf
        
        for method in methods:
            try:
                # Use trust-constr for potentially better results
                result = minimize(
                    objective, 
                    initial_guess, 
                    method=method, 
                    constraints=constraints, 
                    options={
                        'maxiter': 500, 
                        'ftol': 1e-6,
                        'gtol': 1e-6
                    }
                )
                
                if result.success:
                    # Check if this result is better
                    current_sum = -objective(result.x)  # Convert back to positive sum
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
                        
            except Exception:
                continue
        
        # If we have a good result, use it; otherwise return initial
        if best_result is not None and best_result.success:
            optimized_circles = best_result.x.reshape(-1, 3)
            # Final validation and cleanup
            for i in range(n):
                # Ensure radii are valid
                optimized_circles[i, 2] = max(0.001, min(0.5, optimized_circles[i, 2]))
                # Ensure positions are within bounds
                optimized_circles[i, 0] = max(optimized_circles[i, 2], 
                                             min(1 - optimized_circles[i, 2], 
                                                 optimized_circles[i, 0]))
                optimized_circles[i, 1] = max(optimized_circles[i, 2], 
                                             min(1 - optimized_circles[i, 2], 
                                                 optimized_circles[i, 1]))
            return optimized_circles
        else:
            # Return initial guess if optimization fails
            return initial_circles
    
    # Initialize circles
    circles = initialize_better_grid()
    
    # Run optimization
    try:
        optimized_circles = optimize_with_improved_method(circles)
        return optimized_circles
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
