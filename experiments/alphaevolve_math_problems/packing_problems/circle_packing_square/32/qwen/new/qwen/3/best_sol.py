# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
from scipy.spatial import cKDTree

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, spatial indexing, and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # More sophisticated initialization with better spatial distribution
    def initialize_better():
        # Start with hexagonal pattern then refine
        circles = []
        
        # Hexagonal grid with some randomness to avoid regular patterns
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Add some randomness to avoid perfect grid
                x += (random.random() - 0.5) * spacing_x * 0.3
                y += (random.random() - 0.5) * spacing_y * 0.3
                
                # Adjust for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x / 2
                    
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Initial radius - small enough to fit
                r = min(spacing_x, spacing_y) / 4
                circles.append([x, y, r])
        
        # Fill remaining slots with random positions
        while len(circles) < n:
            x = 0.05 + random.random() * 0.9
            y = 0.05 + random.random() * 0.9
            r = 0.02 + random.random() * 0.1  # Smaller range for better packing
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    def initialize_grid():
        # Grid initialization with varied radii
        circles = []
        grid_size = int(math.ceil(math.sqrt(n)))
        cell_width = 1.0 / grid_size
        
        for i in range(n):
            row = i // grid_size
            col = i % grid_size
            x = (col + 0.5) * cell_width
            y = (row + 0.5) * cell_width
            # Set radius to a fraction of cell width - slightly smaller for better packing
            r = min(0.15 * cell_width, 0.25)
            circles.append([x, y, r])
        return np.array(circles)
    
    def initialize_random():
        # Random initialization with better spatial awareness
        circles = []
        for i in range(n):
            # Place in a way that avoids extreme clustering
            x = 0.05 + random.random() * 0.9
            y = 0.05 + random.random() * 0.9
            r = 0.02 + random.random() * 0.1  # Reasonable range for radius
            circles.append([x, y, r])
        return np.array(circles)
    
    # Improved constraint functions with better handling
    def get_constraints(x0):
        cons = []
        
        # Boundary constraints: radius must be <= distance to edges
        def boundary_constraint(i):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                # For containment: r_i <= x_i, x_i <= 1-r_i, r_i <= y_i, y_i <= 1-r_i
                return min(x_i - r_i, 1 - x_i - r_i, y_i - r_i, 1 - y_i - r_i)
            return {'type': 'ineq', 'fun': constraint}
        
        # Non-overlap constraints with spatial indexing for efficiency
        def overlap_constraint(i, j):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                return dist - (r_i + r_j)
            return {'type': 'ineq', 'fun': constraint}
        
        # Add boundary constraints
        for i in range(n):
            cons.append(boundary_constraint(i))
        
        # Add non-overlap constraints with more careful approach
        # Instead of pruning, we'll compute all constraints but with smarter ordering
        # Create spatial index to speed up constraint checking
        points = [(x0[3*i], x0[3*i+1]) for i in range(n)]
        tree = cKDTree(points)
        
        # For each point, find neighbors within a reasonable range
        # But still check all pairs to be thorough for this small problem
        for i in range(n):
            for j in range(i+1, n):
                cons.append(overlap_constraint(i, j))
                
        return cons
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]  # radius is third component
        return -total_radius
    
    # Better optimization with multiple restarts and better parameters
    def optimize_with_restarts(initial_guesses):
        best_result = None
        best_sum = 0
        
        # Try multiple optimization runs with different settings
        for i, x0 in enumerate(initial_guesses):
            try:
                # Different optimization approaches
                methods = ['SLSQP', 'trust-constr']
                bounds = []
                for j in range(n):
                    bounds.extend([(0, 1), (0, 1), (0, 0.5)])
                
                # Get constraints for this specific initial guess
                constraints = get_constraints(x0)
                
                # Try different optimization methods
                for method in methods:
                    try:
                        result = minimize(
                            objective, 
                            x0, 
                            method=method, 
                            bounds=bounds, 
                            constraints=constraints,
                            options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
                        )
                        
                        if result.success:
                            total_radius = -result.fun
                            if total_radius > best_sum:
                                best_sum = total_radius
                                best_result = result
                                break  # Found a good solution, move to next initial guess
                    except:
                        continue
                        
            except Exception as e:
                continue
        
        return best_result, best_sum
    
    # Try different initialization strategies and pick the best
    initializations = [
        initialize_better,
        initialize_grid,
        initialize_random
    ]
    
    # Generate multiple initial guesses
    initial_guesses = []
    for init_func in initializations:
        try:
            circles = init_func()
            x0 = []
            for circle in circles:
                x0.extend(circle)
            initial_guesses.append(x0)
        except:
            continue
    
    # Run optimization
    best_result, best_sum = optimize_with_restarts(initial_guesses)
    
    # If we found a good result, return it; otherwise use the first initialization
    if best_result is not None:
        # Extract optimized results
        optimized_circles = []
        for i in range(n):
            x = best_result.x[3*i]
            y = best_result.x[3*i+1]
            r = best_result.x[3*i+2]
            optimized_circles.append([x, y, r])
        return np.array(optimized_circles)
    else:
        # Fallback to better initialization if all optimizations fail
        circles = initialize_better()
        return circles


# EVOLVE-BLOCK-END
