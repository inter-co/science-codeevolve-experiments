# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach combining geometric initialization with advanced optimization.
    """
    n = 32
    
    # Create initial configuration using a more sophisticated approach
    def create_initial_placement():
        # Use a combination of grid-based placement and strategic positioning
        circles = []
        
        # Place circles in a more efficient grid pattern
        # Try to use approximately sqrt(32) ≈ 5.6 as rows/columns
        rows = 6
        cols = 6
        
        # Use golden ratio based spacing for better distribution
        phi = (1 + math.sqrt(5)) / 2
        spacing_x = 0.15
        spacing_y = 0.15 * math.sqrt(3) / 2
        
        # Generate points in hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, 0.05])  # Initial radius guess
        
        # Add additional circles in strategic positions
        extra_positions = [
            (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
            (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),
            (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)
        ]
        
        for x, y in extra_positions:
            if len(circles) < n:
                circles.append([x, y, 0.05])
        
        # Fill remaining slots if needed with random positions near center
        while len(circles) < n:
            x = 0.3 + random.random() * 0.4
            y = 0.3 + random.random() * 0.4
            circles.append([x, y, 0.05])
            
        return np.array(circles[:n])
    
    # More efficient constraint checking
    def compute_constraints(vars):
        """Compute all constraints efficiently"""
        n_circles = len(vars) // 3
        constraints = []
        
        # Boundary constraints
        for i in range(n_circles):
            x, y, r = vars[3*i:3*i+3]
            # Each circle must fit within bounds (ensure positive distances)
            constraints.extend([
                x - r,           # left boundary
                1 - x - r,       # right boundary
                y - r,           # bottom boundary
                1 - y - r        # top boundary
            ])
        
        # Non-overlap constraints - only check relevant pairs
        # Use spatial indexing for efficiency (but for simplicity, we'll do direct check)
        for i, j in combinations(range(n_circles), 2):
            x1, y1, r1 = vars[3*i:3*i+3]
            x2, y2, r2 = vars[3*j:3*j+3]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            # Distance squared should be >= (r1 + r2)^2
            constraints.append(dist_sq - (r1 + r2)**2)
                
        return np.array(constraints)
    
    # Objective function to maximize (negative because minimize)
    def objective(vars):
        total_radius = 0
        for i in range(len(vars) // 3):
            total_radius += vars[3*i + 2]  # radius is third component
        return -total_radius
    
    # Improved constraint handling with better error checking
    def constraint_func(vars):
        return compute_constraints(vars)
    
    # Create initial placement
    circles = create_initial_placement()
    
    # Flatten into variables [x1, y1, r1, x2, y2, r2, ...]
    initial_vars = circles.flatten()
    
    # Set bounds for variables (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # r < 0.5 to allow some margin
    
    # Try multiple optimization approaches for better results
    best_result = None
    best_sum = 0
    
    # Try with different methods
    methods = ['SLSQP', 'trust-constr']
    
    for method in methods:
        try:
            # Use scipy minimize with constraints
            result = minimize(
                objective,
                initial_vars,
                method=method,
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 1500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                # Calculate sum of radii for this result
                total_radius = -result.fun
                if total_radius > best_sum:
                    best_sum = total_radius
                    best_result = result
                    
        except Exception as e:
            continue
    
    # If we found a good result, use it; otherwise fall back to initial
    if best_result is not None and best_result.success:
        optimized_vars = best_result.x
        circles = optimized_vars.reshape(-1, 3)
    else:
        # If optimization fails, try a more robust approach
        # Apply a simple local optimization to improve initial placement
        try:
            # Simple gradient descent approach for refinement
            vars = initial_vars.copy()
            learning_rate = 0.01
            
            # Run a few iterations of simple optimization
            for _ in range(100):
                # Compute gradients numerically or use simpler approach
                # For now, just do coordinate descent on radii
                new_vars = vars.copy()
                for i in range(n):
                    x, y, r = vars[3*i:3*i+3]
                    # Try slightly increasing radius if possible
                    test_r = min(0.499, r * 1.05)
                    # Check if this would violate constraints
                    valid = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = vars[3*j:3*j+3]
                            dist_sq = (x - x2)**2 + (y - y2)**2
                            if dist_sq < (test_r + r2)**2:
                                valid = False
                                break
                    
                    if valid and test_r > r:
                        new_vars[3*i + 2] = test_r
                
                vars = new_vars
                
            circles = vars.reshape(-1, 3)
            
        except Exception:
            # Final fallback to initial placement
            pass
    
    # Final validation and adjustment
    # Make sure all circles are valid
    for i in range(n):
        x, y, r = circles[i]
        # Ensure valid bounds
        circles[i][0] = max(r, min(1-r, x))
        circles[i][1] = max(r, min(1-r, y))
        circles[i][2] = max(0.001, min(0.499, r))
    
    return circles


# EVOLVE-BLOCK-END
