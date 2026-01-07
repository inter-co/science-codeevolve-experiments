# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using a hexagonal grid pattern for good starting configuration
    def initialize_hexagonal_pattern():
        circles = np.zeros((n, 3))
        
        # Create a hexagonal grid pattern
        rows = 6
        cols = 6
        spacing_x = 0.15
        spacing_y = 0.15
        offset = 0.05
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = 0.1 + j * spacing_x + (i % 2) * spacing_x/2
                y = 0.1 + i * spacing_y
                # Ensure we stay within bounds
                if x <= 0.9 and y <= 0.9:
                    circles[idx] = [x, y, 0.05]
                    idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with random placements near edges
        for i in range(idx, n):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            circles[i] = [x, y, 0.05]
            
        return circles
    
    # Constraint checking
    def check_constraints(circles):
        """Check if all circles satisfy containment and non-overlap constraints"""
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < r1 + r2:
                    return False
        return True
    
    # Objective function to maximize sum of radii
    def objective(params):
        # Reshape params back to circles array
        circles = params.reshape((n, 3))
        # Return negative because we want to maximize (minimize negative)
        return -np.sum(circles[:, 2])
    
    # Constraint functions
    def containment_constraint(params):
        circles = params.reshape((n, 3))
        # For each circle, we need: r <= x <= 1-r and r <= y <= 1-r
        # This gives us constraints: x - r >= 0, 1-r - x >= 0, y - r >= 0, 1-r - y >= 0
        constraints = []
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([x - r, 1 - r - x, y - r, 1 - r - y])
        return np.array(constraints)
    
    def overlap_constraint(params):
        circles = params.reshape((n, 3))
        # For each pair of circles, we need: sqrt((x1-x2)^2 + (y1-y2)^2) >= r1 + r2
        # This gives us: (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                min_dist_sq = (r1 + r2)**2
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)
    
    # Generate initial configuration
    circles = initialize_hexagonal_pattern()
    
    # Refine using optimization
    # Flatten for optimization
    initial_params = circles.flatten()
    
    # Define bounds for parameters (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # r <= 0.5 to allow some margin
    
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: containment_constraint(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
    ]
    
    try:
        # Try multiple optimization approaches
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            refined_circles = result.x.reshape((n, 3))
        else:
            # Fallback to simple local optimization
            refined_circles = circles.copy()
            
        # Apply final refinement step
        # Simple local search: try small perturbations
        best_circles = refined_circles.copy()
        best_sum = np.sum(refined_circles[:, 2])
        
        for _ in range(1000):  # Limited iterations
            # Make small random changes
            test_circles = best_circles.copy()
            idx = random.randint(0, n-1)
            # Slightly adjust one circle
            test_circles[idx, 0] += random.uniform(-0.01, 0.01)
            test_circles[idx, 1] += random.uniform(-0.01, 0.01)
            test_circles[idx, 2] += random.uniform(-0.005, 0.005)
            
            # Keep within bounds
            test_circles[idx, 0] = np.clip(test_circles[idx, 0], 0.001, 0.999)
            test_circles[idx, 1] = np.clip(test_circles[idx, 1], 0.001, 0.999)
            test_circles[idx, 2] = np.clip(test_circles[idx, 2], 0.001, 0.499)
            
            # Check constraints
            if check_constraints(test_circles.reshape((n, 3))):
                test_sum = np.sum(test_circles[:, 2])
                if test_sum > best_sum:
                    best_circles = test_circles
                    best_sum = test_sum
                    
        return best_circles.reshape((n, 3))
        
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
