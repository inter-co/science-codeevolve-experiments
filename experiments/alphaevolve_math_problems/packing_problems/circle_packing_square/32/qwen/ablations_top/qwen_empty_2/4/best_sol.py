# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import time

# Optimized approach based on mathematical programming with better initialization
def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a mathematical programming approach with good initialization and constraint handling.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 32
    max_time = 55.0  # Leave 5 seconds for final processing
    
    # Better initialization using a structured approach inspired by hexagonal packing
    def initialize_better():
        circles = np.zeros((n, 3))
        
        # Create a structured pattern that's more likely to be feasible
        # Using a combination of regular grid with some randomness
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset odd rows for better packing
                offset = spacing_x * 0.5 if i % 2 == 1 else 0
                x = (j + 0.5 + random.uniform(-0.1, 0.1)) * spacing_x + offset
                y = (i + 0.5 + random.uniform(-0.1, 0.1)) * spacing_y
                # Initial radius - small enough to fit in square
                r = min(spacing_x, spacing_y) * 0.25
                
                # Ensure we stay within bounds
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                
                circles[idx] = [x, y, r]
                idx += 1
                
        # Fill remaining circles with random positions and small radii
        for i in range(idx, n):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = random.uniform(0.01, 0.05)
            circles[i] = [x, y, r]
        
        return circles
    
    # Initialize
    circles = initialize_better()
    
    # More aggressive local optimization approach
    def objective(params):
        # params contains [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
        circles_flat = params.reshape(-1, 3)
        return -np.sum(circles_flat[:, 2])  # Negative because we want to maximize
    
    def constraint_func(params):
        circles_flat = params.reshape(-1, 3)
        positions = circles_flat[:, :2]
        radii = circles_flat[:, 2]
        
        # Non-overlap constraints
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + 
                              (positions[i, 1] - positions[j, 1])**2)
                # Should be >= sum of radii (positive means constraint satisfied)
                constraints.append(dist - (radii[i] + radii[j]))
        
        # Boundary constraints (ensure circles stay within unit square)
        for i in range(n):
            constraints.append(radii[i] - 0.001)  # Minimum radius
            constraints.append(1 - radii[i] - positions[i, 0])  # Right boundary
            constraints.append(1 - radii[i] - positions[i, 1])  # Top boundary
            constraints.append(positions[i, 0] - radii[i])  # Left boundary
            constraints.append(positions[i, 1] - radii[i])  # Bottom boundary
            
        return np.array(constraints)
    
    # Run optimization with better settings
    initial_params = circles.flatten()
    
    # Try different optimization methods for better results
    try:
        # First try with L-BFGS-B which often works well for this type of problem
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                         constraints={'type': 'ineq', 'fun': constraint_func},
                         options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-5})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure all circles are within bounds
            for i in range(n):
                # Boundary checks
                optimized_circles[i, 0] = np.clip(optimized_circles[i, 0], 
                                                 optimized_circles[i, 2], 
                                                 1 - optimized_circles[i, 2])
                optimized_circles[i, 1] = np.clip(optimized_circles[i, 1], 
                                                 optimized_circles[i, 2], 
                                                 1 - optimized_circles[i, 2])
            return optimized_circles
    except Exception as e:
        pass
    
    # If first attempt fails, try with SLSQP but with better initial conditions
    try:
        result = minimize(objective, initial_params, method='SLSQP', 
                         constraints={'type': 'ineq', 'fun': constraint_func},
                         options={'maxiter': 200, 'ftol': 1e-6, 'eps': 1e-4})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure all circles are within bounds
            for i in range(n):
                # Boundary checks
                optimized_circles[i, 0] = np.clip(optimized_circles[i, 0], 
                                                 optimized_circles[i, 2], 
                                                 1 - optimized_circles[i, 2])
                optimized_circles[i, 1] = np.clip(optimized_circles[i, 1], 
                                                 optimized_circles[i, 2], 
                                                 1 - optimized_circles[i, 2])
            return optimized_circles
    except Exception as e:
        pass
    
    # Fallback to the initial configuration if all optimization fails
    return circles


# EVOLVE-BLOCK-END
