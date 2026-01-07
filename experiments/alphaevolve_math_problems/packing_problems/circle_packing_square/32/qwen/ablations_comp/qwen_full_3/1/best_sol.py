# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a more sophisticated hexagonal grid pattern
    def initialize_hexagonal_grid():
        circles = np.zeros((n, 3))
        
        # Create a hexagonal lattice pattern for better initial configuration
        rows = 6
        cols = 6
        
        # Calculate spacing for hexagonal packing
        spacing = 0.15  # Adjusted for better packing
        hex_radius = spacing / 2
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for hexagonal packing
                x = 0.1 + (j + 0.5 * (i % 2)) * spacing
                y = 0.1 + i * spacing * math.sqrt(3)/2
                
                # Ensure within bounds with some margin
                if x + hex_radius <= 0.9 and y + hex_radius <= 0.9:
                    circles[idx] = [x, y, hex_radius]
                    idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with careful random placements
        for i in range(idx, n):
            attempts = 0
            while attempts < 1000:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                radius = 0.04  # Start with a reasonable radius
                
                # Check overlap with existing circles
                overlap = False
                for j in range(i):
                    dx = x - circles[j, 0]
                    dy = y - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radius + circles[j, 2]:
                        overlap = True
                        break
                
                if not overlap:
                    circles[i] = [x, y, radius]
                    break
                attempts += 1
            
            # If couldn't place without overlap, just use random position with small radius
            if attempts >= 1000:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                circles[i] = [x, y, 0.02]
        
        return circles
    
    # Constraint handling with better vectorization
    def constraint_violations(params):
        # Reshape parameters back into circles array
        pos_radii = params.reshape(-1, 3)
        x = pos_radii[:, 0]
        y = pos_radii[:, 1]
        r = pos_radii[:, 2]
        
        # Containment constraints: each circle must be fully inside the unit square
        containment_violations = []
        for i in range(n):
            containment_violations.extend([
                x[i] - r[i],           # x - r >= 0
                y[i] - r[i],           # y - r >= 0
                1 - x[i] - r[i],       # 1 - x - r >= 0
                1 - y[i] - r[i]        # 1 - y - r >= 0
            ])
        
        # Non-overlap constraints: distance between centers >= sum of radii
        non_overlap_violations = []
        for i in range(n):
            for j in range(i+1, n):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                sum_radii = r[i] + r[j]
                # Violation is negative when circles overlap
                non_overlap_violations.append(dist_sq - sum_radii**2)
        
        return np.array(containment_violations + non_overlap_violations)
    
    # Objective function to maximize sum of radii
    def objective(params):
        # Reshape parameters back into circles array
        pos_radii = params.reshape(-1, 3)
        radii = pos_radii[:, 2]
        return -np.sum(radii)  # Negative because we want to maximize
    
    # Constraint function for scipy.optimize
    def constraint_func(params):
        violations = constraint_violations(params)
        return violations
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # Bounds for x and y: [0.001, 0.999] to ensure containment with margin
        # Bounds for radius: [0.001, 0.499] to prevent numerical issues
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Try multiple starting points and optimization methods to improve results
    best_solution = None
    best_sum = 0
    
    # Try 15 different initializations to get better results
    for attempt in range(15):
        # Initialize with hexagonal grid
        circles = initialize_hexagonal_grid()
        
        # Flatten initial guess
        initial_guess = circles.flatten()
        
        # Define constraints dictionary
        cons = {
            'type': 'ineq',
            'fun': constraint_func
        }
        
        # Try different optimization methods
        methods_to_try = ['SLSQP', 'trust-constr']
        
        for method in methods_to_try:
            try:
                result = minimize(
                    objective,
                    initial_guess,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 800, 'ftol': 1e-7, 'gtol': 1e-7}
                )
                
                if result.success:
                    optimized_circles = result.x.reshape(-1, 3)
                    current_sum = np.sum(optimized_circles[:, 2])
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_solution = optimized_circles.copy()
            except Exception:
                continue
    
    # If we found a good solution, return it; otherwise return the initial configuration
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to the hexagonal grid initialization
        return initialize_hexagonal_grid()


# EVOLVE-BLOCK-END
