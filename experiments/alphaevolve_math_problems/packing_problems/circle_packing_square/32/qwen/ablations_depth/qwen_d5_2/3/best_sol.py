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
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using a more informed approach
    def initialize_circles():
        # Start with a more strategic placement based on known good configurations
        # Use a grid pattern with some perturbation for better distribution
        sqrt_n = int(np.ceil(np.sqrt(n)))
        spacing_x = 1.0 / (sqrt_n + 1)
        spacing_y = 1.0 / (sqrt_n + 1)
        
        circles = []
        radius_guess = 0.05
        
        # Create a structured grid
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Add small random perturbation to avoid perfect grid issues
                x += random.uniform(-spacing_x*0.1, spacing_x*0.1)
                y += random.uniform(-spacing_y*0.1, spacing_y*0.1)
                # Ensure within bounds
                if x + radius_guess < 1 and y + radius_guess < 1 and x - radius_guess > 0 and y - radius_guess > 0:
                    circles.append([x, y, radius_guess])
        
        # Fill remaining slots with random positions but with better distribution
        while len(circles) < n:
            # Try to place in a way that avoids clustering
            x = np.random.uniform(radius_guess, 1-radius_guess)
            y = np.random.uniform(radius_guess, 1-radius_guess)
            # Small probability to use more strategic placement
            if np.random.random() < 0.3:
                # Place near existing circles to encourage filling gaps
                if circles:
                    ref_circle = random.choice(circles)
                    angle = np.random.uniform(0, 2*np.pi)
                    distance = np.random.uniform(radius_guess*0.5, radius_guess*2)
                    x = ref_circle[0] + distance * np.cos(angle)
                    y = ref_circle[1] + distance * np.sin(angle)
                    # Keep within bounds
                    x = np.clip(x, radius_guess, 1-radius_guess)
                    y = np.clip(y, radius_guess, 1-radius_guess)
            
            circles.append([x, y, radius_guess])
            
        return np.array(circles)
    
    # More efficient constraint creation with better numerical handling
    def create_constraints():
        """Create constraints efficiently"""
        constraints = []
        
        # Containment constraints: for each circle i, we need:
        # x_i >= r_i, y_i >= r_i, 1-x_i >= r_i, 1-y_i >= r_i
        for i in range(n):
            # x >= r
            def containment_x(c, i=i):
                return c[3*i] - c[3*i+2]
            
            # y >= r  
            def containment_y(c, i=i):
                return c[3*i+1] - c[3*i+2]
            
            # 1-x >= r
            def containment_x_max(c, i=i):
                return 1 - c[3*i] - c[3*i+2]
            
            # 1-y >= r
            def containment_y_max(c, i=i):
                return 1 - c[3*i+1] - c[3*i+2]
                
            constraints.append({'type': 'ineq', 'fun': containment_x})
            constraints.append({'type': 'ineq', 'fun': containment_y})
            constraints.append({'type': 'ineq', 'fun': containment_x_max})
            constraints.append({'type': 'ineq', 'fun': containment_y_max})
        
        # Non-overlap constraints: for each pair (i,j), we need:
        # (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
        for i, j in combinations(range(n), 2):
            def non_overlap(c, i=i, j=j):
                x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                # Add small epsilon to avoid numerical issues
                return dist_sq - min_dist_sq + 1e-10
                
            constraints.append({'type': 'ineq', 'fun': non_overlap})
            
        return constraints
    
    # Objective function (negative because we want to maximize)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Negative because we're minimizing sum of radii
    
    # Multi-start optimization to find better local optima
    best_result = None
    best_sum = 0
    
    # Try multiple random starts to avoid local minima
    for start in range(5):
        # Initialize with different random seed
        random.seed(start)
        np.random.seed(start)
        
        circles = initialize_circles()
        initial_guess = circles.flatten()
        
        # Set up constraints
        cons = create_constraints()
        
        # Bounds for variables (x, y, r for each circle)
        bounds = []
        for i in range(n):
            bounds.append((0.001, 0.999))  # x coordinate
            bounds.append((0.001, 0.999))  # y coordinate
            bounds.append((0.001, 0.499))   # radius (slightly less than 0.5 to avoid numerical issues)
        
        # Try with different optimization methods
        try:
            # Try SLSQP first with more iterations
            result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=cons, 
                             options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6})
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            continue
    
    # If no good result found, try L-BFGS-B with more aggressive settings
    if best_result is None:
        try:
            circles = initialize_circles()
            initial_guess = circles.flatten()
            result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6})
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            pass
    
    # Return best result or initial configuration
    if best_result is not None and best_result.success:
        final_circles = best_result.x.reshape(-1, 3)
        return final_circles
    else:
        # Fallback to the initial configuration
        circles = initialize_circles()
        return circles


# EVOLVE-BLOCK-END
