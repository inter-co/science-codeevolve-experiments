# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal initialization with mathematical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    n = 32
    
    # Improved hexagonal initialization like INSPIRATION 1
    def generate_hexagonal_initial():
        # Create a hexagonal grid pattern that better fills the space
        rows = 6
        cols = 6
        spacing = 0.15  # Fixed spacing for better control
        spacing_x = spacing
        spacing_y = spacing * np.sqrt(3) / 2
        
        circles = []
        count = 0
        
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y + (j % 2) * spacing_y / 2
                
                # Ensure we stay within bounds
                if x <= 0.9 and y <= 0.9:
                    # Set initial radius based on distance to edges
                    r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                    circles.append([x, y, r])
                    count += 1
            if count >= n:
                break
        
        # Fill remaining positions with random valid placements
        for i in range(count, n):
            attempts = 0
            while attempts < 100:
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                # Radius based on distance to edges
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                
                # Check if this position is valid with existing circles
                valid = True
                for k in range(i):
                    cx, cy, cr = circles[k]
                    dist_sq = (x - cx)**2 + (y - cy)**2
                    if dist_sq < (r + cr)**2:
                        valid = False
                        break
                
                if valid:
                    circles.append([x, y, r])
                    break
                attempts += 1
        
        return np.array(circles)
    
    # Generate initial configuration
    circles = generate_hexagonal_initial()
    
    # Define constraint functions for optimization
    def contain_constraints(circles_flat):
        """Ensure all circles are contained within the unit square"""
        n = len(circles_flat) // 3
        x = circles_flat[::3]
        y = circles_flat[1::3]
        r = circles_flat[2::3]
        
        # Each circle must satisfy containment constraints
        constraints = []
        for i in range(n):
            # radius must be positive
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2+i*3]})
            # x coordinate constraints
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[0+i*3] - x[2+i*3]})  # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[0+i*3] - x[2+i*3]})  # 1-x >= r
            # y coordinate constraints
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[1+i*3] - x[2+i*3]})  # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[1+i*3] - x[2+i*3]})  # 1-y >= r
            
        return constraints
    
    def non_overlap_constraints(circles_flat):
        """Ensure no two circles overlap"""
        n = len(circles_flat) // 3
        constraints = []
        
        for i in range(n):
            for j in range(i+1, n):
                # Distance between centers must be >= sum of radii
                def constraint_func(x, i=i, j=j):
                    xi, yi, ri = x[i*3], x[i*3+1], x[i*3+2]
                    xj, yj, rj = x[j*3], x[j*3+1], x[j*3+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    # We want sqrt(dist_sq) >= ri + rj, so dist_sq >= (ri + rj)^2
                    return dist_sq - (ri + rj)**2
                    
                constraints.append({'type': 'ineq', 'fun': constraint_func})
                
        return constraints
    
    # Objective function (negative because we're minimizing)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Sum of radii (negated for minimization)
    
    # Flatten the circles array for optimization
    initial_flat = circles.flatten()
    
    # Create constraints
    cons = []
    # Add containment constraints
    for constraint in contain_constraints(initial_flat):
        cons.append(constraint)
    # Add non-overlap constraints
    for constraint in non_overlap_constraints(initial_flat):
        cons.append(constraint)
    
    # Bounds for variables (x, y, r) - ensure radii are positive and positions are valid
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((1e-6, 1-1e-6))  # x in (0,1)
        # y coordinate bounds  
        bounds.append((1e-6, 1-1e-6))  # y in (0,1)
        # radius bounds
        bounds.append((1e-6, 0.5))     # r in (0,0.5) - max possible radius is 0.5
    
    # Optimize using SLSQP method
    try:
        result = minimize(objective, initial_flat, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6})
        
        if result.success:
            optimized_flat = result.x
            circles = optimized_flat.reshape(-1, 3)
        else:
            # If optimization fails, return initial configuration
            pass
    except Exception as e:
        # If optimization fails due to numerical issues, return initial configuration
        pass
    
    # Final cleanup to ensure all circles are properly contained
    for i in range(n):
        x, y, r = circles[i]
        # Clamp coordinates to valid range
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]
    
    # Multiple optimization restarts for better results
    best_circles = circles.copy()
    best_radius_sum = np.sum(best_circles[:, 2])
    
    # Try several random restarts with local improvements
    for restart in range(3):
        # Create a perturbed version of the current best
        perturbed = best_circles.copy()
        
        # Perturb more aggressively in early iterations
        perturbation_strength = 0.05 if restart < 2 else 0.02
        
        for i in range(n):
            if random.random() < 0.4:  # 40% chance to perturb each circle
                perturbed[i, 0] += random.uniform(-perturbation_strength, perturbation_strength)
                perturbed[i, 1] += random.uniform(-perturbation_strength, perturbation_strength)
                # Keep within bounds
                perturbed[i, 0] = max(0.01, min(0.99, perturbed[i, 0]))
                perturbed[i, 1] = max(0.01, min(0.99, perturbed[i, 1]))
        
        # Apply optimization to the perturbed version
        perturbed_flat = perturbed.flatten()
        
        try:
            result = minimize(objective, perturbed_flat, method='SLSQP', bounds=bounds, constraints=cons,
                             options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6})
            
            if result.success:
                refined_flat = result.x
                refined = refined_flat.reshape(-1, 3)
                
                # Check if this improved the solution
                new_radius_sum = np.sum(refined[:, 2])
                if new_radius_sum > best_radius_sum:
                    best_circles = refined
                    best_radius_sum = new_radius_sum
        except:
            pass
    
    # Final cleanup
    for i in range(n):
        x, y, r = best_circles[i]
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        best_circles[i] = [x, y, r]
    
    return best_circles


# EVOLVE-BLOCK-END
