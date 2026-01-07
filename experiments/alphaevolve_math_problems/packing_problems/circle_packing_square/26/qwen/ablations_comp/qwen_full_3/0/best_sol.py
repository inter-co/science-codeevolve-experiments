# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies with high-precision optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 26
    
    # Strategy 1: Hexagonal grid initialization (from inspiration 2)
    def initialize_hexagonal_grid():
        rows = 5
        cols = 6
        spacing_x = 0.9 / cols  # Leave 0.05 margin on each side
        spacing_y = 0.9 / rows
        
        circles = np.zeros((n, 3))
        idx = 0
        
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = 0.05 + j * spacing_x
                y = 0.05 + i * spacing_y
                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x / 2
                circles[idx] = [x, y, min(spacing_x, spacing_y) * 0.2]  # Start with reasonable radius
                idx += 1
                if idx >= n:
                    break
        return circles
    
    # Strategy 2: Grid initialization (from inspiration 2)
    def initialize_grid():
        rows = 5
        cols = 6
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows
        
        circles = np.zeros((n, 3))
        idx = 0
        
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                y = 0.05 + (i + 0.5) * spacing_y
                circles[idx] = [x, y, min(spacing_x, spacing_y) * 0.25]
                idx += 1
                if idx >= n:
                    break
        return circles
    
    # Strategy 3: Random initialization (from inspiration 2)
    def initialize_random():
        circles = np.zeros((n, 3))
        for i in range(n):
            # Better distributed random points
            row = i // 5
            col = i % 5
            x = 0.1 + col * 0.18 + random.uniform(-0.02, 0.02)
            y = 0.1 + row * 0.18 + random.uniform(-0.02, 0.02)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            r = 0.07
            circles[i] = [x, y, r]
        return circles
    
    # Try different initialization strategies and pick the best
    strategies = [
        initialize_hexagonal_grid(),
        initialize_grid(), 
        initialize_random()
    ]
    
    best_circles = None
    best_sum = 0
    
    for strategy in strategies:
        sum_radii = np.sum(strategy[:, 2])
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_circles = strategy.copy()
    
    # Run several optimization attempts with better settings (from inspiration 2)
    best_final = None
    best_final_sum = 0
    
    # Run multiple optimization attempts to get better results
    for attempt in range(20):  # Increased from 15 to 20 (from inspiration 2)
        # Create a slightly randomized version of our best initial circles
        current_circles = best_circles.copy()
        
        # Add more significant random perturbations for better exploration
        for i in range(n):
            current_circles[i, 0] += random.uniform(-0.03, 0.03)
            current_circles[i, 1] += random.uniform(-0.03, 0.03)
            current_circles[i, 2] += random.uniform(-0.015, 0.015)
            # Keep within bounds
            current_circles[i, 0] = np.clip(current_circles[i, 0], 0.001, 0.999)
            current_circles[i, 1] = np.clip(current_circles[i, 1], 0.001, 0.999)
            current_circles[i, 2] = np.clip(current_circles[i, 2], 0.001, 0.499)
        
        # Optimized with higher precision and more iterations (from inspiration 2)
        optimized = optimize_circles_high_precision(current_circles)
        
        # Enhanced refinement (from inspiration 2)
        refined = enhanced_refine_positions(optimized)
        
        # Check if this is better
        current_sum = np.sum(refined[:, 2])
        if current_sum > best_final_sum:
            best_final_sum = current_sum
            best_final = refined.copy()
    
    # If we got a better result from optimization, return it
    if best_final is not None:
        return best_final
    
    # Otherwise, return the best initialization
    return best_circles

def optimize_circles_high_precision(circles):
    """Use scipy SLSQP optimization with high precision (from inspiration 2)"""
    # Flatten circles array for optimization
    circles_flat = circles.flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(len(circles_flat)):
        if i % 3 == 0:  # x coordinate
            bounds.append((0.001, 0.999))
        elif i % 3 == 1:  # y coordinate
            bounds.append((0.001, 0.999))
        else:  # r coordinate
            bounds.append((0.001, 0.5))  # Radius bounded
    
    # Define constraints directly (from inspiration 2)
    def constraint_containment(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        n = len(circles)
        constraints = []
        for i in range(n):
            x, y, r = circles[i]
            # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
            constraints.extend([
                x - r,           # x - r >= 0
                1 - x - r,       # 1 - x - r >= 0
                y - r,           # y - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        return np.array(constraints)

    def constraint_overlap(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        n = len(circles)
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(distance - r1 - r2)
        return np.array(constraints)
    
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_overlap(x)}
    ]
    
    try:
        # Perform optimization with very high precision (from inspiration 2)
        result = minimize(
            lambda x: -np.sum(x[2::3]),  # Maximize sum of radii
            circles_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 5000, 'ftol': 1e-10, 'eps': 1e-8}  # Much tighter tolerances (from inspiration 2)
        )
        
        if result.success:
            circles = result.x.reshape(-1, 3)
        else:
            # If optimization fails, return original circles
            pass
    except Exception as e:
        # If optimization fails, return original circles
        pass
    
    return circles

def enhanced_refine_positions(circles):
    """Enhanced local refinement to maximize sum of radii (from inspiration 2)"""
    n = len(circles)
    
    # Even more aggressive refinement with better convergence handling (from inspiration 2)
    for iteration in range(500):  # Increased iterations (from inspiration 2)
        improved = False
        for i in range(n):
            # Try to increase radius of circle i while maintaining constraints
            old_r = circles[i, 2]
            max_radius = min(
                circles[i, 0], 1 - circles[i, 0],
                circles[i, 1], 1 - circles[i, 1]
            )
            
            # Check overlap constraints with other circles
            for j in range(n):
                if i != j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    max_allowed = distance - r2
                    max_radius = min(max_radius, max_allowed)
            
            # Increase radius if possible - very careful step size (from inspiration 2)
            if max_radius > old_r:
                # Use a very small increment to ensure convergence stability
                # But still make meaningful progress
                increment = min(0.002, (max_radius - old_r) * 0.5)  # Conservative but progressive (from inspiration 2)
                new_r = old_r + increment
                circles[i, 2] = new_r
                improved = True
                
            # Adjust position if needed to accommodate larger radius
            if max_radius > old_r:
                # Keep center within bounds
                circles[i, 0] = np.clip(circles[i, 0], new_r, 1 - new_r)
                circles[i, 1] = np.clip(circles[i, 1], new_r, 1 - new_r)
        
        # Early stopping if no improvement for several iterations (from inspiration 2)
        if not improved and iteration > 100:
            break
    
    return circles


# EVOLVE-BLOCK-END
