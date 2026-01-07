# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies with optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Strategy 1: Hexagonal grid initialization (from inspiration 1)
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
                circles[idx] = [x, y, min(spacing_x, spacing_y) * 0.25]  # Start with reasonable radius
                idx += 1
                if idx >= n:
                    break
        return circles
    
    # Strategy 2: Grid initialization with better spacing (from inspiration 1)
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
    
    # Strategy 3: Random initialization with better distribution (from inspiration 1)
    def initialize_random():
        circles = np.zeros((n, 3))
        for i in range(n):
            # Better distributed random points with slight clustering
            row = i // 5
            col = i % 5
            x = 0.1 + col * 0.18 + random.uniform(-0.02, 0.02)
            y = 0.1 + row * 0.18 + random.uniform(-0.02, 0.02)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            r = 0.07
            circles[i] = [x, y, r]
        return circles
    
    # Strategy 4: Systematic approach inspired by inspiration 2
    def initialize_systematic():
        circles = np.zeros((n, 3))
        for i in range(n):
            row = i // 5
            col = i % 5
            x = 0.1 + col * 0.18
            y = 0.1 + row * 0.18
            # Add jitter to prevent perfect alignment
            x += random.uniform(-0.01, 0.01)
            y += random.uniform(-0.01, 0.01)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            r = 0.075
            circles[i] = [x, y, r]
        return circles
    
    # Try different initialization strategies
    strategies = [
        initialize_hexagonal_grid(),
        initialize_grid(),
        initialize_random(),
        initialize_systematic()
    ]
    
    best_circles = None
    best_sum = 0
    
    for strategy in strategies:
        sum_radii = np.sum(strategy[:, 2])
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_circles = strategy.copy()
    
    # Run several optimization attempts with balanced settings
    best_final = None
    best_final_sum = 0
    
    # Run fewer optimization attempts to stay within time limits but still effective
    # Using inspiration 2's approach with more precise settings but optimized for speed
    for attempt in range(8):  # Reduced from 10 to 8 for better time management
        # Create a slightly randomized version of our best initial circles
        current_circles = best_circles.copy()
        
        # Add moderate random perturbations for better exploration
        for i in range(n):
            current_circles[i, 0] += random.uniform(-0.008, 0.008)  # Smaller perturbation for faster convergence
            current_circles[i, 1] += random.uniform(-0.008, 0.008)
            current_circles[i, 2] += random.uniform(-0.003, 0.003)
            # Keep within bounds
            current_circles[i, 0] = np.clip(current_circles[i, 0], 0.001, 0.999)
            current_circles[i, 1] = np.clip(current_circles[i, 1], 0.001, 0.999)
            current_circles[i, 2] = np.clip(current_circles[i, 2], 0.001, 0.499)
        
        # Optimize with better precision and fewer iterations for speed
        optimized = optimize_circles_precise(current_circles)
        
        # Apply enhanced refinement with better convergence control
        refined = enhanced_refine_positions_precise(optimized)
        
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

def optimize_circles_precise(circles):
    """Use scipy SLSQP optimization with clean constraint handling and precise settings"""
    # Flatten circles array for optimization
    x0 = circles.flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(len(x0)):
        if i % 3 == 0:  # x coordinate
            bounds.append((0.001, 0.999))
        elif i % 3 == 1:  # y coordinate
            bounds.append((0.001, 0.999))
        else:  # r coordinate
            bounds.append((0.001, 0.499))  # Tighter upper bound for radius
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(x):
        return -np.sum(x[2::3])  # Maximize sum of radii
    
    # Constraint function for containment and non-overlap (cleaner approach from inspiration 1)
    def constraints_func(x):
        circles = x.reshape(-1, 3)
        constraints = []
        
        # Containment constraints for each circle
        for i in range(len(circles)):
            x_c, y_c, r = circles[i]
            # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
            constraints.extend([
                x_c - r,           # x - r >= 0
                1 - x_c - r,       # 1 - x - r >= 0
                y_c - r,           # y - r >= 0
                1 - y_c - r        # 1 - y - r >= 0
            ])
        
        # Non-overlap constraints - more efficient implementation
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(distance - r1 - r2)  # distance >= r1 + r2
        
        return np.array(constraints)
    
    # Define constraints properly
    cons = [{'type': 'ineq', 'fun': constraints_func}]
    
    try:
        # Perform optimization with precise settings but limited iterations for time
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9, 'eps': 1e-7}  # Reduced maxiter for faster execution
        )
        
        if result.success:
            circles = result.x.reshape(-1, 3)
        else:
            # If optimization fails, return original circles
            pass
    except Exception:
        # If optimization fails, return original circles
        pass
    
    return circles

def enhanced_refine_positions_precise(circles):
    """Precise local refinement to maximize sum of radii with better convergence control"""
    n = len(circles)
    
    # More controlled refinement process with better convergence criteria
    for iteration in range(150):  # Reduced iterations for time efficiency
        improved = False
        # Process in random order for better exploration
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            # Get current circle info
            x, y, r = circles[i]
            
            # Calculate maximum possible radius
            max_radius = min(x, 1-x, y, 1-y)
            
            # Check overlap constraints with all others more efficiently
            for j in range(n):
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x-x2)**2 + (y-y2)**2)
                    max_radius = min(max_radius, distance - r2)
            
            # Try to increase radius if beneficial (more aggressive approach)
            if max_radius > r and max_radius - r > 1e-6:
                # Use a more aggressive increment for faster improvement
                increment = min(0.001, (max_radius - r) * 0.5)  # More aggressive than before
                new_r = r + increment
                if new_r > r:
                    circles[i, 2] = new_r
                    improved = True
        
        # Early stopping if no improvement for several iterations
        if not improved and iteration > 30:
            break
    
    return circles


# EVOLVE-BLOCK-END
