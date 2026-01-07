# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach combining:
    1. Initial placement using a grid-based heuristic
    2. Optimization using scipy minimize with constraints
    3. Physics-inspired repulsion forces for local improvement
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize circles with a good starting configuration
    # Start with a grid-like pattern and then optimize
    circles = np.zeros((n, 3))
    
    # Create initial grid-based placement
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Distribute circles in a grid pattern
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols * 0.8 + 0.1  # Keep away from edges
        y = (row + 0.5) / rows * 0.8 + 0.1
        circles[i] = [x, y, 0.05]  # Start with small radius
    
    # Set initial radii based on spacing
    max_radius = 0.1
    for i in range(n):
        # Ensure we don't place too close to edges
        min_dist_to_edge = min(circles[i][0], 1-circles[i][0], circles[i][1], 1-circles[i][1])
        circles[i][2] = min(max_radius, min_dist_to_edge * 0.8)
    
    # Convert to flattened parameter vector for optimization
    # Parameters: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    def get_params_from_circles(circles_array):
        params = []
        for i in range(n):
            params.extend([circles_array[i][0], circles_array[i][1], circles_array[i][2]])
        return np.array(params)
    
    def get_circles_from_params(params):
        circles_array = np.zeros((n, 3))
        for i in range(n):
            circles_array[i] = [params[3*i], params[3*i+1], params[3*i+2]]
        return circles_array
    
    def objective(params):
        # We want to maximize sum of radii, so minimize negative sum
        circles_array = get_circles_from_params(params)
        return -np.sum(circles_array[:, 2])
    
    def constraint_containment(params):
        # Ensure all circles are fully contained in unit square
        circles_array = get_circles_from_params(params)
        constraints = []
        
        # For each circle, check containment constraints
        for i in range(n):
            x, y, r = circles_array[i]
            # r <= x <= 1-r and r <= y <= 1-r
            constraints.append(x - r)      # x - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(y - r)      # y - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        return np.array(constraints)
    
    def constraint_nonoverlap(params):
        # Ensure no overlap between circles
        circles_array = get_circles_from_params(params)
        constraints = []
        
        # Check pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                
                # Distance between centers minus radii should be >= 0
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(dist - (r1 + r2))  # dist >= r1 + r2
                
        return np.array(constraints)
    
    # Define bounds for parameters (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x bounds: [r, 1-r] 
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Create initial parameter vector
    initial_params = get_params_from_circles(circles)
    
    # First phase: coarse optimization with relaxed constraints
    try:
        # Use L-BFGS-B which handles bounds well
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6},
            callback=lambda x: None  # Placeholder callback
        )
        
        if result.success:
            optimized_circles = get_circles_from_params(result.x)
        else:
            optimized_circles = circles
    except Exception:
        optimized_circles = circles
    
    # Second phase: fine-tuning with more precise constraints
    # Try to improve further with a local search approach
    try:
        # Use scipy minimize with constraints for better precision
        cons = [
            {'type': 'ineq', 'fun': lambda p: constraint_containment(p)},
            {'type': 'ineq', 'fun': lambda p: constraint_nonoverlap(p)}
        ]
        
        result = minimize(
            objective,
            get_params_from_circles(optimized_circles),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-6}
        )
        
        if result.success:
            final_circles = get_circles_from_params(result.x)
        else:
            final_circles = optimized_circles
    except Exception:
        final_circles = optimized_circles
    
    # Final refinement with a simple iterative improvement
    # Apply some additional heuristics to improve quality
    best_circles = final_circles.copy()
    best_sum = np.sum(best_circles[:, 2])
    
    # Try a few iterations of local optimization
    for _ in range(20):
        # Try to slightly adjust positions to improve sum
        test_circles = best_circles.copy()
        
        # Slightly perturb one circle at a time
        for i in range(n):
            # Save original
            orig_x, orig_y, orig_r = test_circles[i]
            
            # Try small perturbations
            best_local_sum = np.sum(test_circles[:, 2])
            best_local_circle = test_circles[i].copy()
            
            # Try several small adjustments
            for dx in [-0.005, -0.002, 0, 0.002, 0.005]:
                for dy in [-0.005, -0.002, 0, 0.002, 0.005]:
                    for dr in [-0.003, -0.001, 0, 0.001, 0.003]:
                        test_circles[i][0] = max(0.001, min(0.999, orig_x + dx))
                        test_circles[i][1] = max(0.001, min(0.999, orig_y + dy))
                        test_circles[i][2] = max(0.001, min(0.499, orig_r + dr))
                        
                        # Check constraints
                        valid = True
                        # Check containment
                        if (test_circles[i][2] > test_circles[i][0] or 
                            test_circles[i][2] > (1 - test_circles[i][0]) or
                            test_circles[i][2] > test_circles[i][1] or 
                            test_circles[i][2] > (1 - test_circles[i][1])):
                            valid = False
                        
                        # Check overlaps with others
                        if valid:
                            for j in range(n):
                                if i != j:
                                    dist = np.sqrt((test_circles[i][0]-test_circles[j][0])**2 + 
                                                   (test_circles[i][1]-test_circles[j][1])**2)
                                    if dist < (test_circles[i][2] + test_circles[j][2]):
                                        valid = False
                                        break
                        
                        if valid:
                            new_sum = np.sum(test_circles[:, 2])
                            if new_sum > best_local_sum:
                                best_local_sum = new_sum
                                best_local_circle = test_circles[i].copy()
            
            # Accept the best improvement
            if np.sum(best_local_circle) > np.sum(best_circles[i]):
                test_circles[i] = best_local_circle
                best_circles = test_circles.copy()
                best_sum = best_local_sum
        
        # If no improvement, break
        if abs(best_sum - np.sum(best_circles[:, 2])) < 1e-8:
            break
    
    return best_circles


# EVOLVE-BLOCK-END
