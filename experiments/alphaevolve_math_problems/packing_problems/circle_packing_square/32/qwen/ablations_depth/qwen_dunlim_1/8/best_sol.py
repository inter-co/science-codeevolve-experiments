# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles using a more sophisticated approach
    circles = np.zeros((n, 3))
    
    # Create a hexagonal packing pattern for better initial configuration
    # This provides a more uniform distribution that's closer to optimal
    rows = 6
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + 1 + offset) * spacing_x
            y = (i + 1) * spacing_y
            # Initial radius - small enough to fit in grid cell
            r = min(spacing_x, spacing_y) * 0.3
            # Ensure we don't place circles too close to boundaries
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Ensure we have exactly 32 circles
    if idx < n:
        # Fill remaining positions with better initialization
        for i in range(idx, n):
            # Place circles near the edges with smaller radii
            # This helps avoid getting stuck in local optima
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = np.random.uniform(0.01, 0.05)
            circles[i] = [x, y, r]
    
    # Define objective function to maximize sum of radii
    def objective(params):
        # params contains [x1, y1, r1, x2, y2, r2, ...]
        total_radius = 0
        for i in range(0, len(params), 3):
            total_radius += params[i+2]  # Add radius component
        return -total_radius  # Negative because we minimize
    
    # Define constraints with improved numerical stability
    def containment_constraint(params):
        # Check that all circles are within the unit square
        constraints = []
        for i in range(0, len(params), 3):
            x, y, r = params[i], params[i+1], params[i+2]
            # Circle must be contained in unit square with margin for numerical stability
            constraints.append(x - r - 1e-8)  # x - r >= 0
            constraints.append(y - r - 1e-8)  # y - r >= 0
            constraints.append(1 - x - r - 1e-8)  # 1 - x - r >= 0
            constraints.append(1 - y - r - 1e-8)  # 1 - y - r >= 0
        return np.array(constraints)
    
    def overlap_constraint(params):
        # Check that no two circles overlap
        constraints = []
        # Convert params back to circles array for easier processing
        circles_array = []
        for i in range(0, len(params), 3):
            circles_array.append([params[i], params[i+1], params[i+2]])
        
        # For each pair of circles, check if they overlap
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                # Distance between centers squared
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # Minimum allowed distance (sum of radii) squared
                min_dist_sq = (r1 + r2)**2
                # We want dist >= r1 + r2, so we enforce dist_sq >= min_dist_sq
                # This means: dist_sq - min_dist_sq >= 0
                # Add small tolerance to prevent numerical issues
                constraints.append(dist_sq - min_dist_sq - 1e-10)
        return np.array(constraints)
    
    # Set up bounds for optimization with tighter constraints
    bounds = []
    for i in range(n):
        # Bounds for coordinates: [r + 1e-8, 1-r - 1e-8] to ensure containment
        bounds.append((1e-8, 1-1e-8))  # x
        bounds.append((1e-8, 1-1e-8))  # y
        bounds.append((1e-8, 0.4))     # r (reasonable upper bound)
    
    # Try multiple optimization strategies to find better solutions
    best_result = None
    best_sum = 0
    
    # Strategy 1: SLSQP with multiple restarts
    for restart in range(3):
        try:
            # Start with slightly perturbed version of current configuration
            initial_params = circles.flatten() + np.random.normal(0, 0.001, len(circles.flatten()))
            
            # Ensure bounds are respected
            for i in range(0, len(initial_params), 3):
                initial_params[i] = np.clip(initial_params[i], 1e-8, 1-1e-8)  # x
                initial_params[i+1] = np.clip(initial_params[i+1], 1e-8, 1-1e-8)  # y
                initial_params[i+2] = np.clip(initial_params[i+2], 1e-8, 0.4)  # r
            
            # Define constraint dictionaries
            containment_cons = {
                'type': 'ineq',
                'fun': lambda x: containment_constraint(x)
            }
            
            overlap_cons = {
                'type': 'ineq', 
                'fun': lambda x: overlap_constraint(x)
            }
            
            # Run optimization with different tolerances
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=[containment_cons, overlap_cons],
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                # Calculate sum of radii for this solution
                total_radius = -objective(result.x)
                if total_radius > best_sum:
                    best_sum = total_radius
                    best_result = result
                    
        except Exception as e:
            continue
    
    # If we found a better solution, use it; otherwise fallback to initial
    if best_result is not None and best_result.success:
        final_params = best_result.x
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
    else:
        # If optimization fails or doesn't improve, just return the initial configuration
        # but with some improvements
        pass
    
    return circles


# EVOLVE-BLOCK-END
