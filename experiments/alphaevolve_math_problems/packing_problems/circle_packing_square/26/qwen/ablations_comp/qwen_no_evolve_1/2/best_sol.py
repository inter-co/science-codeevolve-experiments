# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple, List
import time

# Global constants for the optimization
MAX_ITERATIONS = 1000
INITIAL_RADIUS = 0.05
GRID_SIZE = 5

def _compute_distances(circles: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between circle centers."""
    centers = circles[:, :2]
    return cdist(centers, centers)

def _check_constraints(circles: np.ndarray) -> bool:
    """Check if all circles satisfy containment and non-overlap constraints."""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False
    
    # Check non-overlap constraints
    distances = _compute_distances(circles)
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            r_i, r_j = circles[i, 2], circles[j, 2]
            if dist < r_i + r_j:
                return False
    
    return True

def _objective_function(params: np.ndarray) -> float:
    """Objective function to maximize sum of radii."""
    # Reshape params into circles array
    n = 26
    circles = params.reshape((n, 3))
    
    # Extract radii
    radii = circles[:, 2]
    
    # Return negative because we're minimizing
    return -np.sum(radii)

def _constraint_containment(circles: np.ndarray) -> np.ndarray:
    """Generate containment constraints for all circles."""
    n = len(circles)
    constraints = []
    
    for i in range(n):
        x, y, r = circles[i]
        # r <= x <= 1-r
        constraints.append(x - r)  # x - r >= 0
        constraints.append(1 - r - x)  # 1 - r - x >= 0
        # r <= y <= 1-r  
        constraints.append(y - r)  # y - r >= 0
        constraints.append(1 - r - y)  # 1 - r - y >= 0
    
    return np.array(constraints)

def _constraint_nonoverlap(circles: np.ndarray) -> np.ndarray:
    """Generate non-overlap constraints for all pairs of circles."""
    n = len(circles)
    constraints = []
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            
            # Distance between centers minus sum of radii must be >= 0
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            constraints.append(dist - (r1 + r2))
    
    return np.array(constraints)

def _initialize_grid_config(n: int) -> np.ndarray:
    """Initialize circles in a grid pattern with small random perturbations."""
    circles = np.zeros((n, 3))
    
    # Create a grid layout
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    
    # Calculate spacing
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Add small random perturbation
            x += np.random.uniform(-0.01, 0.01)
            y += np.random.uniform(-0.01, 0.01)
            # Initial radius
            r = INITIAL_RADIUS
            
            # Ensure it's within bounds
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            circles[idx] = [x, y, r]
            idx += 1
    
    return circles

def _optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions using scipy's minimize function."""
    # Flatten initial circles for optimization
    initial_params = initial_circles.flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(26):
        # x bounds
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Constraints for optimization
    def constraint_func(params):
        circles = params.reshape((26, 3))
        # Return negative values for inequality constraints (g(x) >= 0)
        # For containment: x-r >= 0, 1-r-x >= 0, y-r >= 0, 1-r-y >= 0
        # For non-overlap: dist - (r1+r2) >= 0
        containment = _constraint_containment(circles)
        nonoverlap = _constraint_nonoverlap(circles)
        return np.concatenate([containment, nonoverlap])
    
    # Optimization constraints
    constraints = {'type': 'ineq', 'fun': constraint_func}
    
    try:
        result = minimize(
            _objective_function,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': MAX_ITERATIONS, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape((26, 3))
            # Clip radii to valid range
            optimized_circles[:, 2] = np.clip(optimized_circles[:, 2], 0.001, 0.499)
            return optimized_circles
        else:
            return initial_circles
    except Exception:
        return initial_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    # Multi-start optimization to improve chances of finding global optimum
    best_circles = None
    best_sum = -float('inf')
    
    # Try several random initializations
    for attempt in range(5):
        # Initialize with grid configuration
        initial_circles = _initialize_grid_config(26)
        
        # Apply optimization
        optimized_circles = _optimize_circles(initial_circles)
        
        # Check if the result satisfies constraints
        if _check_constraints(optimized_circles):
            sum_radii = np.sum(optimized_circles[:, 2])
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_circles = optimized_circles.copy()
    
    # If no valid configuration found, return a default one
    if best_circles is None:
        best_circles = _initialize_grid_config(26)
    
    return best_circles


# EVOLVE-BLOCK-END
