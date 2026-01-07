# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple

def _compute_radius_constraints(circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute containment and overlap constraints."""
    n = len(circles)
    
    # Containment constraints: each circle must fit within the unit square
    containment = []
    for i in range(n):
        x, y, r = circles[i]
        containment.append([r, 1-r, r, 1-r])  # min_x, max_x, min_y, max_y
    
    # Overlap constraints: distance between centers >= sum of radii
    overlap = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            overlap.append((i, j, dist, r1+r2))
    
    return np.array(containment), overlap

def _objective_function(circles_flat: np.ndarray) -> float:
    """Objective function to maximize sum of radii."""
    # Reshape flat array back to circles format
    n = len(circles_flat) // 3
    circles = circles_flat.reshape((n, 3))
    
    # Sum of radii (negative because we're minimizing)
    return -np.sum(circles[:, 2])

def _constraint_containment(circles_flat: np.ndarray) -> np.ndarray:
    """Constraint function for containment within unit square."""
    n = len(circles_flat) // 3
    circles = circles_flat.reshape((n, 3))
    
    # Each circle must satisfy: r <= x <= 1-r and r <= y <= 1-r
    # So: x - r >= 0, 1-r - x >= 0, y - r >= 0, 1-r - y >= 0
    constraints = []
    
    for i in range(n):
        x, y, r = circles[i]
        constraints.extend([
            x - r,           # x - r >= 0
            1 - r - x,       # 1 - r - x >= 0
            y - r,           # y - r >= 0
            1 - r - y        # 1 - r - y >= 0
        ])
    
    return np.array(constraints)

def _constraint_overlap(circles_flat: np.ndarray) -> np.ndarray:
    """Constraint function for non-overlapping circles."""
    n = len(circles_flat) // 3
    circles = circles_flat.reshape((n, 3))
    
    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            
            # Distance between centers must be >= sum of radii
            # So: sqrt((x1-x2)^2 + (y1-y2)^2) >= r1 + r2
            # Or equivalently: (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
            # Which means: (x1-x2)^2 + (y1-y2)^2 - (r1+r2)^2 >= 0
            dist_sq = (x1-x2)**2 + (y1-y2)**2
            sum_radii_sq = (r1+r2)**2
            
            constraints.append(dist_sq - sum_radii_sq)
    
    return np.array(constraints)

def _hexagonal_initialization() -> np.ndarray:
    """Initialize circles in a hexagonal pattern for good starting configuration."""
    n = 32
    circles = np.zeros((n, 3))
    
    # Hexagonal packing parameters
    rows = 6  # number of rows needed for ~32 circles
    cols = 6  # number of columns
    
    # Calculate spacing
    spacing_x = 0.9 / cols  # leave some margin
    spacing_y = 0.9 / rows  # leave some margin
    
    # Adjust spacing for hexagonal packing
    hex_spacing_x = spacing_x
    hex_spacing_y = spacing_y * math.sqrt(3) / 2
    
    # Create hexagonal grid
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            x = 0.05 + j * hex_spacing_x
            y = 0.05 + i * hex_spacing_y
            
            # Offset every other row
            if i % 2 == 1:
                x += hex_spacing_x / 2
            
            # Set initial radius to be small but feasible
            r = min(hex_spacing_x, hex_spacing_y) * 0.4
            
            # Ensure it fits in the square
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[count] = [x, y, r]
            count += 1
        
        if count >= n:
            break
    
    return circles

def _optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Use scipy optimization to improve the circle configuration."""
    n = len(initial_circles)
    
    # Flatten the initial circles for optimization
    initial_flat = initial_circles.flatten()
    
    # Define bounds: x, y in [r, 1-r], r in [0, 0.5] (reasonable upper bound)
    bounds = []
    for i in range(n):
        x, y, r = initial_circles[i]
        bounds.extend([(r, 1-r), (r, 1-r), (0.001, 0.5)])  # r min 0.001 to avoid degenerate cases
    
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': _constraint_containment},
        {'type': 'ineq', 'fun': _constraint_overlap}
    ]
    
    # Optimize using SLSQP method
    try:
        result = minimize(
            _objective_function,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape((n, 3))
            return optimized_circles
    except Exception as e:
        pass
    
    # Return initial if optimization fails
    return initial_circles

def _evaluate_solution(circles: np.ndarray) -> float:
    """Calculate the sum of radii for a given solution."""
    return np.sum(circles[:, 2])

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    # Use multiple random initializations to find better solutions
    best_solution = None
    best_sum = 0
    
    # Try several different initialization strategies
    for seed in [42, 123, 456, 789, 999]:
        np.random.seed(seed)
        
        # Strategy 1: Hexagonal initialization
        hex_init = _hexagonal_initialization()
        
        # Strategy 2: Random initialization with some clustering
        rand_init = np.zeros((32, 3))
        for i in range(32):
            # Place in clusters to avoid too much randomness
            cluster = i // 8  # Group into 4 clusters
            x = 0.1 + 0.8 * np.random.random() + (cluster % 2) * 0.4
            y = 0.1 + 0.8 * np.random.random() + (cluster // 2) * 0.4
            r = 0.01 + 0.1 * np.random.random()
            
            # Keep within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            rand_init[i] = [x, y, r]
        
        # Try both initializations with optimization
        for init in [hex_init, rand_init]:
            # Run optimization
            optimized = _optimize_circles(init.copy())
            
            # Evaluate solution
            sum_radii = _evaluate_solution(optimized)
            
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_solution = optimized.copy()
    
    # Final refinement with more aggressive optimization
    if best_solution is not None:
        final_optimized = _optimize_circles(best_solution.copy())
        final_sum = _evaluate_solution(final_optimized)
        
        if final_sum > best_sum:
            best_solution = final_optimized
    
    # If still no solution, return default
    if best_solution is None:
        # Default initialization
        circles = np.zeros((32, 3))
        for i in range(32):
            circles[i] = [0.5, 0.5, 0.05]
        return circles
    
    return best_solution


# EVOLVE-BLOCK-END
