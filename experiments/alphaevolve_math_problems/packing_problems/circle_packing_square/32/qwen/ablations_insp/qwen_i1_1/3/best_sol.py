# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import time
from typing import Tuple

# Global constants
N_CIRCLES = 32
BENCHMARK = 2.937944526205518

def initialize_hexagonal_packing(n: int) -> np.ndarray:
    """Initialize circle positions using hexagonal packing pattern"""
    circles = np.zeros((n, 3))
    
    # Arrange in a roughly hexagonal pattern
    rows = int(math.sqrt(n)) + 2
    cols = int(math.ceil(n / rows))
    
    # Ensure we don't exceed our circle count
    actual_rows = min(rows, int(math.ceil(math.sqrt(n))))
    actual_cols = min(cols, int(math.ceil(n / actual_rows)))
    
    # Calculate spacing
    spacing_x = 0.8 / actual_cols  # Leave 0.1 margin on each side
    spacing_y = 0.8 / actual_rows
    
    # Hexagonal offset for even rows
    hex_offset = spacing_x * 0.5
    
    idx = 0
    for i in range(actual_rows):
        for j in range(actual_cols):
            if idx >= n:
                break
            x = (j + 0.1) * spacing_x + 0.1  # Add margin
            y = (i + 0.1) * spacing_y + 0.1
            
            # Apply hexagonal offset for even rows
            if i % 2 == 0:
                x += hex_offset
                
            # Ensure we're within bounds (with margin)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Start with small radius
            r = 0.03
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with random valid placements
    for i in range(idx, n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = 0.03
        circles[i] = [x, y, r]
        
    return circles

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate sum of all circle radii"""
    return np.sum(circles[:, 2])

def check_constraints(circles: np.ndarray) -> Tuple[bool, float]:
    """Check if all circles satisfy containment and non-overlap constraints"""
    n = len(circles)
    
    # Check containment constraints - ensure each circle fits completely within the unit square
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False, 0.0
    
    # Check non-overlap constraints
    total_overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_dist_sq = (r1 + r2)**2
            if dist_sq < min_dist_sq:
                # Calculate overlap amount
                overlap = min_dist_sq - dist_sq
                total_overlap_penalty += overlap
    
    return True, total_overlap_penalty

def objective_function(params: np.ndarray) -> float:
    """Objective function for optimization - maximize sum of radii"""
    # Reshape params back into circles array
    circles = params.reshape(-1, 3)
    
    # Calculate sum of radii (we want to maximize this)
    radius_sum = calculate_radius_sum(circles)
    
    # Check constraints and apply penalty for violations
    is_valid, overlap_penalty = check_constraints(circles)
    
    if not is_valid:
        # Large penalty for constraint violations
        return -radius_sum + 1000000.0
    
    # Return negative of sum (since we minimize in scipy) plus penalty
    return -radius_sum + overlap_penalty * 1000.0

def constraint_containment(params: np.ndarray) -> np.ndarray:
    """Return containment constraint values (should be >= 0)"""
    circles = params.reshape(-1, 3)
    constraints = []
    
    for i in range(len(circles)):
        x, y, r = circles[i]
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        constraints.extend([
            x - r,           # x >= r
            1 - x - r,       # 1-x >= r  
            y - r,           # y >= r
            1 - y - r        # 1-y >= r
        ])
    
    return np.array(constraints)

def constraint_overlap(params: np.ndarray) -> np.ndarray:
    """Return overlap constraint values (should be >= 0)"""
    circles = params.reshape(-1, 3)
    constraints = []
    
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            # Distance >= r1 + r2 (so we want: distance - (r1 + r2) >= 0)
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            constraints.append(dist - (r1 + r2))
    
    return np.array(constraints)

def optimized_circle_packing() -> np.ndarray:
    """Main optimization function using scipy minimize with proper constraints"""
    
    # Start with good initial configuration
    circles = initialize_hexagonal_packing(N_CIRCLES)
    
    # Extract parameters for optimization (x, y, r for each circle)
    initial_params = circles.flatten()
    
    # Define bounds for each parameter (x, y, r) for each circle
    bounds = []
    for i in range(N_CIRCLES):
        # x bounds: [0.05, 0.95] to keep margin
        # y bounds: [0.05, 0.95] to keep margin  
        # r bounds: [0.001, 0.45] to prevent degenerate cases
        bounds.extend([(0.05, 0.95), (0.05, 0.95), (0.001, 0.45)])
    
    # Set up constraints for scipy minimize
    cons = [
        {'type': 'ineq', 'fun': lambda p: constraint_containment(p)},
        {'type': 'ineq', 'fun': lambda p: constraint_overlap(p)}
    ]
    
    # Optimization settings
    options = {
        'maxiter': 500,
        'ftol': 1e-6,
        'gtol': 1e-6
    }
    
    try:
        # Use SLSQP optimizer which handles constraints well
        result = minimize(
            objective_function,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options=options,
            tol=1e-6
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            return final_circles
        else:
            # Fallback to initial configuration if optimization fails
            return circles
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        print(f"Optimization failed: {e}")
        return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Run the optimized circle packing algorithm
    circles = optimized_circle_packing()
    
    end_time = time.time()
    
    # Final validation and cleanup
    _, overlap_penalty = check_constraints(circles)
    if overlap_penalty > 0.01:  # If there are significant overlaps, try refining
        # Try a simple refinement approach
        refined_circles = circles.copy()
        for _ in range(50):
            improved = False
            for i in range(N_CIRCLES):
                # Try to slightly adjust each circle
                x, y, r = refined_circles[i]
                # Try to slightly increase radius while respecting constraints
                test_r = min(r * 1.05, 0.45)
                temp_circles = refined_circles.copy()
                temp_circles[i, 2] = test_r
                
                if check_constraints(temp_circles)[0]:  # Valid configuration
                    refined_circles = temp_circles
                    improved = True
            if not improved:
                break
        circles = refined_circles
    
    # Ensure all circles are valid
    for i in range(N_CIRCLES):
        x, y, r = circles[i]
        # Clamp values to valid ranges
        circles[i] = [
            max(0.001, min(0.999, x)),
            max(0.001, min(0.999, y)),
            max(0.001, min(0.45, r))
        ]
    
    return circles


# EVOLVE-BLOCK-END
