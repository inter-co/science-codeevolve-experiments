# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from sklearn.cluster import KMeans
import warnings

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-phase approach: initial placement with improved grid layout, followed by 
    constrained optimization with better handling of constraints.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Better initial placement using a more systematic approach
    circles = initialize_better_placement(n)
    
    # Phase 2: Refine using optimization with improved constraint handling
    circles = optimize_circles_improved(circles)
    
    return circles

def initialize_better_placement(n: int) -> np.ndarray:
    """Initialize circle positions using a more systematic approach than simple hexagonal lattice"""
    # For 32 circles, create a more balanced rectangular arrangement with some hexagonal characteristics
    # Try to distribute circles more evenly
    
    # Use a 6x6 grid pattern (36 positions) but only use 32
    rows = 6
    cols = 6
    
    # Calculate spacing
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    # Create initial positions in a grid pattern
    circles = []
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            
            # Add slight randomness to avoid perfect grid issues
            x += np.random.uniform(-spacing_x*0.1, spacing_x*0.1)
            y += np.random.uniform(-spacing_y*0.1, spacing_y*0.1)
            
            # Ensure positions are within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius guess - start with small values that can grow
            radius = 0.05
            circles.append([x, y, radius])
            count += 1
                
        if count >= n:
            break
    
    # If we need more circles, add them randomly but with some structure
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = 0.05
        circles.append([x, y, radius])
    
    return np.array(circles)

def optimize_circles_improved(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii with improved constraint handling"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_flat = []
    for i in range(n):
        initial_flat.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    def objective(x_flat):
        # Extract positions and radii
        total_radius = 0
        for i in range(n):
            total_radius += x_flat[3*i + 2]
        return -total_radius  # Negative because we want to maximize
    
    def constraint_func(x_flat):
        # Check containment and non-overlap constraints efficiently
        constraints = []
        
        # Circle containment in unit square - more robust handling
        for i in range(n):
            x, y, r = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
            
            # Make sure radius is positive and reasonable
            constraints.append(r)
            
            # Circle must be within square boundaries (with margin)
            constraints.append(1 - r - x)  # Right boundary
            constraints.append(1 - r - y)  # Top boundary
            constraints.append(x - r)      # Left boundary
            constraints.append(y - r)      # Bottom boundary
        
        # Non-overlap constraints - more efficient pairwise checking
        # Use a simplified version that's less computationally expensive for large numbers
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
                x2, y2, r2 = x_flat[3*j], x_flat[3*j+1], x_flat[3*j+2]
                
                # Distance constraint: d >= r1 + r2
                dx = x1 - x2
                dy = y1 - y2
                distance_sq = dx*dx + dy*dy
                # To avoid sqrt computation, check if distance^2 >= (r1+r2)^2
                radius_sum = r1 + r2
                constraints.append(distance_sq - radius_sum * radius_sum)
        
        return np.array(constraints)
    
    # Set up bounds: x, y in [r, 1-r], r in [0.001, 0.45] (reasonable upper bound)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.45)])  # x, y, r bounds
    
    # Try multiple optimization approaches
    best_result = None
    best_value = float('-inf')
    
    # First attempt: SLSQP with reduced tolerance
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 500, 'ftol': 1e-4, 'eps': 1e-4}
        )
        
        if result.success:
            # Check if this solution is better
            current_total = -objective(result.x)  # Convert back to maximization
            if current_total > best_value:
                best_value = current_total
                best_result = result
    except Exception as e:
        warnings.warn(f"SLSQP failed: {e}")
    
    # Second attempt: L-BFGS-B if SLSQP fails
    if best_result is None:
        try:
            result = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-4}
            )
            
            if result.success:
                current_total = -objective(result.x)
                if current_total > best_value:
                    best_value = current_total
                    best_result = result
        except Exception as e:
            warnings.warn(f"L-BFGS-B failed: {e}")
    
    # If we have a valid result, return optimized circles
    if best_result is not None and best_result.success:
        optimized = best_result.x
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [optimized[3*i], optimized[3*i+1], optimized[3*i+2]]
        return circles
    
    # If optimization fails, return initial configuration but with some refinement
    return initial_circles.copy()


# EVOLVE-BLOCK-END
