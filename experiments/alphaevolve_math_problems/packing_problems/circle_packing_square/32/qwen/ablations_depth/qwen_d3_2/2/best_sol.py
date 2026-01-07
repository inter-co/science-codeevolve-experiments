# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from typing import Tuple
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining:
    1. Strategic initial placement using a hexagonal lattice with optimization
    2. Multi-start local optimization to escape local minima
    3. Careful constraint handling and validation
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Strategy 1: Multiple initializations with different approaches
    best_circles = None
    best_sum = 0
    
    # Try several initialization strategies
    strategies = [
        ('hexagonal', lambda n: _hexagonal_initialization(n)),
        ('grid', lambda n: _grid_initialization(n)),
        ('random', lambda n: _random_initialization(n))
    ]
    
    for strategy_name, init_func in strategies:
        try:
            circles = init_func(n)
            # Apply multi-start local optimization
            optimized = _multi_start_optimization(circles)
            current_sum = np.sum(optimized[:, 2])
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized.copy()
        except Exception as e:
            # If one initialization fails, continue with others
            continue
    
    # If we have a good starting point, refine it further
    if best_circles is not None:
        # Final refinement with careful optimization
        best_circles = _careful_refinement(best_circles)
    
    return best_circles

def _hexagonal_initialization(n: int) -> np.ndarray:
    """Create initial placement using hexagonal packing principles"""
    # Create a hexagonal grid
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    if rows * cols < n:
        cols += 1
    
    # Calculate spacing for hexagonal packing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Create initial positions with hexagonal offset
    circles = np.zeros((n, 3))
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Hexagonal offset for odd rows
        x_offset = col * spacing_x + (row % 2) * spacing_x * 0.5
        y_offset = row * spacing_y
        
        # Add some randomness to avoid perfect grid
        x = max(0.01, min(0.99, x_offset + random.uniform(-0.01, 0.01)))
        y = max(0.01, min(0.99, y_offset + random.uniform(-0.01, 0.01)))
        
        # Set initial radius based on available space
        r = min(0.08, 0.5 * min(x, 1-x, y, 1-y))
        if r < 0.01:
            r = 0.01
            
        circles[i] = [x, y, r]
    
    return circles

def _grid_initialization(n: int) -> np.ndarray:
    """Create initial placement using grid pattern"""
    side = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / side
    
    circles = np.zeros((n, 3))
    
    for i in range(n):
        row = i // side
        col = i % side
        x = col * spacing + spacing * 0.5
        y = row * spacing + spacing * 0.5
        r = min(0.08, 0.5 * min(x, 1-x, y, 1-y))
        if r < 0.01:
            r = 0.01
        circles[i] = [x, y, r]
    
    return circles

def _random_initialization(n: int) -> np.ndarray:
    """Create initial placement using random placement with constraints"""
    circles = np.zeros((n, 3))
    
    for i in range(n):
        x = random.uniform(0.01, 0.99)
        y = random.uniform(0.01, 0.99)
        r = min(0.08, 0.5 * min(x, 1-x, y, 1-y))
        if r < 0.01:
            r = 0.01
        circles[i] = [x, y, r]
    
    return circles

def _calculate_constraints(circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate all constraints efficiently"""
    n = len(circles)
    
    # Containment constraints
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # For containment: r <= x, y, 1-x, 1-y => 0 <= x-r, y-r, 1-x-r, 1-y-r
    containment = np.column_stack([
        x - r,      # x >= r
        y - r,      # y >= r  
        1 - x - r,  # 1-x >= r
        1 - y - r   # 1-y >= r
    ]).flatten()
    
    # Non-overlap constraints using vectorized computation
    if n > 1:
        # Use broadcasting to compute all pairwise distances efficiently
        diff = circles[:, np.newaxis, :2] - circles[np.newaxis, :, :2]
        dist_sq = np.sum(diff**2, axis=2)
        radius_sums = r[:, np.newaxis] + r[np.newaxis, :]
        overlap = dist_sq - radius_sums**2
        
        # Extract upper triangle (avoiding double counting)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        overlap_constraints = overlap[mask]
    else:
        overlap_constraints = np.array([])
    
    return containment, overlap_constraints

def _objective_function(circles_flat: np.ndarray) -> float:
    """Objective function to maximize sum of radii"""
    # Reshape flat array back to circles format
    n = len(circles_flat) // 3
    circles = circles_flat.reshape((n, 3))
    
    # Sum of all radii (we negate because we're minimizing)
    return -np.sum(circles[:, 2])

def _constraint_function(circles_flat: np.ndarray) -> np.ndarray:
    """Constraint function for scipy optimization"""
    # Reshape flat array back to circles format
    n = len(circles_flat) // 3
    circles = circles_flat.reshape((n, 3))
    
    # Get all constraints
    containment, overlap = _calculate_constraints(circles)
    
    # Return all constraints (positive values mean satisfied)
    return np.concatenate([containment, overlap])

def _single_optimization(initial_circles: np.ndarray) -> np.ndarray:
    """Perform single optimization run with careful error handling"""
    n = len(initial_circles)
    
    # Flatten initial circles for optimization
    initial_flat = initial_circles.flatten()
    
    # Define bounds for each variable (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x coordinate
        bounds.append((0.001, 0.999))  # y coordinate
        bounds.append((0.001, 0.499))  # radius (max 0.5 to avoid overlap issues)
    
    # Define constraints
    constraints = {
        'type': 'ineq',
        'fun': _constraint_function
    }
    
    # Try multiple optimization methods with higher precision and more iterations
    methods = [
        ('SLSQP', {'maxiter': 4000, 'ftol': 1e-9, 'gtol': 1e-9}),
        ('L-BFGS-B', {'maxiter': 4000, 'ftol': 1e-9}),
        ('TNC', {'maxiter': 4000, 'ftol': 1e-9})
    ]
    
    for method, options in methods:
        try:
            result = minimize(
                _objective_function,
                initial_flat,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=options
            )
            
            if result.success:
                optimized_circles = result.x.reshape((n, 3))
                # Verify constraints are satisfied with even stricter tolerance
                containment, overlap = _calculate_constraints(optimized_circles)
                # Very strict numerical tolerance to ensure feasibility
                if np.all(containment >= -1e-8) and np.all(overlap >= -1e-8):
                    return optimized_circles
                    
        except Exception:
            continue
    
    # If optimization fails, return initial placement
    return initial_circles

def _multi_start_optimization(initial_circles: np.ndarray) -> np.ndarray:
    """Run multiple optimization starts with different random perturbations"""
    n = len(initial_circles)
    best_result = initial_circles
    best_sum = np.sum(initial_circles[:, 2])
    
    # Run multiple optimization starts with increased attempts
    for _ in range(12):  # Increased from 8 to 12 for even better exploration
        # Create perturbed version of initial circles
        perturbed = initial_circles.copy()
        
        # Add larger random perturbations to encourage exploration
        for i in range(n):
            perturbed[i, 0] += random.uniform(-0.03, 0.03)
            perturbed[i, 1] += random.uniform(-0.03, 0.03)
            perturbed[i, 2] += random.uniform(-0.02, 0.02)
            
            # Keep within bounds
            perturbed[i, 0] = np.clip(perturbed[i, 0], 0.01, 0.99)
            perturbed[i, 1] = np.clip(perturbed[i, 1], 0.01, 0.99)
            perturbed[i, 2] = np.clip(perturbed[i, 2], 0.001, 0.499)
        
        # Optimize this perturbed version
        optimized = _single_optimization(perturbed)
        current_sum = np.sum(optimized[:, 2])
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = optimized
    
    return best_result

def _careful_refinement(initial_circles: np.ndarray) -> np.ndarray:
    """Apply careful refinement with progressive tightening of constraints"""
    # First do standard optimization
    refined = _single_optimization(initial_circles)
    
    # Try to improve further with additional passes
    for _ in range(5):  # Increased from 3 to 5 for more thorough refinement
        improved = _single_optimization(refined)
        if np.sum(improved[:, 2]) > np.sum(refined[:, 2]) + 1e-8:
            refined = improved
        else:
            break
    
    return refined


# EVOLVE-BLOCK-END
