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
    1. High-quality initial placement using a systematic hexagonal approach
    2. Multi-start local optimization with enhanced strategies
    3. Progressive refinement and adaptive optimization parameters
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Strategy 1: Generate multiple high-quality initial configurations
    best_circles = None
    best_sum = 0
    
    # Try several high-quality initialization strategies
    for attempt in range(3):
        # Different random seeds for variety
        random.seed(42 + attempt)
        
        # Strategy A: Optimized hexagonal with better spacing
        try:
            circles_a = _optimized_hexagonal_initialization(n)
            optimized_a = _multi_start_optimization(circles_a)
            sum_a = np.sum(optimized_a[:, 2])
            
            if sum_a > best_sum:
                best_sum = sum_a
                best_circles = optimized_a.copy()
        except Exception:
            pass
            
        # Strategy B: Grid-based with strategic perturbations
        try:
            circles_b = _grid_with_perturbation_initialization(n)
            optimized_b = _multi_start_optimization(circles_b)
            sum_b = np.sum(optimized_b[:, 2])
            
            if sum_b > best_sum:
                best_sum = sum_b
                best_circles = optimized_b.copy()
        except Exception:
            pass
    
    # Strategy 2: If we have a good starting point, apply aggressive refinement
    if best_circles is not None:
        # Apply iterative refinement with progressively tighter tolerances
        best_circles = _aggressive_refinement(best_circles)
    
    return best_circles

def _optimized_hexagonal_initialization(n: int) -> np.ndarray:
    """Create initial placement using optimized hexagonal packing"""
    # Create a hexagonal grid that's more evenly distributed
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    if rows * cols < n:
        cols += 1
    
    # Calculate spacing with better distribution
    spacing_x = 0.95 / cols  # Leave some margin
    spacing_y = 0.95 / rows
    
    circles = np.zeros((n, 3))
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Hexagonal offset for odd rows
        x_offset = col * spacing_x + (row % 2) * spacing_x * 0.5
        y_offset = row * spacing_y
        
        # Add strategic randomness to avoid symmetry issues
        x = max(0.01, min(0.99, x_offset + random.uniform(-spacing_x*0.05, spacing_x*0.05)))
        y = max(0.01, min(0.99, y_offset + random.uniform(-spacing_y*0.05, spacing_y*0.05)))
        
        # Set initial radius based on proximity to boundaries and center
        # Prefer larger radii near the center
        center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        max_radius = min(0.1, 0.5 * min(x, 1-x, y, 1-y))
        # Bias toward center for larger radii
        r = max_radius * (0.7 + 0.3 * (1 - center_dist))  
        if r < 0.01:
            r = 0.01
            
        circles[i] = [x, y, r]
    
    return circles

def _grid_with_perturbation_initialization(n: int) -> np.ndarray:
    """Create initial placement using grid with strategic perturbations"""
    side = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / side
    
    circles = np.zeros((n, 3))
    
    for i in range(n):
        row = i // side
        col = i % side
        x = col * spacing + spacing * 0.5
        y = row * spacing + spacing * 0.5
        
        # Add strategic perturbation to avoid perfect grid patterns
        perturbation_factor = 0.2
        x += random.uniform(-perturbation_factor * spacing, perturbation_factor * spacing)
        y += random.uniform(-perturbation_factor * spacing, perturbation_factor * spacing)
        
        # Keep within bounds
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        # Set initial radius
        r = min(0.1, 0.5 * min(x, 1-x, y, 1-y))
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

def _single_optimization(initial_circles: np.ndarray, max_iter: int = 2000, ftol: float = 1e-6) -> np.ndarray:
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
    
    # Try multiple optimization methods with different parameters
    methods = [
        ('SLSQP', {'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-6}),
        ('L-BFGS-B', {'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-6}),
        ('TNC', {'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-6})
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
                # Verify constraints are satisfied
                containment, overlap = _calculate_constraints(optimized_circles)
                # Allow slight numerical tolerance
                if np.all(containment >= -1e-5) and np.all(overlap >= -1e-5):
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
    
    # Run multiple optimization starts with increasing perturbation
    for start in range(8):  # 8 different starts for more thorough exploration
        # Create perturbed version of initial circles
        perturbed = initial_circles.copy()
        
        # Add varying random perturbations
        for i in range(n):
            # Larger perturbations for earlier starts, smaller for later
            perturbation_scale = 0.03 * (1.0 - start/12.0)
            perturbed[i, 0] += random.uniform(-perturbation_scale, perturbation_scale)
            perturbed[i, 1] += random.uniform(-perturbation_scale, perturbation_scale)
            perturbed[i, 2] += random.uniform(-perturbation_scale*0.5, perturbation_scale*0.5)
            
            # Keep within bounds
            perturbed[i, 0] = np.clip(perturbed[i, 0], 0.01, 0.99)
            perturbed[i, 1] = np.clip(perturbed[i, 1], 0.01, 0.99)
            perturbed[i, 2] = np.clip(perturbed[i, 2], 0.001, 0.499)
        
        # Optimize this perturbed version with higher tolerance for early attempts
        ftol = 1e-5 if start < 4 else 1e-6
        optimized = _single_optimization(perturbed, max_iter=1200, ftol=ftol)
        current_sum = np.sum(optimized[:, 2])
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = optimized
    
    return best_result

def _aggressive_refinement(initial_circles: np.ndarray) -> np.ndarray:
    """Apply aggressive refinement with progressive tightening"""
    refined = initial_circles.copy()
    
    # Multiple passes with increasingly strict optimization
    for iteration in range(5):  # Increased iterations for more thorough refinement
        # Tighten tolerances progressively
        ftol = 1e-6 if iteration >= 3 else 1e-5
        max_iter = 1200 if iteration >= 3 else 600
        
        # Apply optimization
        refined = _single_optimization(refined, max_iter=max_iter, ftol=ftol)
        
        # Add small random perturbations to escape local minima
        if iteration < 4:  # Don't perturb on last iteration
            for i in range(len(refined)):
                refined[i, 0] += random.uniform(-0.003, 0.003)  # Smaller perturbations
                refined[i, 1] += random.uniform(-0.003, 0.003)
                refined[i, 2] += random.uniform(-0.001, 0.001)
                
                # Keep within bounds
                refined[i, 0] = np.clip(refined[i, 0], 0.01, 0.99)
                refined[i, 1] = np.clip(refined[i, 1], 0.01, 0.99)
                refined[i, 2] = np.clip(refined[i, 2], 0.001, 0.499)
    
    return refined


# EVOLVE-BLOCK-END
