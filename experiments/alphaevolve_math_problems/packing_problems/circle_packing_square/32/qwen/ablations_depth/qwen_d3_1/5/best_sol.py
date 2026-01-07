# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from typing import Tuple
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses an evolutionary approach combining geometric initialization and advanced optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Try multiple initialization strategies and pick the best
    best_circles = None
    best_sum = 0
    
    # Strategy 1: Hexagonal grid initialization
    circles1 = initialize_hexagonal_grid(n)
    circles1 = optimize_with_local_search(circles1, max_iterations=100)
    sum1 = np.sum(circles1[:, 2])
    
    # Strategy 2: Random initialization with constraint satisfaction
    circles2 = initialize_random_satisfying(n)
    circles2 = optimize_with_local_search(circles2, max_iterations=100)
    sum2 = np.sum(circles2[:, 2])
    
    # Strategy 3: Grid-based with perturbation
    circles3 = initialize_grid_perturbed(n)
    circles3 = optimize_with_local_search(circles3, max_iterations=100)
    sum3 = np.sum(circles3[:, 2])
    
    # Strategy 4: Better hexagonal with optimized spacing
    circles4 = initialize_better_hexagonal(n)
    circles4 = optimize_with_local_search(circles4, max_iterations=100)
    sum4 = np.sum(circles4[:, 2])
    
    # Pick the best initialization
    if sum1 > best_sum:
        best_sum = sum1
        best_circles = circles1.copy()
    if sum2 > best_sum:
        best_sum = sum2
        best_circles = circles2.copy()
    if sum3 > best_sum:
        best_sum = sum3
        best_circles = circles3.copy()
    if sum4 > best_sum:
        best_sum = sum4
        best_circles = circles4.copy()
    
    # Final optimization with SLSQP
    if best_circles is not None:
        best_circles = optimize_with_slsqp(best_circles)
    
    # If everything failed, return the best of our attempts
    if best_circles is not None:
        return best_circles
    else:
        # Fallback to hexagonal grid
        return initialize_hexagonal_grid(n)

def initialize_hexagonal_grid(n: int) -> np.ndarray:
    """Initialize using a hexagonal grid pattern."""
    circles = np.zeros((n, 3))
    
    # Determine grid dimensions
    rows = int(math.ceil(math.sqrt(n * 2 / 3)))
    cols = int(math.ceil(n / rows))
    
    # Adjust to ensure we have enough space
    while rows * cols < n:
        rows += 1
    
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Hexagonal offset for even rows
    hex_offset = spacing_x * 0.5
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Position calculation with hexagonal offset
        x = (col + 0.5) * spacing_x
        y = (row + 0.5) * spacing_y
        
        # Apply hexagonal offset for even rows
        if row % 2 == 0:
            x += hex_offset
            
        # Ensure valid positioning within unit square with margin
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        
        # Initial radius - start small
        r = 0.03
        
        circles[i] = [x, y, r]
    
    return circles

def initialize_better_hexagonal(n: int) -> np.ndarray:
    """Initialize with a better hexagonal pattern that's more optimized."""
    circles = np.zeros((n, 3))
    
    # Try to create a more efficient hexagonal pattern
    # This is based on known good packings for 32 circles
    sqrt_n = int(np.ceil(np.sqrt(n)))
    
    # Create a slightly denser hexagonal pattern
    spacing_x = 1.0 / (sqrt_n + 0.5)
    spacing_y = 1.0 / (sqrt_n + 0.5) * np.sqrt(3) / 2
    
    idx = 0
    for i in range(sqrt_n + 1):
        for j in range(sqrt_n + 1):
            if idx >= n:
                break
            # Position with hexagonal offset
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Apply hexagonal offset for odd rows
            if i % 2 == 1:
                x += spacing_x * 0.5
                
            # Clamp to valid range
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            # Initial radius
            r = 0.03
            
            circles[idx] = [x, y, r]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill any remaining positions
    for i in range(idx, n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = 0.03
        circles[i] = [x, y, r]
    
    return circles

def initialize_random_satisfying(n: int) -> np.ndarray:
    """Initialize with random positions that satisfy basic constraints."""
    circles = np.zeros((n, 3))
    
    for i in range(n):
        # Keep trying until we get a valid position
        attempts = 0
        valid = False
        while not valid and attempts < 100:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = 0.03
            
            # Check if this position is valid (will be checked during optimization)
            circles[i] = [x, y, r]
            valid = True
            attempts += 1
            
        if not valid:
            # Fallback to a deterministic placement
            row = i // 8
            col = i % 8
            x = 0.1 + col * 0.11
            y = 0.1 + row * 0.11
            r = 0.03
            circles[i] = [x, y, r]
    
    return circles

def initialize_grid_perturbed(n: int) -> np.ndarray:
    """Initialize using a regular grid with small random perturbations."""
    circles = np.zeros((n, 3))
    
    # Create a grid layout
    sqrt_n = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / sqrt_n
    
    idx = 0
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if idx >= n:
                break
            # Base grid position
            x = (j + 0.5) * spacing
            y = (i + 0.5) * spacing
            
            # Add small random perturbation
            perturbation = 0.05 * spacing
            x += random.uniform(-perturbation, perturbation)
            y += random.uniform(-perturbation, perturbation)
            
            # Clamp to valid range
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            # Initial radius
            r = 0.03
            
            circles[idx] = [x, y, r]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining slots if needed
    for i in range(idx, n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = 0.03
        circles[i] = [x, y, r]
    
    return circles

def optimize_with_local_search(initial_circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Perform local search optimization to improve the configuration."""
    n = len(initial_circles)
    refined_circles = initial_circles.copy()
    
    # Multiple passes of local optimization
    for pass_num in range(5):  # More passes
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try to improve each circle
            for i in range(n):
                old_radius = refined_circles[i, 2]
                
                # Calculate maximum possible radius for this circle
                max_radius = min(
                    refined_circles[i, 0],          # distance to left boundary
                    refined_circles[i, 1],          # distance to bottom boundary
                    1 - refined_circles[i, 0],      # distance to right boundary
                    1 - refined_circles[i, 1]       # distance to top boundary
                )
                
                # Check overlap constraints with all other circles
                valid_radius = max_radius
                for j in range(n):
                    if i != j:
                        dist = np.sqrt(
                            (refined_circles[i, 0] - refined_circles[j, 0])**2 +
                            (refined_circles[i, 1] - refined_circles[j, 1])**2
                        )
                        if dist > 0:
                            min_dist = refined_circles[i, 2] + refined_circles[j, 2]
                            if dist < min_dist:
                                # Need to reduce radius to avoid overlap
                                max_radius_for_overlap = dist - refined_circles[j, 2]
                                valid_radius = min(valid_radius, max_radius_for_overlap)
                
                # Update if we can increase the radius
                if valid_radius > old_radius:
                    new_radius = min(valid_radius, max_radius)
                    refined_circles[i, 2] = new_radius
                    improved = True
        
        # Early stopping if no improvement
        if not improved:
            break
    
    return refined_circles

def optimize_with_slsqp(initial_circles: np.ndarray) -> np.ndarray:
    """Use SLSQP optimization for final refinement."""
    n = len(initial_circles)
    refined_circles = initial_circles.copy()
    
    try:
        # Flatten parameters
        initial_params = refined_circles.flatten()
        
        # Set bounds for parameters
        bounds = []
        for i in range(n):
            # x, y bounds (slightly inside unit square)
            bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.001, 0.49)])
        
        # Define constraints function
        def constraint_func(params):
            circles = params.reshape(-1, 3)
            constraints = []
            
            # Boundary constraints (positive when satisfied)
            for i in range(n):
                x, y, r = circles[i]
                max_r = min(x, y, 1-x, 1-y)
                constraints.append(max_r - r)
            
            # Overlap constraints (negative when satisfied)
            positions = circles[:, :2]
            distances = cdist(positions, positions)
            for i in range(n):
                for j in range(i+1, n):
                    if i != j:
                        dist = distances[i, j]
                        min_dist = circles[i, 2] + circles[j, 2]
                        constraints.append(dist - min_dist)
                        
            return np.array(constraints)
        
        # Objective function (minimize negative sum of radii)
        def obj_func(params):
            circles = params.reshape(-1, 3)
            return -np.sum(circles[:, 2])
        
        # Optimize with SLSQP - more aggressive settings
        result = minimize(
            obj_func,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 200, 'ftol': 1e-8, 'eps': 1e-8, 'iprint': -1}
        )
        
        if result.success:
            refined_circles = result.x.reshape(-1, 3)
            
    except Exception:
        # If optimization fails, return the best we have so far
        pass
    
    return refined_circles


# EVOLVE-BLOCK-END
