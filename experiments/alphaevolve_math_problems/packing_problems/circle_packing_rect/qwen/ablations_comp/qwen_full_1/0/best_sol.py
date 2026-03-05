# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import time


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid mathematical programming approach with improved initialization and constraint handling.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Try different aspect ratios that might work well
    best_width = 1.5
    best_height = 0.5
    best_circles = None
    best_sum = 0
    
    # Try several aspect ratios that have shown success in inspirations
    ratios_to_try = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for ratio in ratios_to_try:
        width = 2.0 * ratio / (1.0 + ratio)
        height = 2.0 - width
        
        # Initialize with hexagonal pattern (better than random)
        circles = initialize_hexagonal_pattern(width, height, 21)
        
        # Try optimization with different methods
        optimized_circles = optimize_with_mathematical_approach(circles, width, height)
        current_sum = np.sum(optimized_circles[:, 2])
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_width = width
            best_height = height
            best_circles = optimized_circles.copy()
    
    # Also try the best square configuration (1.0 x 1.0) as fallback
    width, height = 1.0, 1.0
    circles = initialize_hexagonal_pattern(width, height, 21)
    optimized_circles = optimize_with_mathematical_approach(circles, width, height)
    current_sum = np.sum(optimized_circles[:, 2])
    
    if current_sum > best_sum:
        best_sum = current_sum
        best_width = width
        best_height = height
        best_circles = optimized_circles.copy()
    
    return best_circles


def initialize_hexagonal_pattern(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circle positions using a hexagonal lattice pattern."""
    circles = np.zeros((n, 3))
    
    # For 21 circles, arrange in roughly 5 rows x 4 columns with offset rows
    rows = 5
    cols = 4
    
    # Calculate spacing based on available space
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)
    
    # Adjust for hexagonal packing efficiency
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    # Place circles in hexagonal pattern
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
                
            # Offset every other row for hexagonal packing
            offset = spacing_x * 0.5 if row % 2 == 1 else 0.0
            
            x = (col + 1) * spacing_x + offset
            y = (row + 1) * spacing_y
            
            # Ensure within bounds with safety margin
            safe_margin = spacing_x * 0.2
            x = max(safe_margin, min(width - safe_margin, x))
            y = max(safe_margin, min(height - safe_margin, y))
            
            # Reasonable initial radius
            r = min(spacing_x, spacing_y) * 0.4
            
            circles[idx] = [x, y, r]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining circles strategically
    np.random.seed(42)
    for i in range(idx, n):
        # Better random positioning
        x = random.uniform(spacing_x * 0.5, width - spacing_x * 0.5)
        y = random.uniform(spacing_y * 0.5, height - spacing_y * 0.5)
        r = random.uniform(0.01, min(spacing_x, spacing_y) * 0.3)
        circles[i] = [x, y, r]
    
    return circles


def optimize_with_mathematical_approach(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Use mathematical optimization approach with better constraint handling."""
    
    n = len(initial_circles)
    
    # Flatten the initial configuration
    initial_flat = initial_circles.flatten()
    
    # Define bounds for variables (positions and radii)
    bounds = []
    for i in range(n):
        # x coordinate
        bounds.append((0.001, width - 0.001))
        # y coordinate  
        bounds.append((0.001, height - 0.001))
        # radius (must be positive)
        bounds.append((0.001, min(width, height) / 2 - 0.001))
    
    # Objective function: minimize negative sum of radii (i.e., maximize sum of radii)
    def objective(x_flat):
        circles = x_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    # Constraint function: ensure all constraints are satisfied
    def constraint_func(x_flat):
        circles = x_flat.reshape(-1, 3)
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Boundary constraints
        boundary_constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Circle must be within bounds (with a safety margin)
            boundary_constraints.extend([
                x - r,           # x >= r
                width - x - r,   # width - x >= r
                y - r,           # y >= r
                height - y - r   # height - y >= r
            ])
        
        # Overlap constraints (distance between centers >= sum of radii)
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                # Add small epsilon to prevent numerical issues
                overlap_constraints.append(distance - (radii[i] + radii[j]) + 1e-8)
        
        return np.array(boundary_constraints + overlap_constraints)
    
    # Use a more robust optimization approach with multiple attempts
    try:
        # Try multiple optimization approaches to find better solution
        best_result = None
        best_objective_value = float('inf')
        
        # Try with different optimizers
        for method in ['SLSQP', 'trust-constr']:
            try:
                result = minimize(
                    objective,
                    initial_flat,
                    method=method,
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                if result.success:
                    # Validate the result
                    current_obj = result.fun
                    if current_obj < best_objective_value:
                        best_objective_value = current_obj
                        best_result = result
            except Exception:
                continue
        
        if best_result is not None and best_result.success:
            optimized_circles = best_result.x.reshape(-1, 3)
            return validate_and_refine(optimized_circles, width, height)
        else:
            # Fall back to initial configuration if optimization fails
            return validate_and_refine(initial_circles, width, height)
            
    except Exception:
        # If optimization fails due to any reason, return validated initial
        return validate_and_refine(initial_circles, width, height)


def validate_and_refine(circles, width, height):
    """Refine the solution to ensure all constraints are met and improve quality."""
    # Make a copy to work with
    refined = circles.copy()
    
    # Ensure all circles fit within bounds
    for i in range(len(refined)):
        x, y, r = refined[i]
        # Clip positions to ensure circles are within bounds
        x = np.clip(x, r, width - r)
        y = np.clip(y, r, height - r)
        refined[i] = [x, y, r]
    
    # Perform a few rounds of improvement through local search
    for _ in range(100):
        improved = False
        for i in range(len(refined)):
            # Try to increase radius while maintaining constraints
            x, y, r = refined[i]
            
            # Calculate maximum possible radius
            max_radius = min(x, width - x, y, height - y)
            
            # Check overlap constraints with other circles
            for j in range(len(refined)):
                if i != j:
                    dx = x - refined[j, 0]
                    dy = y - refined[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    max_radius = min(max_radius, distance - refined[j, 2])
            
            # Increase radius if beneficial
            if max_radius > r and max_radius > 0:
                refined[i, 2] = max_radius
                improved = True
        
        # If no improvements were made, stop
        if not improved:
            break
    
    # Final validation
    for i in range(len(refined)):
        x, y, r = refined[i]
        refined[i] = [np.clip(x, r, width - r), np.clip(y, r, height - r), r]
    
    return refined


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
