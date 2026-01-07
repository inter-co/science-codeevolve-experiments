# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

# Global constants
N_CIRCLES = 32
BENCHMARK = 2.937944526205518

def initialize_grid_placement(n):
    """Initialize circles using a grid-based approach"""
    # Create a grid pattern for initial placement
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Adjust grid to fit within unit square with some margin
    margin = 0.05
    grid_size_x = (1 - 2*margin) / cols
    grid_size_y = (1 - 2*margin) / rows
    
    circles = []
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            x = margin + (j + 0.5) * grid_size_x
            y = margin + (i + 0.5) * grid_size_y
            # Initial radius - start with small values that can grow
            r = min(grid_size_x, grid_size_y) * 0.3
            circles.append([x, y, r])
            count += 1
            
    return np.array(circles)

def calculate_penalty(circles):
    """Calculate penalty for overlap and boundary violations"""
    penalty = 0.0
    n = len(circles)
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += 1000.0
    
    # Check overlap constraints
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Compute pairwise distances
    distances = cdist(positions, positions)
    
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            r_i, r_j = radii[i], radii[j]
            # Penalty for overlapping circles
            if dist < r_i + r_j:
                penalty += 10000.0 * (r_i + r_j - dist)**2
    
    return penalty

def objective_function(circles_flat):
    """Objective function to maximize sum of radii"""
    # Reshape flat array back to circles
    circles = circles_flat.reshape(-1, 3)
    
    # Sum of radii (negative because we're minimizing)
    total_radius = np.sum(circles[:, 2])
    
    # Add penalty for constraint violations
    penalty = calculate_penalty(circles)
    
    return -total_radius + penalty

def optimize_circles(initial_circles, max_iter=1000):
    """Optimize circle placement using scipy.optimize"""
    # Flatten initial circles for optimization
    initial_flat = initial_circles.flatten()
    
    # Define bounds for each parameter (x, y, r)
    bounds = []
    for i in range(len(initial_circles)):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds  
        bounds.append((0.001, 0.999))
        # r bounds (must be positive and respect containment)
        bounds.append((0.001, 0.499))
    
    # Use L-BFGS-B optimizer which handles bounds well
    result = minimize(
        objective_function,
        initial_flat,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6},
        tol=1e-6
    )
    
    # Reshape result back to circles
    optimized_circles = result.x.reshape(-1, 3)
    
    # Ensure all circles are valid
    for i in range(len(optimized_circles)):
        x, y, r = optimized_circles[i]
        # Clamp to valid ranges
        optimized_circles[i] = [
            max(0.001, min(0.999, x)),
            max(0.001, min(0.999, y)), 
            max(0.001, min(0.499, r))
        ]
    
    return optimized_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Initialize with grid-based placement
    initial_circles = initialize_grid_placement(N_CIRCLES)
    
    # Apply optimization
    optimized_circles = optimize_circles(initial_circles, max_iter=500)
    
    # Final refinement with a simple physics-inspired approach
    # Move circles apart if they overlap
    for _ in range(100):
        # Calculate pairwise distances
        positions = optimized_circles[:, :2]
        radii = optimized_circles[:, 2]
        
        # Check for overlaps and adjust positions
        moved = False
        for i in range(len(optimized_circles)):
            for j in range(i+1, len(optimized_circles)):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                r_i, r_j = radii[i], radii[j]
                
                if dist < r_i + r_j and dist > 0:
                    # Push circles apart
                    move_dist = (r_i + r_j - dist) * 0.1
                    dx_norm = dx / dist if dist > 0 else 0
                    dy_norm = dy / dist if dist > 0 else 0
                    
                    # Apply movement to both circles
                    optimized_circles[i, 0] += dx_norm * move_dist * 0.5
                    optimized_circles[i, 1] += dy_norm * move_dist * 0.5
                    optimized_circles[j, 0] -= dx_norm * move_dist * 0.5
                    optimized_circles[j, 1] -= dy_norm * move_dist * 0.5
                    moved = True
        
        # Keep circles within bounds
        for i in range(len(optimized_circles)):
            x, y, r = optimized_circles[i]
            optimized_circles[i] = [
                max(r, min(1-r, x)),
                max(r, min(1-r, y)),
                r
            ]
        
        if not moved:
            break
    
    return optimized_circles


# EVOLVE-BLOCK-END
