# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining systematic initialization with advanced optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Try fewer, more carefully selected aspect ratios to save time
    # Focus on ratios that have shown to work well in practice
    aspect_ratios = [
        (1.0, 1.0),      # Square - most common and often optimal
        (1.2, 0.8),      # 3:2 ratio  
        (0.8, 1.2),      # 2:3 ratio
        (1.5, 0.5),      # 3:1 ratio
        (0.5, 1.5),      # 1:3 ratio
        (1.3, 0.7),      # Close to golden ratio
        (0.7, 1.3),      # Reverse golden ratio
        (1.6, 0.4),      # Extreme aspect ratio
        (0.4, 1.6),      # Reverse extreme
        (2.0, 0.2),      # Very wide
        (0.2, 2.0),      # Very tall
        (1.1, 0.9),      # Nearly square
        (0.9, 1.1),      # Reverse nearly square
    ]
    
    best_sum = 0
    best_result = None
    
    # Try multiple optimization attempts per aspect ratio to increase chances
    for width_ratio, height_ratio in aspect_ratios:
        # Normalize to make perimeter = 4
        width = 2 * width_ratio / (width_ratio + height_ratio)
        height = 2 * height_ratio / (width_ratio + height_ratio)
        
        # Single attempt per aspect ratio for efficiency
        # Initialize with a hexagonal packing pattern
        circles = initialize_hexagonal_pattern(width, height, 21)
        
        # Optimize using scipy minimize with constraints
        optimized_circles = optimize_circles(circles, width, height)
        
        # Calculate sum of radii
        sum_radii = sum(circle[2] for circle in optimized_circles)
        
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_result = optimized_circles
    
    # Convert to numpy array
    result = np.zeros((21, 3))
    for i, (x, y, r) in enumerate(best_result):
        result[i] = [x, y, r]
    
    return result


def initialize_hexagonal_pattern(width, height, n):
    """Initialize circles in a hexagonal pattern for better density."""
    # Estimate a good starting radius based on area
    total_area = width * height
    # For 21 circles, assume we want ~85% coverage for good initial packing
    target_area = total_area * 0.85
    avg_circle_area = target_area / n
    max_radius = np.sqrt(avg_circle_area / np.pi) * 1.1  # Add some margin
    
    # Determine grid dimensions for hexagonal packing
    # In hexagonal packing, the area per circle is approximately 2*sqrt(3)*r^2
    rows = max(1, int(height / (max_radius * 1.732)))  # sqrt(3) ≈ 1.732
    cols = max(1, int(width / (max_radius * 1.5)))
    
    # If not enough circles, adjust to ensure we get enough
    if rows * cols < n:
        # Try a square-like arrangement
        side = int(np.ceil(np.sqrt(n)))
        rows = side
        cols = side
        # Recalculate radius based on this arrangement
        max_radius = min(width/cols, height/rows) * 0.45
    
    # Ensure we don't exceed bounds
    actual_rows = min(rows, int(height / (max_radius * 1.732)) + 2)
    actual_cols = min(cols, int(width / (max_radius * 1.5)) + 2)
    
    # Generate circles in hexagonal pattern
    circles = []
    row_spacing = height / (actual_rows + 1)
    col_spacing = width / (actual_cols + 1)
    
    # Create a more careful hexagonal pattern with reduced randomness
    for i in range(actual_rows):
        for j in range(actual_cols):
            if len(circles) >= n:
                break
                
            # Hexagonal offset for even/odd rows
            x_offset = 0 if i % 2 == 0 else col_spacing * 0.5
            x = (j + 0.5) * col_spacing + x_offset
            y = (i + 0.5) * row_spacing
            
            # Add minimal randomization to avoid perfect patterns
            x += np.random.uniform(-col_spacing*0.05, col_spacing*0.05)
            y += np.random.uniform(-row_spacing*0.05, row_spacing*0.05)
            
            # Ensure within bounds
            x = max(max_radius, min(width - max_radius, x))
            y = max(max_radius, min(height - max_radius, y))
            
            circles.append([x, y, max_radius])
    
    # Fill up to n circles if needed with better placements
    while len(circles) < n:
        # Place in a more strategic way rather than purely random
        x = np.random.uniform(max_radius, width - max_radius)
        y = np.random.uniform(max_radius, height - max_radius)
        # Use a slightly larger radius to encourage expansion
        radius = max_radius * (0.7 + np.random.random() * 0.5)
        circles.append([x, y, radius])
    
    return circles[:n]


def optimize_circles(initial_circles, width, height):
    """
    Optimize circle positions and radii using constrained optimization.
    """
    n = len(initial_circles)
    # Flatten initial configuration for optimization
    initial_flat = []
    for x, y, r in initial_circles:
        initial_flat.extend([x, y, r])
    
    def objective(params):
        # Extract parameters
        circles = []
        for i in range(n):
            x = params[i*3]
            y = params[i*3 + 1]
            r = params[i*3 + 2]
            circles.append([x, y, r])
        
        # Return negative sum of radii (we want to maximize)
        return -sum(circle[2] for circle in circles)
    
    def constraint_func(params):
        # Ensure all circles are within bounds
        circles = []
        for i in range(n):
            x = params[i*3]
            y = params[i*3 + 1]
            r = params[i*3 + 2]
            circles.append([x, y, r])
        
        # Constraint: circles must be within rectangle bounds
        bounds_constraints = []
        for i in range(n):
            x, y, r = circles[i]
            bounds_constraints.extend([
                x - r,  # x - r >= 0
                width - x - r,  # width - x - r >= 0
                y - r,  # y - r >= 0
                height - y - r  # height - y - r >= 0
            ])
        
        # Constraint: no overlaps
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)
                # Distance between centers should be >= sum of radii
                overlap_constraints.append(distance - (r1 + r2))
        
        return np.array(bounds_constraints + overlap_constraints)
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, width - 0.001))
        # y bounds  
        bounds.append((0.001, height - 0.001))
        # r bounds - make sure it's reasonable relative to container size
        max_r = min(width, height) * 0.49
        bounds.append((0.001, max_r))
    
    # Try multiple optimization methods for robustness with more aggressive tolerances
    # First try trust-constr which often handles constraints better
    try:
        result = minimize(objective, initial_flat, method='trust-constr', 
                         bounds=bounds, constraints={'type': 'ineq', 'fun': constraint_func},
                         options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10})
        
        if result.success:
            # Extract optimized circles
            optimized_circles = []
            for i in range(n):
                x = result.x[i*3]
                y = result.x[i*3 + 1]
                r = result.x[i*3 + 2]
                optimized_circles.append([x, y, r])
            return optimized_circles
    except Exception:
        pass
    
    # If trust-constr fails, try SLSQP with tighter tolerances
    try:
        result = minimize(objective, initial_flat, method='SLSQP', 
                         bounds=bounds, constraints={'type': 'ineq', 'fun': constraint_func},
                         options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10})
        
        if result.success:
            # Extract optimized circles
            optimized_circles = []
            for i in range(n):
                x = result.x[i*3]
                y = result.x[i*3 + 1]
                r = result.x[i*3 + 2]
                optimized_circles.append([x, y, r])
            return optimized_circles
    except Exception:
        pass
    
    # If all optimization methods fail, return initial configuration
    return initial_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
