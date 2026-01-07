# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies with robust optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with hexagonal packing pattern for better starting configuration
    def initialize_hexagonal():
        # Hexagonal packing in a square
        # Calculate grid parameters for approximately 32 circles
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        
        # Create hexagonal grid
        circles = []
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # For hexagonal packing, offset every other row
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                # Initial radius estimate - smaller than spacing to allow for optimization
                r = min(x, 1-x, y, 1-y) * 0.4
                if r > 0:
                    circles.append([x, y, r])
        
        # Ensure we have exactly 32 circles
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.05])
            
        return np.array(circles[:n])
    
    # Alternative initialization using spiral pattern for better coverage
    def initialize_spiral():
        circles = []
        # Spiral pattern from center outward
        angle_step = 0.3
        radius_step = 0.1
        max_radius = 0.4
        angle = 0
        radius = 0.1
        
        while len(circles) < n:
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            r = min(0.05, 0.5 * min(1-x, x, 1-y, y))
            if 0 <= x <= 1 and 0 <= y <= 1 and r > 0:
                circles.append([x, y, r])
            angle += angle_step
            if angle >= 2 * math.pi:
                angle = 0
                radius += radius_step
                if radius > max_radius:
                    break
                    
        # Fill remaining positions with random circles
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = min(0.05, 0.5 * min(1-x, x, 1-y, y))
            circles.append([x, y, r])
        
        return np.array(circles)
    
    # Better initialization using a more structured approach
    def initialize_better():
        # Create a grid-like pattern with some randomness for better distribution
        circles = []
        
        # Grid dimensions
        grid_size = int(math.ceil(math.sqrt(n)))
        if grid_size * grid_size < n:
            grid_size += 1
            
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        # Place circles in a grid pattern
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Add slight randomness to avoid perfect grid
                x += random.uniform(-spacing_x*0.1, spacing_x*0.1)
                y += random.uniform(-spacing_y*0.1, spacing_y*0.1)
                
                # Ensure within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                
                # Initial radius - based on proximity to boundaries
                r = min(x, 1-x, y, 1-y) * 0.3
                if r > 0.001:
                    circles.append([x, y, r])
                    count += 1
            if count >= n:
                break
        
        # Fill any remaining slots
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = min(0.05, 0.5 * min(1-x, x, 1-y, y))
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Objective function - maximize sum of radii (minimize negative sum)
    def objective(params):
        # params is flattened array [x1, y1, r1, x2, y2, r2, ...]
        radii = params[2::3]  # Every third element starting from index 2
        return -np.sum(radii)
    
    # Constraints - using scipy-friendly format
    def constraint_radius_bounds(params):
        # Ensure each circle is within the unit square with its radius
        result = []
        for i in range(n):
            idx = 3 * i
            x, y, r = params[idx], params[idx+1], params[idx+2]
            
            # Circle must be within bounds with radius
            result.extend([
                r,                      # r >= 0
                x - r,                  # x >= r
                1 - x - r,              # x + r <= 1
                y - r,                  # y >= r
                1 - y - r               # y + r <= 1
            ])
        return np.array(result)
    
    def constraint_no_overlap(params):
        # Ensure no two circles overlap
        result = []
        for i in range(n):
            for j in range(i+1, n):
                idx_i = 3 * i
                idx_j = 3 * j
                xi, yi, ri = params[idx_i], params[idx_i+1], params[idx_i+2]
                xj, yj, rj = params[idx_j], params[idx_j+1], params[idx_j+2]
                
                # Distance squared between centers
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                
                # Overlap condition: distance < sum of radii (we want >=)
                # To avoid numerical issues, we ensure distance >= sum of radii + small epsilon
                overlap = dist_sq - (ri + rj)**2
                result.append(overlap)  # Should be >= 0 for non-overlapping
        return np.array(result)
    
    # Try multiple initialization strategies and pick the best
    initial_configs = []
    
    # Try hexagonal initialization
    try:
        hex_config = initialize_hexagonal()
        initial_configs.append(hex_config)
    except Exception:
        pass
    
    # Try spiral initialization
    try:
        spiral_config = initialize_spiral()
        initial_configs.append(spiral_config)
    except Exception:
        pass
    
    # Try better initialization
    try:
        better_config = initialize_better()
        initial_configs.append(better_config)
    except Exception:
        pass
    
    # If no good initial configs, create a basic one
    if not initial_configs:
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [0.5, 0.5, 0.05]
        initial_configs.append(circles)
    
    best_result = None
    best_sum = 0
    
    # Test each initialization
    for initial_config in initial_configs:
        # Flatten for optimization
        initial_params = initial_config.flatten()
        
        # Set up bounds for [x1,y1,r1,x2,y2,r2,...]
        bounds = []
        for i in range(n):
            bounds.extend([
                (0.001, 0.999),  # x coordinates
                (0.001, 0.999),  # y coordinates
                (0.001, 0.499)   # radii
            ])
        
        # Define constraints
        cons = [
            {'type': 'ineq', 'fun': constraint_radius_bounds},
            {'type': 'ineq', 'fun': constraint_no_overlap}
        ]
        
        try:
            # Run optimization with robust settings and different methods
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6},
                tol=1e-6
            )
            
            # Extract optimized solution
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = optimized_circles
        except Exception:
            continue
    
    # If we found a good result, return it; otherwise, return the best initial config
    if best_result is not None:
        return best_result
    else:
        # Return the best initial configuration if optimization failed
        return initial_configs[0]


# EVOLVE-BLOCK-END
