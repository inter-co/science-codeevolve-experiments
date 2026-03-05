# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a sophisticated hybrid approach combining geometric insights, multi-start optimization, and adaptive refinement.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Focus on aspect ratios that typically perform well for circle packing
    aspect_ratios = [
        (1.5, 0.5), (1.4, 0.6), (1.3, 0.7), (1.2, 0.8), (1.1, 0.9),
        (1.0, 1.0), (0.9, 1.1), (0.8, 1.2), (0.7, 1.3), (0.6, 1.4), (0.5, 1.5),
        (2.0, 0.2), (0.2, 2.0)
    ]
    
    best_sum = 0
    best_circles = None
    
    def initialize_hexagonal_pattern(w: float, h: float, n: int) -> np.ndarray:
        """Initialize using hexagonal packing pattern with better mathematical approach"""
        circles = np.zeros((n, 3))
        
        # Create hexagonal grid pattern - optimized for 21 circles
        # Using 5 rows and 4 columns as this gives good coverage
        rows = 5
        cols = 4
        
        # Calculate spacing with margin
        margin = 0.02
        actual_w = w - 2 * margin
        actual_h = h - 2 * margin
        
        # Calculate spacing to fit all circles
        spacing_x = actual_w / (cols - 1) if cols > 1 else actual_w
        spacing_y = actual_h / (rows - 1) if rows > 1 else actual_h
        
        # Use hexagonal packing with offset rows for better efficiency
        max_radius = min(spacing_x, spacing_y) * 0.3
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset for better packing
                x = margin + (j + 0.5 * (i % 2)) * spacing_x
                y = margin + i * spacing_y
                
                # Ensure within bounds and clip if needed
                x = np.clip(x, margin, w - margin)
                y = np.clip(y, margin, h - margin)
                
                # Set radius based on proximity to boundaries
                r = min(x - margin, y - margin, w - margin - x, h - margin - y) * 0.4
                r = max(0.001, min(max_radius, r))
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with strategic placement
        for i in range(idx, n):
            # Try to place near boundaries to utilize space better
            x = random.uniform(margin, w - margin)
            y = random.uniform(margin, h - margin)
            # Radius based on distance to nearest boundary
            r = min(x, y, w - x, h - y) * 0.3
            r = max(0.001, min(0.3, r))
            circles[i] = [x, y, r]
        
        return circles
    
    def objective(params):
        """Objective function to maximize sum of radii"""
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    def constraint_func(params, w, h):
        """Constraint function for optimization - vectorized for efficiency"""
        circles = params.reshape(-1, 3)
        constraints_list = []
        
        # Boundary constraints: each circle must be within rectangle
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Each constraint should be >= 0 for feasibility
            constraints_list.extend([
                x - r,           # Left boundary constraint
                y - r,           # Bottom boundary constraint  
                w - x - r,       # Right boundary constraint
                h - y - r        # Top boundary constraint
            ])
        
        # Overlap constraints: distance between centers >= sum of radii
        # Vectorized approach for better performance
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                # For no overlap, distance >= r1 + r2, so we want distance - (r1 + r2) >= 0
                constraints_list.append(distance - (r1 + r2))
        
        return np.array(constraints_list)
    
    # Try each aspect ratio with multiple optimizations
    for w, h in aspect_ratios:
        # Create initial configuration using better hexagonal pattern
        circles = initialize_hexagonal_pattern(w, h, 21)
        initial_vars = circles.flatten()
        
        # Set bounds for optimization
        bounds = []
        for i in range(21):
            bounds.extend([
                (0.001, w - 0.001),      # x bounds
                (0.001, h - 0.001),      # y bounds
                (0.001, min(w, h) * 0.5) # r bounds
            ])
        
        # Define constraints
        def constraint_wrapper(params):
            return constraint_func(params, w, h)
        
        cons = {'type': 'ineq', 'fun': constraint_wrapper}
        
        # Try optimization with different methods - prioritize robustness
        methods_to_try = ['trust-constr', 'SLSQP']
        best_local_sum = 0
        best_local_result = None
        
        # Limit optimization iterations to keep runtime reasonable
        max_iter = 1000  # Reduced from 3000 to improve speed
        
        for method in methods_to_try:
            try:
                result = minimize(
                    objective,
                    initial_vars,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-9}
                )
                
                if result.success:
                    final_circles = result.x.reshape(-1, 3)
                    total_radius = np.sum(final_circles[:, 2])
                    
                    if total_radius > best_local_sum:
                        best_local_sum = total_radius
                        best_local_result = result
                        
            except Exception:
                continue
        
        # Keep the best configuration found
        if best_local_result is not None and best_local_sum > best_sum:
            best_sum = best_local_sum
            best_circles = best_local_result.x.reshape(-1, 3)
    
    # If no good solution found, create a reasonable fallback
    if best_circles is None:
        # Use standard square with hexagonal initialization
        w, h = 1.0, 1.0
        circles = initialize_hexagonal_pattern(w, h, 21)
        best_circles = circles
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
