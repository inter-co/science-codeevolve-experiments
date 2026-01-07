# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize using a better hexagonal layout that starts closer to optimal (from inspiration 2)
    def initialize_better_hexagonal():
        circles = []
        
        # Use a 5x5 hexagonal grid with optimized spacing
        rows = 5
        cols = 5
        
        # Calculate spacing more carefully to allow for better optimization
        initial_radius = 0.08  # Slightly smaller initial radius to allow expansion
        spacing_x = 2 * initial_radius
        spacing_y = initial_radius * math.sqrt(3)
        
        # Center the pattern in the unit square
        offset_x = (1.0 - (cols - 1) * spacing_x) / 2
        offset_y = (1.0 - (rows - 1) * spacing_y) / 2
        
        # Generate hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row
                x_offset = (i % 2) * spacing_x / 2
                x = offset_x + j * spacing_x + x_offset
                y = offset_y + i * spacing_y
                
                # Add more substantial randomness to avoid getting stuck in local minima
                x += np.random.uniform(-spacing_x/8, spacing_x/8)
                y += np.random.uniform(-spacing_y/8, spacing_y/8)
                
                # Ensure we're within bounds with safety margin
                x = max(initial_radius, min(1 - initial_radius, x))
                y = max(initial_radius, min(1 - initial_radius, y))
                
                circles.append([x, y, initial_radius])
        
        # Trim or pad to exactly 26 circles
        if len(circles) > n:
            circles = circles[:n]
        elif len(circles) < n:
            # Fill remaining slots with strategic positioning
            for i in range(n - len(circles)):
                # Place in center region with more varied positions
                x = 0.2 + 0.6 * np.random.random()
                y = 0.2 + 0.6 * np.random.random()
                r = 0.05 + 0.03 * np.random.random()  # Variable initial radii
                circles.append([x, y, r])
        
        return np.array(circles)
    
    # Vectorized constraint functions for better performance (from inspiration 2)
    def constraint_containment(circles_flat):
        """Vectorized containment constraints"""
        circles = circles_flat.reshape(-1, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # All four boundary constraints in one go
        return np.concatenate([
            x - r,           # Left boundary
            1 - x - r,       # Right boundary  
            y - r,           # Bottom boundary
            1 - y - r        # Top boundary
        ])
    
    def constraint_nonoverlap(circles_flat):
        """Fully vectorized non-overlap constraints"""
        circles = circles_flat.reshape(-1, 3)
        
        # Create all pairwise comparisons using broadcasting for maximum efficiency
        n_circles = len(circles)
        if n_circles < 2:
            return np.array([])
        
        # Extract coordinates and radii
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Use broadcasting to compute all pairwise distances at once
        # Reshape for broadcasting: (n, 1) and (1, n)
        x_i = x.reshape(-1, 1)
        y_i = y.reshape(-1, 1)
        r_i = r.reshape(-1, 1)
        
        x_j = x.reshape(1, -1)
        y_j = y.reshape(1, -1)
        r_j = r.reshape(1, -1)
        
        # Compute squared distances between all pairs
        dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
        
        # Compute minimum required distances (sum of radii)
        min_dist_sq = (r_i + r_j)**2
        
        # We want dist_sq >= min_dist_sq, so we return dist_sq - min_dist_sq
        # But we only want the upper triangle (unique pairs) to avoid duplicates
        mask = np.triu(np.ones((n_circles, n_circles), dtype=bool), k=1)
        constraints = dist_sq[mask] - min_dist_sq[mask]
        
        return constraints
    
    # Multiple restarts with better strategies (from inspiration 2)
    best_result = None
    best_sum = -float('inf')
    
    # Try fewer but more strategic restarts (10 instead of 15)
    max_restarts = 10
    
    for restart in range(max_restarts):
        # Set seed for reproducibility
        np.random.seed(restart * 1000 + 42)
        
        # Initialize circles using better approach
        circles = initialize_better_hexagonal()
        
        # Define optimization bounds: [x1, y1, r1, x2, y2, r2, ...]
        bounds = []
        for i in range(n):
            # x coordinate bounds: [r, 1-r] (ensuring circle fits)
            # y coordinate bounds: [r, 1-r] (ensuring circle fits)
            # r coordinate bounds: [1e-6, 0.4] (tighter upper bound)
            bounds.extend([(1e-6, 0.999), (1e-6, 0.999), (1e-6, 0.4)])
        
        # Flatten initial circles for optimization
        initial_params = circles.flatten()
        
        # Objective function: minimize negative sum of radii (to maximize sum)
        def objective(params):
            circles_local = params.reshape(-1, 3)
            return -np.sum(circles_local[:, 2])
        
        # Set up constraints for scipy.optimize using vectorized versions
        constraints = [
            {'type': 'ineq', 'fun': constraint_containment},
            {'type': 'ineq', 'fun': constraint_nonoverlap}
        ]
        
        # Run optimization with both methods for robustness (from inspiration 2)
        methods = ['SLSQP', 'trust-constr']
        local_best_result = None
        local_best_sum = -float('inf')
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 800, 'ftol': 1e-9, 'gtol': 1e-9}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > local_best_sum:
                        local_best_sum = current_sum
                        local_best_result = result
            except Exception:
                continue  # Skip this method if it fails
        
        # Update global best if we found something better
        if local_best_result is not None and local_best_result.success:
            current_sum = -local_best_result.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = local_best_result
    
    # Return the best result found
    if best_result is not None and best_result.success:
        final_circles = best_result.x.reshape(-1, 3)
        # Ensure all circles are properly contained and valid
        for i in range(len(final_circles)):
            x, y, r = final_circles[i]
            # Clamp positions to be within safe bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            # Clamp radius to reasonable bounds
            r = max(1e-6, min(0.4, r))
            final_circles[i] = [x, y, r]
        return final_circles
    else:
        # If all optimizations fail, return the best initial configuration
        return initialize_better_hexagonal()


# EVOLVE-BLOCK-END
