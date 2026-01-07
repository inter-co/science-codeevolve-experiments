# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import warnings
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach: hexagonal initialization + constrained optimization with 
    efficient constraint handling and multi-start optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with hexagonal packing pattern for better starting configuration
    def initialize_hexagonal():
        circles = np.zeros((n, 3))
        
        # Hexagonal packing parameters
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Hexagonal offset for even rows
        hex_offset = spacing_x * 0.5
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 0.5) * spacing_x
                if i % 2 == 1:  # Offset odd rows
                    x += hex_offset
                y = (i + 0.5) * spacing_y
                # Initial radius - small enough to fit in grid cell
                r = min(spacing_x, spacing_y) * 0.2
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with random placements near edges
        while idx < n:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = 0.05
            circles[idx] = [x, y, r]
            idx += 1
            
        return circles
    
    # Objective function to maximize sum of radii
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Negative because we minimize
    
    # Constraint functions using explicit bounds and constraints
    def get_constraints(circles_flat):
        # Bounds for x, y, r for each circle
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Constraints list
        constraints = []
        
        # Boundary constraints for each circle
        for i in range(n):
            # x >= r and x + r <= 1
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})  # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})  # x + r <= 1
            # y >= r and y + r <= 1  
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})  # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})  # y + r <= 1
        
        # Efficient non-overlap constraints using spatial indexing
        # Build spatial tree once for all overlap checks
        positions = np.zeros((n, 2))
        for i in range(n):
            positions[i] = [circles_flat[3*i], circles_flat[3*i+1]]
        
        # Use KDTree for efficient neighbor search
        tree = cKDTree(positions)
        
        # Find neighbors within a reasonable distance (max possible radius * 2)
        # This reduces the number of overlap checks significantly
        max_radius = 0.5  # Upper bound on radius
        neighbors = tree.query_pairs(r=2*max_radius, output_type='ndarray')
        
        # Create constraints only for nearby pairs
        for i, j in neighbors:
            if i < j:  # Avoid duplicates
                def overlap_constraint(x, i=i, j=j):
                    x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                    x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    min_dist_sq = (r_i + r_j)**2
                    return dist_sq - min_dist_sq
                
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return bounds, constraints
    
    # Multi-start optimization with different initial configurations
    best_result = None
    best_sum = 0
    
    # Try multiple optimization runs with different initializations
    for attempt in range(5):  # Increased attempts for better exploration
        try:
            # Initialize with hexagonal pattern
            circles = initialize_hexagonal()
            
            # Add some noise to initial solution for diversity
            if attempt > 0:
                noise = np.random.normal(0, 0.01, circles.size)
                circles_noisy = circles.flatten() + noise
                # Clip to valid ranges
                circles_noisy[0::3] = np.clip(circles_noisy[0::3], 0.001, 0.999)  # x coordinates
                circles_noisy[1::3] = np.clip(circles_noisy[1::3], 0.001, 0.999)  # y coordinates
                circles_noisy[2::3] = np.clip(circles_noisy[2::3], 0.001, 0.499)  # radii
                circles = circles_noisy.reshape(-1, 3)
            
            x0 = circles.flatten()
            
            # Get bounds and constraints (pass current state for dynamic constraint building)
            bounds, constraints = get_constraints(x0)
            
            # Run optimization with SLSQP (better for constrained problems)
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6})
            
            if result.success:
                current_sum = -objective(result.x)  # Convert back to positive sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result.x.reshape(-1, 3)
                    
        except Exception as e:
            continue
    
    # Return best result or initial configuration
    if best_result is not None and best_result.size > 0:
        return best_result
    else:
        # Return initial hexagonal configuration if optimization fails
        return initialize_hexagonal()


# EVOLVE-BLOCK-END
