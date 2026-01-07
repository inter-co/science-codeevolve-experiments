# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining grid initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles using a grid-based approach for good initial placement
    circles = np.zeros((n, 3))
    
    # Create a grid pattern as initial guess
    grid_size = int(math.ceil(math.sqrt(n)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            # Initial radius - small enough to fit in grid cell
            r = min(spacing_x, spacing_y) * 0.4
            circles[idx] = [x, y, r]
            idx += 1
    
    # Ensure we have exactly 32 circles
    if idx < n:
        # Fill remaining positions with random placements
        for i in range(idx, n):
            circles[i] = [
                np.random.uniform(0.05, 0.95),
                np.random.uniform(0.05, 0.95),
                np.random.uniform(0.01, 0.1)
            ]
    
    # Define constraint functions
    def get_distance_matrix(circles_array):
        """Compute pairwise distances between circle centers"""
        centers = circles_array[:, :2]
        return cdist(centers, centers)
    
    def radius_constraint(circles_array):
        """Check if all circles satisfy containment constraints"""
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            if r > x or r > y or r > (1-x) or r > (1-y):
                return False
        return True
    
    def overlap_constraint(circles_array):
        """Check if any circles overlap"""
        dist_matrix = get_distance_matrix(circles_array)
        # Check all pairs of circles
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                dist = dist_matrix[i, j]
                r_i, r_j = circles_array[i, 2], circles_array[j, 2]
                if dist < (r_i + r_j):  # Overlapping
                    return False
        return True
    
    # Objective function to maximize (negative because minimize)
    def objective(circles_flat):
        total_radius = np.sum(circles_flat[2::3])  # Sum of all radii
        return -total_radius
    
    # Constraints
    def containment_constraint(circles_flat):
        # Each circle must be fully contained in the unit square
        circles_array = circles_flat.reshape(-1, 3)
        penalties = []
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # Penalty for violating containment
            penalty = max(0, r - x) + max(0, r - y) + max(0, r - (1-x)) + max(0, r - (1-y))
            penalties.append(penalty)
        return -np.sum(penalties)  # Negative because we want positive values
    
    def overlap_penalty(circles_flat):
        # Penalty for overlapping circles
        circles_array = circles_flat.reshape(-1, 3)
        dist_matrix = get_distance_matrix(circles_array)
        penalty = 0
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                dist = dist_matrix[i, j]
                r_i, r_j = circles_array[i, 2], circles_array[j, 2]
                if dist < (r_i + r_j):
                    # Penalty proportional to how much they overlap
                    overlap = (r_i + r_j) - dist
                    penalty += overlap**2
        return -penalty  # Negative because we want positive values
    
    # Bounds for optimization
    bounds = []
    for i in range(n):
        # x, y, r bounds
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Flatten initial circles
    initial_flat = circles.flatten()
    
    # Apply optimization with constraints
    try:
        # First, try to improve the initial solution using a simpler optimization approach
        # We'll do a coordinate descent approach first to get a better starting point
        
        # Simple local search approach
        best_circles = circles.copy()
        best_sum = np.sum(best_circles[:, 2])
        
        # Try to improve by adjusting one circle at a time
        for _ in range(1000):  # Limited iterations for performance
            improved = False
            for i in range(n):
                # Save current state
                old_circle = best_circles[i].copy()
                
                # Try small adjustments to position and radius
                step = 0.01
                
                # Try to increase radius while maintaining constraints
                test_r = old_circle[2] + step
                if test_r <= 0.499:  # Max radius constraint
                    # Check if this adjustment maintains constraints
                    temp_circles = best_circles.copy()
                    temp_circles[i, 2] = test_r
                    
                    # Check containment
                    x, y, r = temp_circles[i]
                    if (r <= x and r <= y and r <= (1-x) and r <= (1-y)):
                        # Check overlap with others
                        valid = True
                        for j in range(n):
                            if i != j:
                                x1, y1, r1 = temp_circles[i]
                                x2, y2, r2 = temp_circles[j]
                                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                                if dist < (r1 + r2):
                                    valid = False
                                    break
                        
                        if valid:
                            best_circles[i, 2] = test_r
                            improved = True
            
            if not improved:
                break
                
        # Final optimization using scipy minimize
        # Use only the most promising solution from local search
        final_circles = best_circles.copy()
        
        # Convert to flat array for scipy optimization
        x0 = final_circles.flatten()
        
        # Define constraints for scipy
        cons = []
        
        # Add containment constraint
        def contain_constraint(x):
            circles_array = x.reshape(-1, 3)
            # Return positive if all constraints satisfied
            result = []
            for i in range(len(circles_array)):
                x_pos, y_pos, r = circles_array[i]
                result.append(x_pos - r)      # x - r >= 0
                result.append(y_pos - r)      # y - r >= 0
                result.append(1 - x_pos - r)  # 1 - x - r >= 0
                result.append(1 - y_pos - r)  # 1 - y - r >= 0
            return np.array(result)
            
        cons.append({'type': 'ineq', 'fun': contain_constraint})
        
        # Add overlap constraint (simplified version)
        def overlap_constraint_func(x):
            circles_array = x.reshape(-1, 3)
            # For each pair of circles, we want their distance >= sum of radii
            # So we want: distance - (r1 + r2) >= 0
            result = []
            for i in range(len(circles_array)):
                for j in range(i+1, len(circles_array)):
                    x1, y1, r1 = circles_array[i]
                    x2, y2, r2 = circles_array[j]
                    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    result.append(dist - (r1 + r2))
            return np.array(result)
            
        cons.append({'type': 'ineq', 'fun': overlap_constraint_func})
        
        # Run optimization
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if res.success:
            optimized_circles = res.x.reshape(-1, 3)
            # Make sure the result satisfies constraints
            final_result = optimized_circles.copy()
        else:
            # If optimization failed, return our best local search result
            final_result = best_circles
            
    except Exception as e:
        # Fallback to local search result if optimization fails
        final_result = best_circles
    
    # Final validation and cleanup
    for i in range(n):
        # Ensure radius is reasonable
        final_result[i, 2] = max(0.001, min(0.499, final_result[i, 2]))
        
        # Ensure containment
        x, y, r = final_result[i]
        final_result[i] = [
            max(r, min(1-r, x)),
            max(r, min(1-r, y)),
            r
        ]
    
    return final_result


# EVOLVE-BLOCK-END
