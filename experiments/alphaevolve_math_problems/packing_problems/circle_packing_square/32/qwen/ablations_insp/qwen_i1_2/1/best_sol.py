# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import warnings
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach: geometric initialization + constrained optimization with 
    efficient spatial indexing for overlap detection.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better geometric initialization using a more structured approach
    def initialize_geometric():
        circles = np.zeros((n, 3))
        
        # Start with a regular grid pattern with slight perturbation
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Create a more sophisticated initial placement
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Position with slight jitter
                x = (j + 0.5) * spacing_x + np.random.uniform(-0.01, 0.01)
                y = (i + 0.5) * spacing_y + np.random.uniform(-0.01, 0.01)
                
                # Initial radius - small enough to fit in grid cell
                r = min(spacing_x, spacing_y) * 0.15
                
                # Ensure valid bounds
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                r = np.clip(r, 0.001, 0.499)
                
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with random placements near edges
        while idx < n:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = np.random.uniform(0.02, 0.1)
            circles[idx] = [x, y, r]
            idx += 1
            
        return circles
    
    # Objective function to maximize sum of radii
    def objective(params):
        return -np.sum(params[2::3])  # Negative because we minimize
    
    # Constraint functions - properly bound and avoid closure issues
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
            def boundary_x_min(x, i=i):
                return x[3*i] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': boundary_x_min})  # x >= r
            
            def boundary_x_max(x, i=i):
                return 1 - x[3*i] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': boundary_x_max})  # x + r <= 1
            
            # y >= r and y + r <= 1  
            def boundary_y_min(x, i=i):
                return x[3*i+1] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': boundary_y_min})  # y >= r
            
            def boundary_y_max(x, i=i):
                return 1 - x[3*i+1] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': boundary_y_max})  # y + r <= 1
        
        # Non-overlap constraints using efficient spatial indexing
        def get_overlap_constraints():
            constraints_list = []
            
            # Create positions array for spatial queries
            positions = np.zeros((n, 2))
            for i in range(n):
                positions[i] = [circles_flat[3*i], circles_flat[3*i+1]]
            
            # Build KDTree for efficient neighbor search
            tree = cKDTree(positions)
            
            # Find neighbors within a reasonable distance (2 * max_radius)
            # This avoids creating too many constraints
            for i in range(n):
                # Find nearby points to check for overlap
                neighbors = tree.query_ball_point(positions[i], 2 * 0.5, p=np.inf)
                # Only consider pairs where j > i to avoid duplicates
                for j in neighbors:
                    if i < j:
                        def overlap_constraint(x, i=i, j=j):
                            x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                            x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                            min_dist_sq = (r_i + r_j)**2
                            return dist_sq - min_dist_sq
                        
                        constraints_list.append({'type': 'ineq', 'fun': overlap_constraint})
            
            return constraints_list
        
        # Add overlap constraints
        constraints.extend(get_overlap_constraints())
        return bounds, constraints
    
    # Multi-start optimization with different initial configurations
    best_result = None
    best_sum = 0
    
    # Try multiple optimization runs with different initializations
    for attempt in range(5):  # Increased attempts for better exploration
        try:
            # Initialize with geometric pattern
            circles = initialize_geometric()
            
            # Add some noise to initial solution for diversity
            if attempt > 0:
                # Apply noise to positions and radii
                circles_noisy = circles.copy()
                # Add small random perturbations
                circles_noisy[:, 0] += np.random.normal(0, 0.01, n)  # x coordinates
                circles_noisy[:, 1] += np.random.normal(0, 0.01, n)  # y coordinates
                circles_noisy[:, 2] += np.random.normal(0, 0.005, n)  # radii
                
                # Clip to valid ranges
                circles_noisy[:, 0] = np.clip(circles_noisy[:, 0], 0.001, 0.999)  # x coordinates
                circles_noisy[:, 1] = np.clip(circles_noisy[:, 1], 0.001, 0.999)  # y coordinates
                circles_noisy[:, 2] = np.clip(circles_noisy[:, 2], 0.001, 0.499)  # radii
                
                circles = circles_noisy
            
            x0 = circles.flatten()
            
            # Get bounds and constraints
            bounds, constraints = get_constraints(x0)
            
            # Run optimization with SLSQP (better for constrained problems)
            # Use a more conservative approach with fewer iterations to avoid timeouts
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6})
            
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
        # Return initial geometric configuration if optimization fails
        return initialize_geometric()


# EVOLVE-BLOCK-END
