# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial placement via Poisson disk sampling followed by optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Generate initial candidate positions using a modified Poisson disk sampling
    # This creates a good initial distribution that avoids clustering
    def poisson_disk_sampling():
        # Grid-based approach with adaptive spacing
        min_radius = 0.01
        max_radius = 0.2
        grid_size = 0.1
        
        # Create initial grid points
        grid_points = []
        for i in range(int(1/grid_size)):
            for j in range(int(1/grid_size)):
                x = (i + 0.5) * grid_size
                y = (j + 0.5) * grid_size
                if 0 <= x <= 1 and 0 <= y <= 1:
                    grid_points.append([x, y])
        
        # Select points with appropriate spacing
        selected = []
        for point in grid_points:
            x, y = point
            # Check if point is far enough from existing points
            valid = True
            for selected_point in selected:
                dx = x - selected_point[0]
                dy = y - selected_point[1]
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < 0.1:  # Minimum distance constraint
                    valid = False
                    break
            if valid:
                selected.append(point)
        
        # Limit to n points
        selected = selected[:n]
        return selected
    
    # Phase 2: Initialize with candidate positions
    candidate_positions = poisson_disk_sampling()
    
    # Initialize with equal small radii
    initial_radii = np.full(n, 0.05)
    
    # Combine positions and radii into single array for optimization
    # Format: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = []
    for i, pos in enumerate(candidate_positions):
        initial_params.extend([pos[0], pos[1], initial_radii[i]])
    
    # Phase 3: Optimization using constrained minimization
    def objective(params):
        # Convert params back to circles array
        circles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            circles.append([x, y, r])
        
        # Calculate negative sum of radii (since we want to maximize)
        total_radius = sum(circle[2] for circle in circles)
        return -total_radius
    
    def constraint_func(params):
        # Ensure all circles fit in the unit square
        constraints = []
        
        for i in range(n):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            
            # Boundary constraints
            constraints.append(x - r)  # x >= r
            constraints.append(y - r)  # y >= r
            constraints.append(1 - x - r)  # 1-x >= r
            constraints.append(1 - y - r)  # 1-y >= r
            
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = params[3*i], params[3*i + 1], params[3*i + 2]
                x2, y2, r2 = params[3*j], params[3*j + 1], params[3*j + 2]
                
                # Distance constraint: sqrt((x1-x2)^2 + (y1-y2)^2) >= r1 + r2
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # We want dist_sq >= min_dist_sq, so we return dist_sq - min_dist_sq
                constraints.append(dist_sq - min_dist_sq)
        
        return np.array(constraints)
    
    # Define bounds for parameters (x, y, r)
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))  # Avoid boundary issues
        # y bounds  
        bounds.append((0.001, 0.999))
        # r bounds
        bounds.append((0.001, 0.499))
    
    # Use scipy's minimize with SLSQP method which handles constraints well
    try:
        # First run with relaxed constraints to get initial feasible solution
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        # Extract final solution
        final_params = result.x
        
        # Convert back to circles array
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [final_params[3*i], final_params[3*i + 1], final_params[3*i + 2]]
            
        return circles
        
    except Exception as e:
        # Fallback to simple initialization if optimization fails
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [candidate_positions[i][0], candidate_positions[i][1], 0.05]
        return circles


# EVOLVE-BLOCK-END
