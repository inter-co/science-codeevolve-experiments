# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a grid-based initialization followed by constrained optimization with improved approach.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles using a refined grid-based approach inspired by hexagonal packing
    def initialize_grid():
        circles = np.zeros((n, 3))
        
        # Create a more sophisticated grid layout
        # For 32 circles, use roughly 6x6 grid (but adjust for better packing)
        grid_size = int(np.ceil(np.sqrt(n)))
        
        # Adjust grid size to ensure we can fit all circles
        while grid_size * grid_size < n:
            grid_size += 1
            
        # Calculate spacing
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        # Place initial circles in a grid pattern with slight randomness
        placed = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if placed >= n:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                
                # Add small random offset to avoid perfect grid artifacts
                x += np.random.uniform(-spacing_x*0.1, spacing_x*0.1)
                y += np.random.uniform(-spacing_y*0.1, spacing_y*0.1)
                
                # Initial radius - small enough to fit in grid cell
                r = min(spacing_x, spacing_y) * 0.4
                
                # Ensure we don't place too close to boundaries
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                
                circles[placed] = [x, y, r]
                placed += 1
            if placed >= n:
                break
        
        # Set remaining circles with small radii if needed
        for i in range(placed, n):
            circles[i] = [0.5, 0.5, 0.01]
            
        return circles
    
    # Create initial configuration
    circles = initialize_grid()
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(x):
        return -np.sum(x[2::3])  # Negative because we minimize
    
    # Constraint functions
    def get_constraints():
        """Return constraint functions for optimization"""
        cons = []
        
        # Boundary constraints: ensure circles stay within unit square
        def boundary_constraint(x):
            result = []
            for i in range(n):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                # r <= xi <= 1-r and r <= yi <= 1-r
                result.extend([
                    xi - ri,           # xi >= ri
                    yi - ri,           # yi >= ri
                    1 - xi - ri,       # 1-xi >= ri
                    1 - yi - ri        # 1-yi >= ri
                ])
            return np.array(result)
        
        # Non-overlap constraints: ensure minimum distance between centers
        def overlap_constraint(x):
            result = []
            for i in range(n):
                for j in range(i+1, n):
                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                    distance = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    # Distance between centers must be >= sum of radii
                    result.append(distance - ri - rj)
            return np.array(result)
        
        cons.append({'type': 'ineq', 'fun': boundary_constraint})
        cons.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return cons
    
    # Set bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Run optimization with better parameters
    try:
        # Flatten initial guess
        x0 = circles.flatten()
        
        # Get constraints
        constraints = get_constraints()
        
        # Try different optimization methods - SLSQP often works well for this problem
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1500, 'ftol': 1e-7, 'eps': 1e-7}
        )
        
        if result.success:
            # Return optimized circles
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial configuration if optimization fails
            return circles
            
    except Exception as e:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
