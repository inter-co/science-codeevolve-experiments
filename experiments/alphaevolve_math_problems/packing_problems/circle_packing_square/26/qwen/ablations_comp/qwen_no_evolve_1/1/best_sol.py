# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal packing initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize using hexagonal packing pattern for good starting configuration
    def initialize_hexagonal_packing():
        # Create initial positions using hexagonal lattice
        circles = np.zeros((n, 3))
        
        # Hexagonal packing parameters
        sqrt3 = math.sqrt(3)
        # Try to fit circles in a hexagonal pattern
        rows = int(math.sqrt(n / (sqrt3/2)) * 1.2)  # Adjusted for better packing
        cols = max(1, int(n / rows) + 1)
        
        # Ensure we don't exceed n circles
        actual_n = min(rows * cols, n)
        
        # Calculate spacing
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Create hexagonal pattern
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row
                x_offset = (i % 2) * spacing_x / 2
                x = (j * spacing_x) + x_offset + spacing_x/2
                y = (i * spacing_y) + spacing_y/2
                
                # Ensure we're within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius estimate - based on available space
                    max_radius = min(x, 1-x, y, 1-y)
                    circles[idx] = [x, y, max_radius * 0.4]  # Start with smaller radius
                    idx += 1
                else:
                    continue
            if idx >= n:
                break
        
        # Fill remaining slots if needed
        while idx < n:
            # Place remaining circles randomly within valid bounds
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            max_radius = min(x, 1-x, y, 1-y)
            circles[idx] = [x, y, max_radius * 0.3]
            idx += 1
            
        return circles
    
    # Initialize
    circles = initialize_hexagonal_packing()
    
    # Define constraint functions for scipy optimization
    def constraint_distance(circles_flat):
        """Ensure no overlapping circles"""
        # Reshape flat array back to circles
        circles_array = circles_flat.reshape(-1, 3)
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        # Compute pairwise distances
        dist_matrix = cdist(positions, positions)
        n_circles = len(positions)
        
        # For each pair of circles, enforce minimum distance
        constraints = []
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                # Distance constraint: d >= r_i + r_j
                min_dist = radii[i] + radii[j]
                actual_dist = dist_matrix[i, j]
                # Negative value indicates violation
                constraints.append(actual_dist - min_dist)
        
        return np.array(constraints)
    
    def constraint_bounds(circles_flat):
        """Ensure all circles stay within the unit square"""
        circles_array = circles_flat.reshape(-1, 3)
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        constraints = []
        
        # Boundary constraints: r <= x <= 1-r and r <= y <= 1-r
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # x - r >= 0
            constraints.append(x - r)
            # 1 - x - r >= 0  
            constraints.append(1 - x - r)
            # y - r >= 0
            constraints.append(y - r)
            # 1 - y - r >= 0
            constraints.append(1 - y - r)
            
        return np.array(constraints)
    
    def objective(circles_flat):
        """Maximize sum of radii (minimize negative sum)"""
        circles_array = circles_flat.reshape(-1, 3)
        return -np.sum(circles_array[:, 2])
    
    # Prepare initial guess
    initial_guess = circles.flatten()
    
    # Set up constraints for scipy optimization
    # Note: scipy uses constraints as "fun >= 0", so we need to flip signs appropriately
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_bounds(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_distance(x)}
    ]
    
    # Bounds for variables: [x1, y1, r1, x2, y2, r2, ...]
    bounds = []
    for i in range(n):
        # x bounds: [r, 1-r]
        bounds.append((0.001, 0.999))  # Avoid exact boundaries
        # y bounds: [r, 1-r] 
        bounds.append((0.001, 0.999))
        # r bounds: [0.001, 0.5] (reasonable range)
        bounds.append((0.001, 0.499))
    
    try:
        # Perform optimization
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # If optimization fails, return the initial configuration
            return circles
    except Exception as e:
        # If optimization fails due to any error, return initial configuration
        return circles


# EVOLVE-BLOCK-END
