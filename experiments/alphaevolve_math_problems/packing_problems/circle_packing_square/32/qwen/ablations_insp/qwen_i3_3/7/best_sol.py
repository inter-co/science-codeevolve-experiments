# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial hexagonal lattice placement with scipy optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    n = 32
    
    # Initialize with hexagonal lattice pattern (similar to inspiration 1 but more robust)
    def initialize_hexagonal_placement():
        # Create a hexagonal grid that fits in unit square
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        
        # Hexagonal packing parameters
        hex_radius = 0.1  # Initial guess
        spacing_x = hex_radius * 2
        spacing_y = hex_radius * math.sqrt(3)
        
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = spacing_x * j + hex_radius
                y = spacing_y * i + hex_radius
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                circles.append([x, y, hex_radius])
        
        # Adjust to fit in unit square
        if circles:
            max_x = max(c[0] + c[2] for c in circles)
            max_y = max(c[1] + c[2] for c in circles)
            
            scale = min(1/max_x, 1/max_y) if max_x > 1 or max_y > 1 else 1
            
            for i in range(len(circles)):
                circles[i][0] *= scale
                circles[i][1] *= scale
                circles[i][2] *= scale
                
        # Fill remaining slots with small circles if needed
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.01])
            
        return np.array(circles[:n])
    
    # Create initial configuration
    initial_circles = initialize_hexagonal_placement()
    
    # Prepare for optimization - flatten for scipy
    initial_flat = initial_circles.flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r
    
    # Objective function to maximize sum of radii (minimize negative)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Maximize sum of radii
    
    # Constraint functions
    def get_constraints():
        constraints = []
        
        # Containment constraints for each circle
        for i in range(n):
            def contain_constraint(c, i=i):
                x, y, r = c[3*i], c[3*i+1], c[3*i+2]
                return min(x - r, y - r, 1 - x - r, 1 - y - r)
            
            constraints.append({'type': 'ineq', 'fun': contain_constraint})
        
        # Non-overlap constraints for each pair
        for i in range(n):
            for j in range(i+1, n):
                def overlap_constraint(c, i=i, j=j):
                    x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                    x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist_sq = (r1 + r2)**2
                    return dist_sq - min_dist_sq
                    
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
                
        return constraints
    
    # Get constraints once
    cons = get_constraints()
    
    # Run optimization using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure final constraints are satisfied by clamping
            for i in range(len(optimized_circles)):
                x, y, r = optimized_circles[i]
                # Clamp to valid ranges
                optimized_circles[i] = [
                    max(r, min(1-r, x)),
                    max(r, min(1-r, y)), 
                    max(0.001, min(0.499, r))
                ]
            return optimized_circles
    except Exception as e:
        # If optimization fails, return initial placement
        pass
    
    # Return initial placement if optimization failed
    return initial_circles


# EVOLVE-BLOCK-END
