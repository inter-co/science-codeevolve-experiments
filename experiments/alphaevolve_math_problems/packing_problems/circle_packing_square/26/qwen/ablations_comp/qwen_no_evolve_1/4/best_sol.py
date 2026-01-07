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
    
    # Initialize with a hexagonal packing pattern as a starting configuration
    def initialize_hexagonal():
        circles = np.zeros((n, 3))
        
        # Try to arrange in a hexagonal pattern
        rows = int(math.sqrt(n)) + 1
        cols = int(n / rows) + 1
        
        # Adjust dimensions to fit in unit square
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Calculate initial positions and radii
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Alternate row offset for hexagonal packing
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                
                # Set initial radius based on proximity to edges
                min_dist_to_edge = min(x, 1-x, y, 1-y)
                radius = min_dist_to_edge / 2.0
                
                # Ensure we don't exceed bounds
                if x - radius < 0 or x + radius > 1 or y - radius < 0 or y + radius > 1:
                    continue
                    
                circles[idx] = [x, y, radius]
                idx += 1
                
            if idx >= n:
                break
                
        # Fill remaining slots with small radii if needed
        for i in range(idx, n):
            circles[i] = [0.5, 0.5, 0.01]
            
        return circles
    
    # Create initial configuration
    circles = initialize_hexagonal()
    
    # Define constraints for optimization
    def get_constraints():
        cons = []
        
        # Boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})  # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})  # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})  # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})  # 1 - y - r >= 0
            
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i, j=j: np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) - (x[3*i+2] + x[3*j+2])
                })
                
        return cons
    
    # Objective function to maximize sum of radii
    def objective(x):
        # We want to maximize sum of radii, so we minimize negative sum
        return -np.sum(x[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Flatten initial configuration for optimization
    x0 = circles.flatten()
    
    # Get constraints
    constraints = get_constraints()
    
    # Optimization bounds (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y in [0,1], r in [0, 0.5]
    
    # Run optimization
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            # Extract optimized solution
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial configuration if optimization fails
            return circles
    except Exception as e:
        # Fallback to initial configuration
        return circles


# EVOLVE-BLOCK-END
