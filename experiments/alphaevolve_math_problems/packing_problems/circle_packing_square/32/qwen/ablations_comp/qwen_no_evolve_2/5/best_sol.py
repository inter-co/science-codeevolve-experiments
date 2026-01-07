# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with hexagonal packing pattern
    def initialize_hexagonal_layout():
        # Try to arrange in a hexagonal pattern that fits well in unit square
        # For 32 circles, we can try 6 rows with varying column counts
        rows = 6
        cols_per_row = [5, 6, 5, 6, 5, 6]  # alternating pattern
        
        circles = []
        radius = 0.05  # Initial guess
        
        # Hexagonal packing parameters
        horizontal_spacing = radius * 2
        vertical_spacing = radius * math.sqrt(3)
        
        y_offset = 0.0
        for i, cols in enumerate(cols_per_row):
            # Offset every other row for hexagonal pattern
            x_offset = (i % 2) * (horizontal_spacing / 2)
            
            for j in range(cols):
                x = x_offset + j * horizontal_spacing + radius
                y = y_offset + i * vertical_spacing + radius
                
                # Ensure circles stay within bounds
                if x - radius >= 0 and x + radius <= 1 and y - radius >= 0 and y + radius <= 1:
                    circles.append([x, y, radius])
        
        # If we don't have enough circles, fill remaining spots with smaller radii
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.01])  # Place at center with small radius
            
        return np.array(circles[:n])
    
    # Create initial configuration
    circles = initialize_hexagonal_layout()
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must be within unit square
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3 + 2]})  # radius >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i*3 + 2] - x[i*3]})  # x + r <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i*3 + 2] - x[i*3 + 1]})  # y + r <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3 + 2]})  # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3 + 1] - x[i*3 + 2]})  # y - r >= 0
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i, j=j: (
                        np.sqrt((x[i*3] - x[j*3])**2 + (x[i*3+1] - x[j*3+1])**2) 
                        - x[i*3+2] - x[j*3+2]
                    )
                })
        
        return cons
    
    # Objective function to maximize sum of radii
    def objective(x):
        # Return negative because we want to maximize sum of radii
        return -np.sum(x[2::3])  # Sum of all radii (indices 2, 5, 8, ...)
    
    # Flatten initial circles array for optimization
    x0 = circles.flatten()
    
    # Get constraints
    constraints = get_constraints()
    
    # Perform optimization using SLSQP method
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6},
            bounds=[(0, 1) if i % 3 in [0, 1] else (1e-6, 0.5) for i in range(3*n)]
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial configuration if optimization fails
            return circles
    except Exception:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
