# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial placement based on hexagonal packing followed by optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initial placement using a hexagonal lattice pattern
    def generate_hexagonal_initial():
        # Arrange circles in a roughly hexagonal pattern
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Adjust for hexagonal pattern
                if i % 2 == 1:
                    x += spacing_x / 2
                
                # Initial radius estimate (small enough to fit)
                r = min(x, 1-x, y, 1-y) / 2
                if r > 0:
                    circles.append([x, y, r])
        
        # Fill remaining circles with small radii in corners
        while len(circles) < n:
            # Place in corners with minimal overlap
            x = 0.1 if len(circles) % 2 == 0 else 0.9
            y = 0.1 if len(circles) % 3 == 0 else 0.9
            r = 0.01
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Generate initial configuration
    initial_circles = generate_hexagonal_initial()
    
    # Define constraint functions
    def get_constraints():
        cons = []
        # Boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})  # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})  # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})  # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})  # 1 - y - r >= 0
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i, j=j: 
                        np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) - x[3*i+2] - x[3*j+2]
                })
        return cons
    
    # Objective function to maximize (negative because we minimize)
    def objective(x):
        return -np.sum(x[2::3])  # Sum of all radii (negative for maximization)
    
    # Flatten initial configuration for optimization
    x0 = initial_circles.flatten()
    
    # Run optimization
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=get_constraints(),
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure all values are valid
            for i in range(n):
                # Clip radii to positive values
                optimized_circles[i, 2] = max(0, optimized_circles[i, 2])
                # Ensure positions are within bounds
                optimized_circles[i, 0] = np.clip(optimized_circles[i, 0], 
                                                 optimized_circles[i, 2], 1 - optimized_circles[i, 2])
                optimized_circles[i, 1] = np.clip(optimized_circles[i, 1], 
                                                 optimized_circles[i, 2], 1 - optimized_circles[i, 2])
            return optimized_circles
    except Exception as e:
        pass
    
    # If optimization fails, return initial configuration
    return initial_circles


# EVOLVE-BLOCK-END
