# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial placement based on hexagonal packing, followed by optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initial strategy: arrange in a hexagonal pattern
    # For 32 circles, we can arrange in roughly a 5x7 grid with some adjustment
    rows = 5
    cols = 7
    
    # Create initial hexagonal arrangement
    circles = np.zeros((n, 3))
    
    # Hexagonal packing parameters
    # In a hexagonal lattice, horizontal spacing = 2*r, vertical spacing = sqrt(3)*r
    # We'll start with a small radius and adjust
    initial_radius = 0.05
    
    # Place circles in a hexagonal pattern
    row_offset = 0
    count = 0
    
    for i in range(rows):
        y = initial_radius + i * initial_radius * math.sqrt(3)
        if i % 2 == 0:
            row_offset = 0
        else:
            row_offset = initial_radius
            
        for j in range(cols):
            if count >= n:
                break
            x = initial_radius + j * 2 * initial_radius + row_offset
            # Ensure circles fit within unit square
            if x + initial_radius <= 1 and y + initial_radius <= 1:
                circles[count] = [x, y, initial_radius]
                count += 1
                
        if count >= n:
            break
    
    # Fill remaining circles with smaller radii if needed
    while count < n:
        circles[count] = [0.5, 0.5, initial_radius * 0.5]
        count += 1
    
    # Optimization: Use scipy minimize to maximize sum of radii
    # We'll optimize the positions and radii simultaneously
    
    # Flatten initial configuration for optimization
    initial_vars = []
    for i in range(n):
        initial_vars.extend([circles[i][0], circles[i][1], circles[i][2]])
    
    def objective(vars):
        # Calculate negative sum of radii (since we want to maximize)
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i + 2]
        return -total_radius
    
    def constraint_func(vars):
        # Check containment and non-overlap constraints
        constraints = []
        
        # Containment constraints: each circle must fit in the unit square
        for i in range(n):
            x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
            # Circle must be inside unit square
            constraints.append(x - r)  # x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        # Non-overlap constraints: distance between centers >= sum of radii
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                # Distance between centers >= sum of radii
                constraints.append(dist - r1 - r2)
                
        return np.array(constraints)
    
    # Define bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x coordinate: [r, 1-r]
        bounds.append((0.0001, 0.9999))  # x
        bounds.append((0.0001, 0.9999))  # y
        bounds.append((0.0001, 0.4999))  # r (max radius is 0.5)
    
    # Use SLSQP optimization method
    try:
        result = minimize(objective, initial_vars, method='SLSQP', bounds=bounds, 
                         constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_vars = result.x
            for i in range(n):
                circles[i] = [optimized_vars[3*i], optimized_vars[3*i+1], optimized_vars[3*i+2]]
        else:
            # If optimization fails, return initial configuration
            pass
    except Exception as e:
        # If optimization fails, return initial configuration
        pass
    
    # Final refinement: adjust positions to ensure valid constraints
    # This is a simplified version of a more complex refinement process
    for i in range(n):
        x, y, r = circles[i]
        # Ensure proper containment
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
