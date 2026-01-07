# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial heuristic placement and constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Stage 1: Initial placement using hexagonal packing pattern
    # Arrange circles in a roughly hexagonal grid pattern
    circles = np.zeros((n, 3))
    
    # Try to place circles in a hexagonal pattern
    rows = 5
    cols = 6
    if n <= rows * cols:
        # Create a hexagonal lattice pattern
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        sqrt3 = math.sqrt(3)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for hexagonal packing
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + offset) * spacing_x
                y = i * spacing_y
                # Ensure circles fit within bounds
                max_radius = min(x, 1-x, y, 1-y)
                if max_radius > 0:
                    # Start with small radius and adjust later
                    circles[idx] = [x, y, max_radius * 0.4]
                    idx += 1
                if idx >= n:
                    break
    
    # Fill remaining positions with random placements that respect boundaries
    for i in range(idx, n):
        while True:
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            # Find maximum possible radius at this location
            max_radius = min(x, 1-x, y, 1-y)
            if max_radius > 0:
                circles[i] = [x, y, max_radius * 0.3]
                break
    
    # Stage 2: Optimization using scipy minimize with constraints
    # Define objective function to maximize sum of radii
    def objective(params):
        # params contains [x1,y1,r1,x2,y2,r2,...,x26,y26,r26]
        total_radius = 0
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            total_radius += r
        return -total_radius  # negative because we want to maximize
    
    # Define constraint functions
    def constraint_containment(i, params):
        x, y, r = params[3*i], params[3*i+1], params[3*i+2]
        return min(r, x-r, 1-x-r, y-r, 1-y-r)  # Return positive if valid
    
    def constraint_nonoverlap(i, j, params):
        x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
        x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        return distance - (r1 + r2)  # Return positive if non-overlapping
    
    # Build constraints list
    constraints = []
    
    # Add containment constraints
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda params, i=i: constraint_containment(i, params)
        })
    
    # Add non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, i=i, j=j: constraint_nonoverlap(i, j, params)
            })
    
    # Bounds for variables [x1,y1,r1,x2,y2,r2,...,x26,y26,r26]
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Flatten initial guess
    initial_guess = []
    for i in range(n):
        initial_guess.extend([circles[i][0], circles[i][1], circles[i][2]])
    
    # Perform optimization
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            # Extract optimized values
            for i in range(n):
                circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        pass
    
    # Final refinement: simple local search to improve quality
    for _ in range(100):
        improved = False
        for i in range(n):
            best_radius = circles[i][2]
            best_x, best_y = circles[i][0], circles[i][1]
            
            # Try small perturbations
            for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                    for dr in [-0.005, -0.002, 0, 0.002, 0.005]:
                        new_x = circles[i][0] + dx
                        new_y = circles[i][1] + dy
                        new_r = circles[i][2] + dr
                        
                        # Check if new values are valid
                        if (new_x >= new_r and new_x <= 1-new_r and 
                            new_y >= new_r and new_y <= 1-new_r and 
                            new_r > 0.001):
                            
                            # Check overlap with all other circles
                            valid = True
                            for j in range(n):
                                if i != j:
                                    dist = math.sqrt((new_x - circles[j][0])**2 + 
                                                    (new_y - circles[j][1])**2)
                                    if dist < (new_r + circles[j][2]):
                                        valid = False
                                        break
                            
                            if valid and new_r > best_radius:
                                best_radius = new_r
                                best_x, best_y = new_x, new_y
                                improved = True
            
            circles[i] = [best_x, best_y, best_radius]
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
