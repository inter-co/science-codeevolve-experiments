# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial hexagonal packing followed by constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initial configuration: hexagonal packing pattern
    def create_hexagonal_pattern():
        # Arrange circles in a hexagonal grid pattern
        rows = 6
        cols = 6
        circles = []
        
        # Hexagonal packing parameters
        radius_guess = 0.08  # Initial guess for radius
        spacing = 2 * radius_guess  # Center-to-center distance
        
        # Create hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing + (i % 2) * spacing/2
                y = 0.1 + i * spacing * math.sqrt(3)/2
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, radius_guess])
        
        # If we don't have enough circles, fill with remaining ones
        while len(circles) < n:
            circles.append([0.5, 0.5, radius_guess])
            
        return np.array(circles[:n])
    
    # Constraint functions
    def constraint_containment(circles_flat):
        """Ensure all circles are fully contained in the unit square"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Circle must be within bounds
            constraints.append(x - r)  # x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        return np.array(constraints)
    
    def constraint_nonoverlap(circles_flat):
        """Ensure no two circles overlap"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Distance between centers must be >= sum of radii
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # Constraint: distance^2 - (r1+r2)^2 >= 0
                constraints.append(dist_sq - min_dist_sq)
                
        return np.array(constraints)
    
    # Objective function (negative because we want to maximize)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Create initial configuration
    initial_circles = create_hexagonal_pattern()
    initial_flat = initial_circles.flatten()
    
    # Set up constraints
    cons = []
    
    # Add containment constraints
    cons.append({'type': 'ineq', 'fun': lambda x: constraint_containment(x)})
    
    # Add non-overlap constraints
    cons.append({'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)})
    
    # Optimization bounds (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0.001, 0.999))
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds
        bounds.append((0.001, 0.499))
    
    # Run optimization
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            return final_circles
        else:
            # Return initial configuration if optimization fails
            return initial_circles
    except Exception:
        # Return initial configuration if optimization fails
        return initial_circles


# EVOLVE-BLOCK-END
