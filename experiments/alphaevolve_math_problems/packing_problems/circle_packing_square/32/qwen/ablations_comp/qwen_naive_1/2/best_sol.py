# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach combining initial hexagonal placement with constrained optimization.
    """
    n = 32
    
    # Initialize with hexagonal packing pattern
    def initialize_hexagonal_placement():
        # Try to place circles in a hexagonal pattern
        circles = []
        
        # Create a grid that roughly fits 32 circles
        rows = int(math.sqrt(n))
        cols = int(math.ceil(n / rows))
        
        # Adjust for better packing
        if rows * cols < n:
            rows += 1
            
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Hexagonal packing with offset rows
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                    
                # Ensure we're within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append([x, y, min(x, 1-x, y, 1-y)])
                    
        # Fill remaining spots
        while len(circles) < n:
            # Add random placements near edges for diversity
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = min(x, 1-x, y, 1-y)
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Initialize
    circles = initialize_hexagonal_placement()
    
    # Define constraint functions
    def get_constraints(circles):
        """Get all constraint functions for optimization"""
        constraints = []
        
        # Boundary constraints: each circle must fit entirely in unit square
        def boundary_constraint(i):
            def constraint(x):
                x_c, y_c, r = x[3*i], x[3*i+1], x[3*i+2]
                return min(r, x_c - r, 1 - x_c - r, y_c - r, 1 - y_c - r)
            return constraint
        
        # Non-overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(i, j):
            def constraint(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                return dist - (r_i + r_j)
            return constraint
        
        # Add boundary constraints for all circles
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': boundary_constraint(i)})
            
        # Add non-overlap constraints for all pairs
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
                
        return constraints
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]  # radius is third component
        return -total_radius
    
    # Flatten initial circles array for optimization
    x0 = circles.flatten()
    
    # Get constraints
    constraints = get_constraints(circles)
    
    # Bounds for variables: x,y in [0,1], r > 0
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (1e-6, 0.5)])  # r bounded to prevent overflow
    
    try:
        # Optimize using SLSQP method which handles constraints well
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6},
            callback=None
        )
        
        if result.success:
            # Extract optimized circles
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial placement if optimization fails
            return circles
            
    except Exception as e:
        # Fallback to initial placement if optimization fails
        return circles


# EVOLVE-BLOCK-END
