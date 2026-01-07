# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial geometric placement and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Initial heuristic placement using hexagonal grid
    def initial_placement():
        # Try to arrange circles in a hexagonal pattern
        # For 32 circles, we can try a 6x6 grid with some adjustments
        circles = []
        
        # Create a hexagonal grid pattern
        rows = 6
        cols = 6
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Hexagonal offset for alternate rows
        hex_offset = spacing_x * 0.5
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += hex_offset
                circles.append([x, y, min(x, 1-x, y, 1-y) * 0.4])
        
        # Fill remaining positions
        while len(circles) < n:
            # Add random positions with small radii
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = min(x, 1-x, y, 1-y) * 0.3
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Generate initial configuration
    circles = initial_placement()
    
    # Define constraint functions
    def radius_constraint(i):
        """Ensure circle i stays within bounds"""
        def constraint(x):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            return min(xi - ri, 1 - xi - ri, yi - ri, 1 - yi - ri)
        return constraint
    
    def overlap_constraint(i, j):
        """Ensure circles i and j don't overlap"""
        def constraint(x):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            return dist - ri - rj
        return constraint
    
    # Flatten initial configuration for optimization
    initial_flat = circles.flatten()
    
    # Set up constraints
    constraints = []
    
    # Boundary constraints for each circle
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': radius_constraint(i)})
    
    # Overlap constraints for each pair of circles
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]  # radius is at index 3*i+2
        return -total_radius
    
    # Bounds for variables (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Optimize
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # If optimization fails, return initial placement
            return circles
    except Exception:
        # If anything goes wrong, return initial placement
        return circles


# EVOLVE-BLOCK-END
