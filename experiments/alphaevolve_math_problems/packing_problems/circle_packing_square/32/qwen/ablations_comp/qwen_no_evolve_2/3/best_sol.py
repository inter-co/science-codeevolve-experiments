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
    
    # Initialize using a hexagonal grid pattern for good initial placement
    def initialize_hexagonal_layout():
        # Create a hexagonal grid pattern that fits within the unit square
        circles = []
        
        # Hexagonal packing parameters
        sqrt3 = math.sqrt(3)
        # Try to fit circles in a hexagonal pattern
        rows = int(math.sqrt(n) * 1.2)
        cols = int(n / rows) + 1
        
        # Adjust grid spacing based on number of circles needed
        spacing_x = 1.0 / max(cols, 1)
        spacing_y = spacing_x * sqrt3 / 2
        
        # Generate positions
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Offset odd rows
                if i % 2 == 1:
                    x += spacing_x / 2
                    
                # Ensure within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius guess - small but valid
                    r = min(x, 1-x, y, 1-y) * 0.4
                    if r > 0:
                        circles.append([x, y, r])
                        count += 1
                        
        # Fill remaining positions with random valid placements
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = min(x, 1-x, y, 1-y) * 0.3
            if r > 0:
                circles.append([x, y, r])
                
        return np.array(circles)
    
    # Initialize
    circles = initialize_hexagonal_layout()
    
    # Define constraint functions
    def get_constraints():
        """Generate constraint functions for the optimization"""
        cons = []
        
        # Boundary constraints: each circle must be fully contained
        def boundary_constraint(i):
            def constraint(params):
                x, y, r = params[i*3], params[i*3+1], params[i*3+2]
                # Circle must be within unit square
                return min(x - r, 1 - x - r, y - r, 1 - y - r)
            return constraint
            
        # Non-overlap constraints
        def overlap_constraint(i, j):
            def constraint(params):
                x1, y1, r1 = params[i*3], params[i*3+1], params[i*3+2]
                x2, y2, r2 = params[j*3], params[j*3+1], params[j*3+2]
                # Distance between centers minus sum of radii
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                return dist - (r1 + r2)
            return constraint
        
        # Add boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
            
        # Add non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
                
        return cons
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        total_radius = 0
        for i in range(n):
            total_radius += params[i*3+2]  # radius is third component
        return -total_radius
    
    # Bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x, y: [0.001, 0.999] to prevent boundary issues
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.499))  # r (maximum possible radius is 0.5)
    
    # Get constraints
    constraints = get_constraints()
    
    # Flatten initial configuration
    initial_params = circles.flatten()
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6},
            tol=1e-6
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure final positions are valid
            for i in range(n):
                x, y, r = optimized_circles[i]
                # Clip values to valid ranges
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                r = np.clip(r, 0.001, min(x, 1-x, y, 1-y))
                optimized_circles[i] = [x, y, r]
            return optimized_circles
        else:
            # Return the initial configuration if optimization fails
            return circles
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
