# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal packing initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using hexagonal packing pattern for better density
    def initialize_hexagonal():
        circles = np.zeros((n, 3))
        
        # Create hexagonal packing pattern
        sqrt3 = math.sqrt(3)
        # Estimate initial radius based on hexagonal packing efficiency
        radius_estimate = 0.08
        
        # Determine grid dimensions for hexagonal packing
        rows = int(math.ceil(math.sqrt(n) * 1.2))
        cols = int(math.ceil(n / rows))
        
        if rows * cols < n:
            rows += 1
            
        # Calculate spacing based on radius
        dx = 2 * radius_estimate
        dy = sqrt3 * radius_estimate
        
        placed = 0
        for row in range(rows):
            y = radius_estimate + row * dy
            if y >= 1 - radius_estimate:
                break
                
            # Offset every other row for hexagonal packing
            x_offset = 0 if row % 2 == 0 else dx / 2
            col = 0
            
            while col < cols and placed < n:
                x = radius_estimate + x_offset + col * dx
                if x >= 1 - radius_estimate:
                    break
                    
                # Add small random perturbation to avoid perfect patterns
                x += random.uniform(-dx*0.05, dx*0.05)
                y += random.uniform(-dy*0.05, dy*0.05)
                
                # Clip to ensure circles stay within bounds
                x = max(radius_estimate, min(1 - radius_estimate, x))
                y = max(radius_estimate, min(1 - radius_estimate, y))
                
                circles[placed] = [x, y, radius_estimate]
                placed += 1
                col += 1
                
        # Fill remaining positions with small circles if needed
        for i in range(placed, n):
            circles[i] = [0.5, 0.5, 0.01]
            
        return circles
    
    # Create initial configuration
    circles = initialize_hexagonal()
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(x):
        return -np.sum(x[2::3])  # Negative because we minimize
    
    # Improved constraint handling using scipy's built-in approach
    def create_constraints():
        """Create constraints for optimization"""
        constraints = []
        
        # Boundary constraints for each circle
        def boundary_constraint(i):
            def cons(x):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                # All four boundary constraints: xi >= ri, yi >= ri, 1-xi >= ri, 1-yi >= ri
                return np.array([
                    xi - ri,           # xi >= ri
                    yi - ri,           # yi >= ri
                    1 - xi - ri,       # 1-xi >= ri
                    1 - yi - ri        # 1-yi >= ri
                ])
            return cons
        
        # Overlap constraints for each pair of circles
        def overlap_constraint(i, j):
            def cons(x):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                distance = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                # Distance between centers >= sum of radii (negative when violated)
                return distance - ri - rj
            return cons
        
        # Add boundary constraints
        for i in range(n):
            constraints.append({
                'type': 'ineq', 
                'fun': boundary_constraint(i)
            })
        
        # Add overlap constraints - only check nearby circles for efficiency
        # Use spatial indexing to reduce constraint checking
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({
                    'type': 'ineq', 
                    'fun': overlap_constraint(i, j)
                })
        
        return constraints
    
    # Set bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Run optimization with better parameters and multiple attempts
    try:
        # Flatten initial guess
        x0 = circles.flatten()
        
        # Create constraints
        constraints = create_constraints()
        
        # Try optimization with different methods and parameters
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-8, 'disp': False}
        )
        
        if result.success:
            # Return optimized circles
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial configuration if optimization fails
            return circles
            
    except Exception as e:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
