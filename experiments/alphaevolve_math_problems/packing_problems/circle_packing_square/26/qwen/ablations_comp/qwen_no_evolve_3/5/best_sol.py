# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach: initial hexagonal packing + constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores 
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initial heuristic placement using hexagonal packing pattern
    def generate_hexagonal_initial():
        # Try to place circles in a hexagonal pattern
        circles = []
        # Hexagonal packing parameters
        sqrt3 = math.sqrt(3)
        # Estimate initial radius based on hexagonal packing density
        # For 26 circles in unit square, approximate radius ~ 0.08-0.12
        r_estimate = 0.1
        rows = int(math.sqrt(n))
        cols = (n + rows - 1) // rows
        
        # Create hexagonal grid
        y_spacing = r_estimate * 2
        x_spacing = r_estimate * sqrt3
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = r_estimate + j * x_spacing
                y = r_estimate + i * y_spacing
                # Offset every other row
                if i % 2 == 1:
                    x += x_spacing / 2
                # Check bounds
                if x <= 1 - r_estimate and y <= 1 - r_estimate:
                    circles.append([x, y, r_estimate])
        
        # Fill remaining spots with random valid placements
        while len(circles) < n:
            x = np.random.uniform(r_estimate, 1 - r_estimate)
            y = np.random.uniform(r_estimate, 1 - r_estimate)
            circles.append([x, y, r_estimate])
            
        return np.array(circles)
    
    # Generate initial configuration
    initial_circles = generate_hexagonal_initial()
    
    # Define constraint functions
    def get_constraints(circles):
        """Generate constraint functions for optimization"""
        constraints = []
        
        # Boundary constraints: each circle must fit entirely in unit square
        def boundary_constraint(i):
            def c(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                return min(x_i - r_i, 1 - x_i - r_i, y_i - r_i, 1 - y_i - r_i)
            return c
        
        # Non-overlap constraints
        def overlap_constraint(i, j):
            def c(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                return dist_sq - (r_i + r_j)**2
            return c
        
        # Add boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Add non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
                
        return constraints
    
    # Flatten initial circles for optimization
    x0 = initial_circles.flatten()
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(x):
        radii = x[2::3]  # Extract all radii
        return -np.sum(radii)
    
    # Get constraints
    constraints = get_constraints(initial_circles)
    
    # Bounds for variables: x, y in [r, 1-r], r > 0
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])
    
    try:
        # Run optimization with SLSQP method
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=lambda x: None  # No callback needed
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure final validation
            return validate_and_refine(optimized_circles)
        else:
            # Fall back to initial configuration if optimization fails
            return initial_circles
            
    except Exception as e:
        # Fallback to initial configuration on any error
        return initial_circles

def validate_and_refine(circles):
    """Validate configuration and perform final refinement"""
    # Ensure all circles are within bounds
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Clamp positions to valid range
        circles[i][0] = max(r, min(1-r, x))
        circles[i][1] = max(r, min(1-r, y))
    
    return circles


# EVOLVE-BLOCK-END
