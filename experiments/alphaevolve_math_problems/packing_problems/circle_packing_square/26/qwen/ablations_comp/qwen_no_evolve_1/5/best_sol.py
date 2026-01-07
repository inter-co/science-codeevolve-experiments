# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a geometric pattern
    def initialize_circles():
        # Start with a hexagonal packing pattern as initial guess
        circles = []
        
        # Place circles in a grid-like pattern but slightly randomized
        rows = 5
        cols = 5
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Create a more structured initial placement
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Add small random perturbation to avoid perfect symmetry
                x += np.random.uniform(-0.01, 0.01)
                y += np.random.uniform(-0.01, 0.01)
                # Ensure we're within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                circles.append([x, y, 0.05])
        
        # Fill remaining positions with random placements
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        return np.array(circles)
    
    # Define constraint functions
    def get_constraints(circles):
        """Generate constraint functions for optimization"""
        constraints = []
        
        # Boundary constraints: each circle must fit within the unit square
        def boundary_constraint(i):
            def constraint(x):
                x_pos, y_pos, r = x[3*i], x[3*i+1], x[3*i+2]
                # Circle must stay within bounds
                return min(x_pos - r, 1 - x_pos - r, y_pos - r, 1 - y_pos - r)
            return constraint
        
        # Non-overlap constraints
        def overlap_constraint(i, j):
            def constraint(x):
                x1, y1, r1 = x[3*i], x[3*i+1], x[3*i+2]
                x2, y2, r2 = x[3*j], x[3*j+1], x[3*j+2]
                # Distance between centers minus radii must be non-negative
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                return math.sqrt(dist_sq) - (r1 + r2)
            return constraint
            
        # Add boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Add non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
                
        return constraints
    
    # Objective function to maximize sum of radii
    def objective(x):
        # Return negative because we want to maximize (scipy minimizes)
        return -sum(x[3*i+2] for i in range(n))
    
    # Initialize
    circles = initialize_circles()
    
    # Flatten initial guess for scipy
    x0 = circles.flatten()
    
    # Set bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x bounds: [r, 1-r]
        bounds.append((0.001, 0.999))  # x coordinate
        bounds.append((0.001, 0.999))  # y coordinate
        bounds.append((0.001, 0.499))  # radius (max radius is 0.5)
    
    # Get constraints
    constraints = get_constraints(circles)
    
    # Perform optimization
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape((n, 3))
            return optimized_circles
        else:
            # Fallback to initial configuration if optimization fails
            return circles
    except Exception:
        # Fallback to initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
