# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a clean hexagonal grid initialization followed by SLSQP optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a clean hexagonal lattice pattern
    def initialize_hexagonal():
        # Create a hexagonal lattice pattern that fits in the unit square
        # Using the approach from INSPIRATION 1 but with slight improvements
        rows = 6
        cols = 6
        circles = []
        
        # Hexagonal spacing - more uniform than simple grid
        spacing_x = 1.0 / (cols + 1)
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        # Adjust spacing so we fit in unit square
        max_radius = min(spacing_x, spacing_y) / 2
        
        # Create hexagonal pattern with offset rows
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for hexagonal packing
                x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                # Ensure within bounds
                if x > max_radius and x < 1 - max_radius and y > max_radius and y < 1 - max_radius:
                    circles.append([x, y, max_radius])
        
        # Fill remaining positions with random valid circles
        attempts = 0
        while len(circles) < n and attempts < 1000:
            attempts += 1
            x = np.random.uniform(max_radius, 1 - max_radius)
            y = np.random.uniform(max_radius, 1 - max_radius)
            # Check if this position is valid (not too close to existing circles)
            valid = True
            for cx, cy, r in circles:
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < 2 * r:  # Too close to another circle
                    valid = False
                    break
            if valid:
                circles.append([x, y, max_radius])
        
        # If we still don't have enough circles, add more randomly
        while len(circles) < n:
            x = np.random.uniform(max_radius, 1 - max_radius)
            y = np.random.uniform(max_radius, 1 - max_radius)
            circles.append([x, y, max_radius])
            
        return np.array(circles[:n])
    
    # Generate initial configuration
    circles = initialize_hexagonal()
    
    # Define objective function to maximize sum of radii
    def objective(params):
        # params contains [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
        # Extract radii and return negative sum (scipy minimizes)
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i + 2]
        return -total_radius
    
    # Define constraints efficiently
    def get_constraints():
        constraints = []
        
        # Boundary constraints: each circle must fit in the unit square
        def boundary_constraint(i):
            def constraint(params):
                x = params[3*i]
                y = params[3*i + 1]
                r = params[3*i + 2]
                # Circle must be fully inside the unit square
                return min(x - r, 1 - x - r, y - r, 1 - y - r)
            return constraint
        
        # Non-overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(i, j):
            def constraint(params):
                x1 = params[3*i]
                y1 = params[3*i + 1]
                r1 = params[3*i + 2]
                x2 = params[3*j]
                y2 = params[3*j + 1]
                r2 = params[3*j + 2]
                # Distance between centers minus sum of radii should be >= 0
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                return math.sqrt(dist_sq) - (r1 + r2)
            return constraint
        
        # Add boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Add non-overlap constraints - check all pairs for better solution quality
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        return constraints
    
    # Bounds for parameters: [x, y, r] for each circle
    bounds = []
    for i in range(n):
        # x, y: [0.001, 0.999] - keeping away from boundaries to avoid numerical issues
        # r: [0.001, 0.499] - reasonable range to prevent numerical issues
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Initial parameter vector
    initial_params = []
    for x, y, r in circles:
        initial_params.extend([x, y, r])
    
    # Get constraints
    constraints = get_constraints()
    
    # Optimize using SLSQP method which handles constraints well
    # Increase iterations and tighten tolerances for better convergence
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1500, 'ftol': 1e-7, 'gtol': 1e-7}
        )
        
        if result.success:
            # Extract final configuration
            final_circles = []
            for i in range(n):
                x = result.x[3*i]
                y = result.x[3*i + 1]
                r = result.x[3*i + 2]
                final_circles.append([x, y, r])
            return np.array(final_circles)
        else:
            # If optimization fails, return the initial configuration
            return circles
    except Exception as e:
        # If optimization fails due to error, return the initial configuration
        return circles


# EVOLVE-BLOCK-END
