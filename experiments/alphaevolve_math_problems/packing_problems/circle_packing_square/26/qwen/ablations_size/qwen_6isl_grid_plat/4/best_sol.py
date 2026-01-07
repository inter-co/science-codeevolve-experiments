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
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    n = 26
    
    # Initialize using a hexagonal lattice pattern inspired by best practices
    def initialize_hexagonal():
        circles = []
        
        # Create a hexagonal grid pattern that fits 26 circles
        # Using 5 rows and 5 columns with slight adjustments
        rows = 5
        cols = 5
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Adjust spacing to allow for radius optimization
        base_radius = min(spacing_x, spacing_y) * 0.4
        
        # Place circles in a grid pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Add slight randomness to avoid perfect symmetry
                x += (np.random.random() - 0.5) * spacing_x * 0.2
                y += (np.random.random() - 0.5) * spacing_y * 0.2
                # Clamp to valid range
                x = max(base_radius, min(1 - base_radius, x))
                y = max(base_radius, min(1 - base_radius, y))
                circles.append([x, y, base_radius])
        
        # Fill remaining circles with random valid positions
        while len(circles) < n:
            x = np.random.random() * (1 - 2 * base_radius) + base_radius
            y = np.random.random() * (1 - 2 * base_radius) + base_radius
            circles.append([x, y, base_radius])
            
        return np.array(circles)
    
    # Initialize
    circles = initialize_hexagonal()
    
    # Define constraints and objective
    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
        # We want to maximize sum of radii, so we minimize negative sum
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i + 2]  # radius is at index 2, 5, 8, ...
        return -total_radius
    
    def constraint_containment(params):
        # Ensure all circles are within the unit square
        constraints = []
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            # Circle must be fully inside the square
            constraints.append(x - r)  # x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
        return np.array(constraints)
    
    def constraint_nonoverlap(params):
        # Ensure no two circles overlap
        constraints = []
        # Convert flat array to circle representation
        circle_list = []
        for i in range(n):
            circle_list.append([params[3*i], params[3*i+1], params[3*i+2]])
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circle_list[i]
                x2, y2, r2 = circle_list[j]
                # Distance between centers must be >= sum of radii
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                # We want dist_sq >= min_dist_sq, so we return dist_sq - min_dist_sq
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)
    
    # Flatten initial guess
    initial_guess = circles.flatten()
    
    # Set up bounds (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Define constraints for optimization
    cons = []
    # Containment constraints
    cons.append({'type': 'ineq', 'fun': lambda x: constraint_containment(x)})
    # Non-overlap constraints
    cons.append({'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)})
    
    # Perform optimization using SLSQP which works well for this type of problem
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-6}
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            # Ensure all circles are within bounds
            for i in range(n):
                x, y, r = final_circles[i]
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                final_circles[i] = [x, y, r]
            return final_circles
    except Exception as e:
        # If optimization fails, return initial configuration
        pass
    
    # If optimization fails, return initial configuration
    return circles


# EVOLVE-BLOCK-END
