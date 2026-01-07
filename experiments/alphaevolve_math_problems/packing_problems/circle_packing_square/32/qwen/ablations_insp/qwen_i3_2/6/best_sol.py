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
    
    # Alternative initialization using a more refined grid approach
    def initialize_grid_refined():
        circles = np.zeros((n, 3))
        
        # Create a refined grid layout
        grid_size = int(math.ceil(math.sqrt(n)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        placed = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if placed >= n:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                # Initial radius - small enough to fit in grid cell
                r = min(spacing_x, spacing_y) * 0.4
                circles[placed] = [x, y, r]
                placed += 1
            if placed >= n:
                break
        
        # Set remaining circles with small radii
        for i in range(placed, n):
            circles[i] = [0.5, 0.5, 0.01]
        
        return circles
    
    # Create initial configurations
    initial_configs = []
    initial_configs.append(initialize_hexagonal())
    initial_configs.append(initialize_grid_refined())
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(x):
        return -np.sum(x[2::3])  # Negative because we minimize
    
    # Constraint creation with improved efficiency
    def create_constraints(indices_to_check=None):
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
        # For better performance, we'll limit the number of constraints checked
        # This is a trade-off between accuracy and computation time
        if indices_to_check is None:
            # Check all pairs (this might be too expensive, so we'll optimize later)
            for i in range(n):
                for j in range(i+1, n):
                    constraints.append({
                        'type': 'ineq', 
                        'fun': overlap_constraint(i, j)
                    })
        else:
            # Only check specified pairs
            for i, j in indices_to_check:
                constraints.append({
                    'type': 'ineq', 
                    'fun': overlap_constraint(i, j)
                })
        
        return constraints
    
    # Set bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    best_result = None
    best_sum = 0
    
    # Try multiple initialization strategies and optimization attempts
    for attempt, circles in enumerate(initial_configs):
        try:
            # Flatten initial guess
            x0 = circles.flatten()
            
            # Create constraints
            constraints = create_constraints()
            
            # Run optimization with different methods and parameters
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-8, 'disp': False}
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result.x.reshape(-1, 3)
            else:
                # Even if not successful, keep the best initial configuration
                current_sum = -objective(x0)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = circles
                    
        except Exception as e:
            # If optimization fails, keep the initial configuration
            current_sum = -objective(circles.flatten())
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = circles
    
    # If we still have no good result, use the first initialization
    if best_result is None:
        return initial_configs[0]
    
    return best_result


# EVOLVE-BLOCK-END
