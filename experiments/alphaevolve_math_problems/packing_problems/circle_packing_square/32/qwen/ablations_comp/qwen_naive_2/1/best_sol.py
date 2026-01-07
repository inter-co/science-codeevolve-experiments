# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import time

# Global constants
N_CIRCLES = 32
BOUNDARY_MARGIN = 1e-6  # Small margin to ensure containment

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a physics-inspired constrained optimization approach.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    
    # Initialize with a good starting configuration using hexagonal packing
    def generate_initial_configuration():
        # Create a hexagonal grid pattern for initial placement
        circles = []
        
        # Parameters for hexagonal packing
        rows = 6
        cols = 6
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Generate points in a hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= N_CIRCLES:
                    break
                    
                # Offset every other row for hexagonal packing
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + 0.5 + offset) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Add small random noise
                x += random.uniform(-0.01, 0.01)
                y += random.uniform(-0.01, 0.01)
                
                # Keep within bounds
                x = max(BOUNDARY_MARGIN, min(1 - BOUNDARY_MARGIN, x))
                y = max(BOUNDARY_MARGIN, min(1 - BOUNDARY_MARGIN, y))
                
                # Initial radius estimate based on available space
                radius = min(x, 1-x, y, 1-y) * 0.3
                
                circles.append([x, y, radius])
                
        # Fill remaining circles with random placement near edges
        while len(circles) < N_CIRCLES:
            x = random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            y = random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            radius = min(x, 1-x, y, 1-y) * 0.3
            circles.append([x, y, radius])
            
        return np.array(circles)
    
    # Define constraints for optimization
    def get_constraints():
        """Return constraints for the optimization problem"""
        constraints = []
        
        # Boundary constraints: each circle must be within the unit square
        def boundary_constraint(params):
            # params: [x1, y1, r1, x2, y2, r2, ...]
            values = []
            for i in range(N_CIRCLES):
                x = params[3*i]
                y = params[3*i + 1]
                r = params[3*i + 2]
                # Ensure circle is within bounds
                values.extend([
                    x - r,           # x - r >= 0
                    1 - x - r,       # 1 - x - r >= 0
                    y - r,           # y - r >= 0
                    1 - y - r        # 1 - y - r >= 0
                ])
            return np.array(values)
        
        # Overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(params):
            values = []
            for i in range(N_CIRCLES):
                for j in range(i+1, N_CIRCLES):
                    x1, y1, r1 = params[3*i], params[3*i + 1], params[3*i + 2]
                    x2, y2, r2 = params[3*j], params[3*j + 1], params[3*j + 2]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    # dist >= r1 + r2 (so we want dist - r1 - r2 >= 0)
                    values.append(dist - r1 - r2)
            return np.array(values)
        
        # Add boundary constraints
        constraints.append({'type': 'ineq', 'fun': boundary_constraint})
        # Add overlap constraints
        constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return constraints
    
    # Objective function to maximize (negative for minimization)
    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ...]
        total_radius = 0
        for i in range(N_CIRCLES):
            total_radius += params[3*i + 2]  # radius is at index 3*i + 2
        return -total_radius  # Negative because we want to maximize
    
    # Run optimization
    # Start with good initial configuration
    initial_config = generate_initial_configuration()
    
    # Flatten initial configuration for optimization
    initial_params = initial_config.flatten()
    
    # Set bounds for parameters (x, y, r) for each circle
    bounds = []
    for i in range(N_CIRCLES):
        # x bounds
        bounds.append((BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN))
        # y bounds
        bounds.append((BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN))
        # r bounds (positive, but limited by space)
        bounds.append((0.001, 0.5))
    
    # Get constraints
    constraints = get_constraints()
    
    # Optimization settings
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
    
    # Run optimization
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=1e-6
        )
        
        # Extract final solution
        final_params = result.x
        circles = np.zeros((N_CIRCLES, 3))
        for i in range(N_CIRCLES):
            circles[i] = [
                final_params[3*i],      # x
                final_params[3*i + 1],  # y
                final_params[3*i + 2]   # r
            ]
            
        # Final refinement step
        circles = refine_solution(circles)
        
        return circles
        
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        print(f"Optimization failed: {e}")
        return refine_solution(initial_config)

def refine_solution(circles: np.ndarray) -> np.ndarray:
    """
    Apply local refinement to improve solution quality
    """
    # Simple local optimization: try to increase radii while maintaining constraints
    for _ in range(100):  # Multiple refinement iterations
        improved = False
        for i in range(len(circles)):
            # Try to increase radius of circle i
            old_radius = circles[i, 2]
            max_radius = min(
                circles[i, 0], 1 - circles[i, 0],
                circles[i, 1], 1 - circles[i, 1]
            )
            
            # Check overlap with other circles
            valid_radius = max_radius
            for j in range(len(circles)):
                if i != j:
                    dist = np.sqrt(
                        (circles[i, 0] - circles[j, 0])**2 + 
                        (circles[i, 1] - circles[j, 1])**2
                    )
                    min_dist = circles[i, 2] + circles[j, 2]
                    if dist < min_dist:
                        # Reduce radius to avoid overlap
                        valid_radius = min(valid_radius, dist - circles[j, 2] - 0.0001)
            
            # Update radius if beneficial
            if valid_radius > old_radius and valid_radius > 0.001:
                circles[i, 2] = min(valid_radius, max_radius)
                improved = True
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
