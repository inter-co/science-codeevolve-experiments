# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a mathematical programming approach with proper constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Mathematical approach: use sequential quadratic programming with proper constraints
    # We'll parameterize the problem as [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
    
    def setup_optimization_problem():
        # Initial configuration using a systematic approach
        # Start with a grid-like arrangement but allow optimization to adjust
        initial_positions = []
        
        # Create a structured initial placement
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        # Place points in a grid pattern with slight perturbation
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n:
                    break
                x = (j + 1) * spacing_x + np.random.normal(0, spacing_x/8)
                y = (i + 1) * spacing_y + np.random.normal(0, spacing_y/8)
                # Keep within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                initial_positions.append([x, y, 0.02])  # Small initial radius
                count += 1
            if count >= n:
                break
        
        # Fill remaining positions if needed
        while len(initial_positions) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            initial_positions.append([x, y, 0.02])
            
        return np.array(initial_positions)
    
    # Generate initial configuration
    circles = setup_optimization_problem()
    
    # Flatten initial parameters
    initial_params = circles.flatten()
    
    # Define bounds for optimization
    # Each circle has 3 parameters: x, y, r
    bounds = []
    for i in range(n):
        # x bounds: [0.001, 0.999] to ensure margin
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Constraints function
    def create_constraints():
        """Create constraint functions for non-overlapping and containment"""
        constraints = []
        
        # Containment constraints: for each circle, radius <= distance to edges
        def containment_constraint(params):
            result = []
            for i in range(n):
                x = params[3*i]
                y = params[3*i+1]
                r = params[3*i+2]
                
                # Distance to boundaries
                dist_left = x
                dist_right = 1 - x
                dist_bottom = y
                dist_top = 1 - y
                
                # Maximum allowed radius for containment
                max_radius = min(dist_left, dist_right, dist_bottom, dist_top)
                
                # Constraint: r <= max_radius (we want r to be as large as possible)
                # So we want: max_radius - r >= 0
                result.append(max_radius - r)
            return np.array(result)
        
        # Non-overlap constraints
        def overlap_constraint(params):
            result = []
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                    x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                    
                    # Euclidean distance
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    # Constraint: distance >= r1 + r2 (non-overlapping)
                    # So we want: dist - (r1 + r2) >= 0
                    result.append(dist - (r1 + r2))
            return np.array(result)
        
        # Add both types of constraints
        constraints.append({'type': 'ineq', 'fun': containment_constraint})
        constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return constraints
    
    # Create constraints
    constraints = create_constraints()
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        # Extract radii
        radii = params[2::3]  # Every third element starting from index 2
        return -np.sum(radii)  # Negative because we minimize
    
    # Run optimization
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective, 
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        # Extract final solution
        if result.success:
            final_params = result.x
            circles_final = final_params.reshape(n, 3)
        else:
            # Fallback to initial configuration if optimization fails
            circles_final = circles
            
    except Exception as e:
        # If optimization fails, return the initial configuration
        circles_final = circles
    
    # Final validation and adjustment
    # Make sure all circles respect boundary constraints
    for i in range(n):
        x, y, r = circles_final[i]
        # Adjust radius to fit within boundaries
        max_r = min(x, 1-x, y, 1-y)
        circles_final[i][2] = min(r, max_r)
    
    return circles_final


# EVOLVE-BLOCK-END
