# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses mathematical optimization with proper constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initial configuration: place circles in a grid pattern with small radii
    np.random.seed(42)
    
    # Create initial configuration with more strategic placement
    circles = np.zeros((n, 3))
    
    # Place in a grid-like pattern with some randomness
    rows = cols = int(np.ceil(np.sqrt(n)))
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x + np.random.uniform(-0.01, 0.01)
            y = (i + 1) * spacing_y + np.random.uniform(-0.01, 0.01)
            circles[idx] = [x, y, 0.02]
            idx += 1
        if idx >= n:
            break
    
    # Flatten parameters: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
    def pack_circles(params):
        # Reshape into circles array
        circles_flat = params.reshape((n, 3))
        
        # Extract coordinates and radii
        positions = circles_flat[:, :2]
        radii = circles_flat[:, 2]
        
        # Objective: maximize sum of radii (minimize negative sum)
        objective = -np.sum(radii)
        
        # Constraints
        constraints = []
        
        # Boundary constraints: each circle must fit completely in unit square
        for i in range(n):
            x, y, r = positions[i, 0], positions[i, 1], radii[i]
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3 + 0] - p[i*3 + 2]})  # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i*3 + 0] - p[i*3 + 2]})  # 1-x >= r
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3 + 1] - p[i*3 + 2]})  # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i*3 + 1] - p[i*3 + 2]})  # 1-y >= r
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                def overlap_constraint(p, i=i, j=j):
                    x1, y1, r1 = p[i*3], p[i*3+1], p[i*3+2]
                    x2, y2, r2 = p[j*3], p[j*3+1], p[j*3+2]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    return distance - (r1 + r2)
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return objective, constraints
    
    # More robust optimization approach using scipy minimize
    # First, create a better starting point with more careful initialization
    def get_initial_guess():
        # Better initialization: start with a known good configuration approach
        init_params = np.zeros(n * 3)
        
        # Fill with grid-based positions and small radii
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        for i in range(n):
            row = i // grid_size
            col = i % grid_size
            x = (col + 1) * spacing_x + np.random.uniform(-0.005, 0.005)
            y = (row + 1) * spacing_y + np.random.uniform(-0.005, 0.005)
            r = 0.02
            
            # Ensure they're within bounds
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            init_params[i*3] = x
            init_params[i*3+1] = y
            init_params[i*3+2] = r
            
        return init_params
    
    # Define the optimization problem properly
    def objective_and_constraints(params):
        circles_flat = params.reshape((n, 3))
        positions = circles_flat[:, :2]
        radii = circles_flat[:, 2]
        
        # Objective: maximize sum of radii (negative because minimize)
        obj_value = -np.sum(radii)
        
        # Constraints list
        cons = []
        
        # Boundary constraints
        for i in range(n):
            x, y, r = positions[i, 0], positions[i, 1], radii[i]
            # x >= r, 1-x >= r, y >= r, 1-y >= r
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3] - p[i*3+2]})  # x >= r
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i*3] - p[i*3+2]})  # 1-x >= r
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3+1] - p[i*3+2]})  # y >= r
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i*3+1] - p[i*3+2]})  # 1-y >= r
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                def overlap_constraint(p, i=i, j=j):
                    x1, y1, r1 = p[i*3], p[i*3+1], p[i*3+2]
                    x2, y2, r2 = p[j*3], p[j*3+1], p[j*3+2]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    return distance - (r1 + r2)
                cons.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return obj_value, cons
    
    # Use a more efficient approach with a custom optimizer
    # Since scipy.optimize.minimize has limitations with many constraints,
    # we'll use a hybrid approach with better initialization and bounds
    
    # Get initial guess
    initial_params = get_initial_guess()
    
    # Set bounds for parameters (x, y, r)
    bounds = []
    for i in range(n):
        # Bounds for x and y: [r, 1-r] 
        # But since we want to optimize, let's set broader bounds for optimization
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.49)])
    
    # Try a few different optimization strategies
    best_result = None
    best_sum = 0
    
    # Strategy 1: L-BFGS-B with bounds (fast)
    try:
        # We'll use a simplified version focusing on optimization of the sum of radii
        # with proper constraint handling
        
        # Simple but effective approach: first find a good initial solution
        # Then use gradient-based optimization
        
        # Create a more structured approach
        def compute_sum_of_radii(params):
            circles_flat = params.reshape((n, 3))
            return -np.sum(circles_flat[:, 2])  # Negative because we minimize
            
        def compute_constraints(params):
            circles_flat = params.reshape((n, 3))
            positions = circles_flat[:, :2]
            radii = circles_flat[:, 2]
            
            # Collect all constraint violations
            violations = []
            
            # Boundary constraints
            for i in range(n):
                x, y, r = positions[i, 0], positions[i, 1], radii[i]
                violations.append(r - x)  # x >= r
                violations.append(r - (1 - x))  # 1-x >= r
                violations.append(r - y)  # y >= r
                violations.append(r - (1 - y))  # 1-y >= r
                
            # Non-overlap constraints
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = positions[i, 0], positions[i, 1], radii[i]
                    x2, y2, r2 = positions[j, 0], positions[j, 1], radii[j]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    violations.append(distance - (r1 + r2))
                    
            return np.array(violations)
        
        # Start with a better initialization using a known good pattern
        def initialize_better():
            # Use a hexagonal packing approach for better density
            circles = np.zeros((n, 3))
            # Create a more uniform distribution
            grid_size = int(np.ceil(np.sqrt(n)))
            spacing_x = 0.95 / (grid_size + 1)
            spacing_y = 0.95 / (grid_size + 1)
            
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= n:
                        break
                    x = 0.025 + (j + 1) * spacing_x + np.random.uniform(-0.005, 0.005)
                    y = 0.025 + (i + 1) * spacing_y + np.random.uniform(-0.005, 0.005)
                    r = 0.02
                    circles[idx] = [x, y, r]
                    idx += 1
                if idx >= n:
                    break
                    
            return circles.flatten()
        
        # Final approach: Direct optimization with good initial conditions
        initial_params = initialize_better()
        
        # Optimization using scipy minimize with SLSQP method
        # This is a much simpler and more reliable approach
        result = minimize(
            compute_sum_of_radii,
            initial_params,
            method='SLSQP',
            bounds=[(0.001, 0.999), (0.001, 0.999), (0.001, 0.49)] * n,
            options={'maxiter': 5000, 'ftol': 1e-8, 'gtol': 1e-8},
            constraints=[
                {'type': 'ineq', 'fun': lambda p, i=i: p[i*3] - p[i*3+2]} for i in range(n)
            ] + [
                {'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i*3] - p[i*3+2]} for i in range(n)
            ] + [
                {'type': 'ineq', 'fun': lambda p, i=i: p[i*3+1] - p[i*3+2]} for i in range(n)
            ] + [
                {'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i*3+1] - p[i*3+2]} for i in range(n)
            ] + [
                {'type': 'ineq', 'fun': lambda p, i=i, j=j: np.sqrt((p[i*3]-p[j*3])**2 + (p[i*3+1]-p[j*3+1])**2) - (p[i*3+2] + p[j*3+2])} 
                for i in range(n) for j in range(i+1, n)
            ]
        )
        
        # Extract the best result
        if result.success:
            circles_final = result.x.reshape((n, 3))
        else:
            # Fallback to initial configuration if optimization fails
            circles_final = initial_params.reshape((n, 3))
            
        # Post-process to ensure constraints are satisfied
        # Clip positions to ensure they're within bounds
        for i in range(n):
            r = circles_final[i, 2]
            circles_final[i, 0] = np.clip(circles_final[i, 0], r, 1-r)
            circles_final[i, 1] = np.clip(circles_final[i, 1], r, 1-r)
            
        # Final validation - make sure no overlaps
        max_iter = 100
        for iter_count in range(max_iter):
            overlap_found = False
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles_final[i, 0], circles_final[i, 1], circles_final[i, 2]
                    x2, y2, r2 = circles_final[j, 0], circles_final[j, 1], circles_final[j, 2]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if distance < r1 + r2:
                        # Move circles apart
                        overlap = (r1 + r2) - distance
                        dx = (x2 - x1) / distance * overlap * 0.5
                        dy = (y2 - y1) / distance * overlap * 0.5
                        circles_final[i, 0] -= dx
                        circles_final[i, 1] -= dy
                        circles_final[j, 0] += dx
                        circles_final[j, 1] += dy
                        overlap_found = True
                        
                        # Keep within bounds
                        r1 = circles_final[i, 2]
                        r2 = circles_final[j, 2]
                        circles_final[i, 0] = np.clip(circles_final[i, 0], r1, 1-r1)
                        circles_final[i, 1] = np.clip(circles_final[i, 1], r1, 1-r1)
                        circles_final[j, 0] = np.clip(circles_final[j, 0], r2, 1-r2)
                        circles_final[j, 1] = np.clip(circles_final[j, 1], r2, 1-r2)
            
            if not overlap_found:
                break
        
        # Return final result
        return circles_final
        
    except Exception as e:
        # If optimization fails, return a reasonable fallback
        warnings.warn(f"Optimization failed: {str(e)}")
        # Return a simple configuration
        circles = np.zeros((n, 3))
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x + np.random.uniform(-0.01, 0.01)
                y = (i + 1) * spacing_y + np.random.uniform(-0.01, 0.01)
                circles[idx] = [x, y, 0.02]
                idx += 1
            if idx >= n:
                break
        
        return circles


# EVOLVE-BLOCK-END
