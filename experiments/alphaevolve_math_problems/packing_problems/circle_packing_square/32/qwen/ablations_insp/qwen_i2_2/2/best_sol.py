# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random
import time
from itertools import combinations

# Try to import cvxpy for better convex optimization
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a mathematically rigorous approach leveraging convex optimization concepts.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 32
    
    # Better initialization using a more structured approach
    # Start with a grid pattern and refine
    circles = np.zeros((n, 3))
    
    # Create a more systematic grid pattern with better distribution
    # Use 6x6 grid for 32 circles (with some refinement)
    rows, cols = 6, 6
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset odd rows for better hexagonal-like packing
            x_offset = spacing_x * 0.5 if i % 2 == 1 else 0
            x = (j + 0.5) * spacing_x + x_offset
            y = (i + 0.5) * spacing_y
            
            # Initial radius - start with a reasonable value
            r = min(spacing_x, spacing_y) * 0.35
            
            # Ensure it fits in the square
            if x + r <= 1 and y + r <= 1 and x - r >= 0 and y - r >= 0:
                circles[idx] = [x, y, r]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with strategic placements near boundaries
    # to avoid getting trapped in local optima
    for i in range(idx, n):
        # Try placing near edges first to improve global coverage
        if i < n:
            # Place randomly but bias towards center areas with good radius
            attempts = 0
            while attempts < 50:
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                
                # Calculate maximum possible radius at this location
                max_r = min(x, 1-x, y, 1-y)
                
                # Start with a more aggressive radius to encourage better packing
                r = max(0.01, min(max_r * 0.8, 0.12))
                
                if r > 0.01:
                    circles[i] = [x, y, r]
                    break
                attempts += 1
    
    # If we have cvxpy available, use a more robust convex optimization approach
    if HAS_CVXPY:
        try:
            # Define variables for optimization
            x_vars = cp.Variable(n)
            y_vars = cp.Variable(n)
            r_vars = cp.Variable(n)
            
            # Objective: maximize sum of radii
            objective = cp.Maximize(cp.sum(r_vars))
            
            # Constraints
            constraints = []
            
            # Boundary constraints
            for i in range(n):
                constraints.append(x_vars[i] >= r_vars[i])
                constraints.append(y_vars[i] >= r_vars[i])
                constraints.append(x_vars[i] <= 1 - r_vars[i])
                constraints.append(y_vars[i] <= 1 - r_vars[i])
            
            # Non-overlap constraints (more efficient than pairwise)
            for i in range(n):
                for j in range(i+1, n):
                    # Distance constraint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
                    constraints.append(
                        cp.square(x_vars[i] - x_vars[j]) + 
                        cp.square(y_vars[i] - y_vars[j]) >= 
                        cp.square(r_vars[i] + r_vars[j])
                    )
            
            # Create and solve the problem
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS, verbose=False, max_iters=1000)
            
            # Extract results if successful
            if prob.status in ["optimal", "optimal_inaccurate"]:
                final_circles = np.zeros((n, 3))
                for i in range(n):
                    final_circles[i] = [x_vars[i].value, y_vars[i].value, r_vars[i].value]
                return final_circles
                
        except Exception as e:
            # Fall back to numerical optimization if cvxpy fails
            pass
    
    # Fallback to numerical optimization with improved strategy
    def objective(x):
        # Maximize sum of radii (minimize negative sum)
        return -np.sum(x[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    def create_efficient_constraints():
        """Create constraints more efficiently - only necessary ones"""
        cons = []
        
        # Boundary constraints
        for i in range(n):
            def boundary_constraint(i):
                def func(x):
                    x_pos, y_pos, r = x[3*i], x[3*i+1], x[3*i+2]
                    # Return positive when all boundary constraints are satisfied
                    return min(x_pos - r, 1 - x_pos - r, y_pos - r, 1 - y_pos - r)
                return func
            
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Non-overlap constraints - reduce the number significantly by using spatial indexing
        # Instead of all pairs, we can limit to nearby circles
        from scipy.spatial import cKDTree
        
        # Build spatial tree for faster neighbor lookup
        coords = np.array([[circles[i][0], circles[i][1]] for i in range(n)])
        tree = cKDTree(coords)
        
        # For each circle, only consider nearby neighbors for overlap constraints
        # This reduces the number of constraints significantly
        for i in range(n):
            # Find neighbors within a reasonable distance
            neighbors = tree.query_ball_point(coords[i], 0.5)
            for j in neighbors:
                if i < j:  # Only add each pair once
                    def overlap_constraint(i, j):
                        def func(x):
                            x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                            x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                            # Return positive when constraint satisfied (distance >= r_i + r_j)
                            return dist_sq - (r_i + r_j)**2
                        return func
                    
                    cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
            
        return cons
    
    # Create constraints once (but with fewer constraints for efficiency)
    cons = create_efficient_constraints()
    
    # Flatten initial guess
    x0 = np.array([circles[i][j] for i in range(n) for j in range(3)])
    
    # Set bounds: x, y in [0,1], r in [0, 0.5] (reasonable upper bound)
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Optimization parameters
    options = {'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6, 'disp': False}
    
    try:
        # Try L-BFGS-B first (often very effective for smooth problems)
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options=options, callback=None)
        
        # If that fails or doesn't converge well, try SLSQP with constraints
        if not result.success or result.fun > -100:
            # Reduce number of iterations for faster execution
            options['maxiter'] = 300
            result = minimize(objective, x0, method='SLSQP', constraints=cons, 
                             bounds=bounds, options=options)
        
        # Extract final solution
        final_circles = np.zeros((n, 3))
        for i in range(n):
            final_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            
        return final_circles
        
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        print(f"Optimization failed: {e}")
        return circles


# EVOLVE-BLOCK-END
