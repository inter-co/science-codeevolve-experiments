# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from scipy.spatial import cKDTree

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and adaptive optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Multi-start strategy to avoid local optima
    best_sum = 0
    best_circles = None
    
    # Try multiple initialization strategies with different seeds
    seeds = [42, 123, 456, 789, 999, 1337, 5555]
    
    for seed in seeds:
        # Initialize circles using a more sophisticated hexagonal grid approach
        circles = initialize_hexagonal_grid(n, seed)
        
        # Refine using optimization with careful constraint handling
        circles = optimize_circles_refined(circles)
        
        # Evaluate current solution
        current_sum = np.sum(circles[:, 2])
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
    
    # If no good solution found, return the best from the last attempt
    return best_circles if best_circles is not None else initialize_hexagonal_grid(n, 42)

def initialize_hexagonal_grid(n: int, seed: int = 0) -> np.ndarray:
    """Initialize circle positions using a hexagonal grid pattern for better initial distribution."""
    np.random.seed(seed)
    
    # Create a more refined hexagonal grid - use 6x6 grid for 32 circles
    rows = 6
    cols = 6
    
    # Calculate spacing to fit nicely in unit square
    spacing_x = 0.9 / cols  # Slightly smaller than 0.9 to allow for better packing
    spacing_y = 0.9 / rows
    
    circles = []
    count = 0
    
    # Create hexagonal pattern
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
                
            # Offset every other row for hexagonal packing
            x_offset = (i % 2) * spacing_x / 2
            x = 0.05 + (j + 0.5) * spacing_x + x_offset  # Centered with margin
            y = 0.05 + (i + 0.5) * spacing_y
            
            # Add slight randomness to avoid perfect grid that might cause optimization issues
            x += np.random.normal(0, 0.002 * spacing_x)
            y += np.random.normal(0, 0.002 * spacing_y)
            
            # Initial radius guess - start with a reasonable value
            r = min(spacing_x, spacing_y) * 0.35  # Slightly larger initial radius
            
            # Ensure we stay within bounds
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                circles.append([x, y, max(0.001, r)])
                count += 1
                
        if count >= n:
            break
    
    # Fill remaining circles with random placements near center with more care
    while len(circles) < n:
        x = 0.45 + np.random.uniform(-0.15, 0.15)  # Centered area
        y = 0.45 + np.random.uniform(-0.15, 0.15)
        r = 0.02 + np.random.uniform(0, 0.03)  # Range for variation
        
        # Ensure radius respects boundaries
        r = min(r, x, 1-x, y, 1-y)
        circles.append([x, y, max(0.001, r)])
        
    return np.array(circles)

def optimize_circles_refined(initial_circles: np.ndarray) -> np.ndarray:
    """Use refined constrained optimization with better handling."""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i, 0], initial_circles[i, 1], initial_circles[i, 2]])
    
    # Objective function: negative sum of radii (we want to maximize sum of radii)
    def objective(params):
        total_radius = sum(params[3*i+2] for i in range(n))
        return -total_radius
    
    # Create constraints with better numerical handling and spatial indexing
    def create_constraints():
        cons = []
        
        # Boundary constraints: radius <= x <= 1-radius and radius <= y <= 1-radius
        for i in range(n):
            # x >= r (equivalent to x - r >= 0)
            cons.append({'type': 'ineq', 'fun': lambda params, idx=i: params[3*idx] - params[3*idx+2]})
            # y >= r (equivalent to y - r >= 0)  
            cons.append({'type': 'ineq', 'fun': lambda params, idx=i: params[3*idx+1] - params[3*idx+2]})
            # x <= 1-r (equivalent to 1 - x - r >= 0)
            cons.append({'type': 'ineq', 'fun': lambda params, idx=i: 1 - params[3*idx] - params[3*idx+2]})
            # y <= 1-r (equivalent to 1 - y - r >= 0)
            cons.append({'type': 'ineq', 'fun': lambda params, idx=i: 1 - params[3*idx+1] - params[3*idx+2]})
            
        # Non-overlap constraints using spatial indexing for efficiency
        # First compute all pairwise distances to identify close pairs
        try:
            # Use spatial indexing to reduce constraint evaluation
            positions = np.array([[params[3*i], params[3*i+1]] for i in range(n)])
            tree = cKDTree(positions)
            
            # Find pairs within a certain distance threshold
            pairs = tree.query_pairs(0.1, output_type='ndarray')  # Adjust threshold as needed
            
            # Add constraints for all pairs (more efficient than full matrix)
            for i, j in zip(pairs[0], pairs[1]):
                def overlap_constraint(params, idx1=i, idx2=j):
                    x1, y1, r1 = params[3*idx1], params[3*idx1+1], params[3*idx1+2]
                    x2, y2, r2 = params[3*idx2], params[3*idx2+1], params[3*idx2+2]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    # Add small epsilon to prevent numerical issues
                    return dist_sq - (r1 + r2)**2 + 1e-12
                
                cons.append({'type': 'ineq', 'fun': overlap_constraint})
                
        except Exception:
            # Fallback to direct computation if spatial indexing fails
            for i in range(n):
                for j in range(i+1, n):
                    def overlap_constraint(params, idx1=i, idx2=j):
                        x1, y1, r1 = params[3*idx1], params[3*idx1+1], params[3*idx1+2]
                        x2, y2, r2 = params[3*idx2], params[3*idx2+1], params[3*idx2+2]
                        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                        # Add small epsilon to prevent numerical issues
                        return dist_sq - (r1 + r2)**2 + 1e-12
                    
                    cons.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return cons
    
    # Bounds for parameters: x in [r, 1-r], y in [r, 1-r], r in [0.001, 0.499] 
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Get constraints
    constraints = create_constraints()
    
    # Try multiple optimization approaches with better settings
    methods = ['SLSQP', 'trust-constr']
    
    for method in methods:
        try:
            # Use more aggressive settings for better convergence
            result = minimize(objective, initial_params, method=method, bounds=bounds, 
                             constraints=constraints, options={'maxiter': 500, 'ftol': 1e-7, 'gtol': 1e-7})
            
            if result.success:
                final_params = result.x
                circles = np.zeros((n, 3))
                for i in range(n):
                    circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
                return circles
        except Exception as e:
            # Log error and continue with next method
            continue
    
    # Fallback: try with L-BFGS-B with even more conservative settings
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 400, 'ftol': 1e-7})
        
        if result.success:
            final_params = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
            return circles
    except Exception:
        pass
    
    # Final fallback: try a simple iterative improvement approach
    try:
        circles = initial_circles.copy()
        for iteration in range(50):
            improved = False
            # Try to increase radii where possible
            for i in range(n):
                x, y, r = circles[i]
                # Maximum possible radius at this position
                max_r = min(x, y, 1-x, 1-y)
                
                # Try to increase radius
                if r < max_r and r < 0.499:
                    new_r = min(r * 1.02, max_r)
                    if new_r > r + 0.0001:
                        # Test if this change is valid with neighbors
                        valid = True
                        for j in range(n):
                            if i != j:
                                x1, y1, r1 = circles[i][0], circles[i][1], new_r
                                x2, y2, r2 = circles[j]
                                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                                if dist_sq < (r1 + r2)**2:
                                    valid = False
                                    break
                        
                        if valid:
                            circles[i, 2] = new_r
                            improved = True
            if not improved:
                break
        return circles
    except Exception:
        pass
    
    # If all optimization methods fail, return the initial configuration
    return initial_circles


# EVOLVE-BLOCK-END
