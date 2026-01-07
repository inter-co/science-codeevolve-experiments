# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import random
import time
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with gradient-based optimization
    and spatial indexing for efficient constraint checking.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 32
    
    # Improved initialization using a more sophisticated approach
    # Start with a regular hexagonal lattice pattern
    circles = np.zeros((n, 3))
    
    # Create a hexagonal packing pattern with more careful spacing
    rows = 6
    cols = 6
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Use a more refined hexagonal packing approach
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset odd rows for hexagonal packing
            x_offset = spacing_x * 0.5 if i % 2 == 1 else 0
            x = (j + 0.5) * spacing_x + x_offset
            y = (i + 0.5) * spacing_y
            
            # Initial radius - start with smaller values to allow more optimization
            r = min(spacing_x, spacing_y) * 0.35
            
            # Ensure it fits in the square
            if x + r <= 1 and y + r <= 1 and x - r >= 0 and y - r >= 0:
                circles[idx] = [x, y, r]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with strategic placement
    for i in range(idx, n):
        attempts = 0
        while attempts < 100:
            # Try to place near corners or edges for better packing
            if np.random.random() < 0.6:  # 60% chance to try strategic placement
                # Corner placement
                corner = np.random.choice(['tl', 'tr', 'bl', 'br'])
                if corner == 'tl':
                    x = np.random.uniform(0.01, 0.15)
                    y = np.random.uniform(0.01, 0.15)
                elif corner == 'tr':
                    x = np.random.uniform(0.85, 0.99)
                    y = np.random.uniform(0.01, 0.15)
                elif corner == 'bl':
                    x = np.random.uniform(0.01, 0.15)
                    y = np.random.uniform(0.85, 0.99)
                else:  # br
                    x = np.random.uniform(0.85, 0.99)
                    y = np.random.uniform(0.85, 0.99)
            else:
                # Random placement with bias towards center
                x = np.random.beta(2, 2) * 0.8 + 0.1  # Beta distribution centered around 0.5
                y = np.random.beta(2, 2) * 0.8 + 0.1
                
            # Calculate maximum possible radius at this location
            max_r = min(x, 1-x, y, 1-y)
            
            # Use a more conservative initial radius to prevent overlap issues
            r = max(0.005, min(max_r * 0.5, 0.12))
            
            if r > 0.005:
                circles[i] = [x, y, r]
                break
            attempts += 1
    
    # Optimization with improved constraint handling and spatial indexing
    def objective(x):
        # Maximize sum of radii (minimize negative sum)
        return -np.sum(x[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Spatial indexing for faster constraint evaluation
    def evaluate_constraints(x, spatial_index=False):
        """Evaluate constraints more efficiently"""
        # Check boundary constraints
        boundary_satisfied = True
        for i in range(n):
            x_pos, y_pos, r = x[3*i], x[3*i+1], x[3*i+2]
            if x_pos - r < 0 or x_pos + r > 1 or y_pos - r < 0 or y_pos + r > 1:
                boundary_satisfied = False
                break
        
        if not boundary_satisfied:
            return False
            
        # Check overlap constraints using spatial indexing if enabled
        if spatial_index:
            # Create spatial tree for fast neighbor search
            points = x.reshape(-1, 3)[:, :2]  # Extract (x,y) coordinates
            tree = cKDTree(points)
            
            # Find neighbors within 2*max_radius distance (optimization)
            max_radius = np.max(x[2::3])
            pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
            
            # Check only relevant pairs
            for i, j in pairs:
                if i < j:  # Only check each pair once
                    x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                    x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    if dist_sq < (r_i + r_j)**2 - 1e-10:  # Small epsilon for numerical stability
                        return False
        else:
            # Brute force check for small number of circles
            for i in range(n):
                for j in range(i+1, n):
                    x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                    x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    if dist_sq < (r_i + r_j)**2 - 1e-10:  # Small epsilon for numerical stability
                        return False
                        
        return True
    
    # Create constraints more efficiently
    def create_constraints():
        """Create constraints more efficiently"""
        cons = []
        
        # Boundary constraints
        for i in range(n):
            def make_boundary_func(i):
                def boundary_func(x):
                    x_pos, y_pos, r = x[3*i], x[3*i+1], x[3*i+2]
                    return min(x_pos - r, 1 - x_pos - r, y_pos - r, 1 - y_pos - r)
                return boundary_func
            cons.append({'type': 'ineq', 'fun': make_boundary_func(i)})
        
        # Non-overlap constraints - use spatial indexing approach for efficiency
        # Instead of all pairs, we can use spatial indexing or just a subset for optimization
        # But for full accuracy, we'll use all pairs but with better structure
        for i in range(n):
            for j in range(i+1, n):
                def make_overlap_func(i, j):
                    def overlap_func(x):
                        x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                        x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                        dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                        # Return positive when constraint satisfied (distance >= r_i + r_j)
                        return dist_sq - (r_i + r_j)**2
                    return overlap_func
                cons.append({'type': 'ineq', 'fun': make_overlap_func(i, j)})
                
        return cons
    
    # Flatten initial guess
    x0 = np.array([circles[i][j] for i in range(n) for j in range(3)])
    
    # Set bounds: x, y in [0,1], r in [0, 0.5] (reasonable upper bound)
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Optimization parameters
    options = {'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6, 'disp': False}
    
    # Try different optimization approaches in order of preference
    result = None
    
    try:
        # First approach: L-BFGS-B (often fastest for this kind of problem)
        print("Trying L-BFGS-B...")
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options=options)
        
        # If L-BFGS-B fails or produces poor results, try other methods
        if not result.success or result.fun > -100:
            print("L-BFGS-B failed or produced poor results, trying SLSQP...")
            cons = create_constraints()
            result = minimize(objective, x0, method='SLSQP', constraints=cons, 
                             bounds=bounds, options=options)
        
        # If still failing, try Trust-Constr which is often more robust
        if not result.success or result.fun > -100:
            print("SLSQP also failed, trying Trust-Constr...")
            try:
                result = minimize(objective, x0, method='trust-constr', bounds=bounds, 
                                 options=options)
            except:
                pass  # If trust-constr fails, continue with fallback
        
        # Extract final solution if successful
        if result and result.success:
            final_circles = np.zeros((n, 3))
            for i in range(n):
                final_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            return final_circles
        
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        pass  # Continue to fallback
    
    # Fallback to initial configuration if optimization fails
    print("Falling back to initial configuration")
    return circles


# EVOLVE-BLOCK-END
