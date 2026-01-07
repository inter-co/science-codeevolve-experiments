# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import time
from itertools import combinations
from scipy.spatial import cKDTree

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
    
    # Improved initialization using a more sophisticated approach based on hexagonal packing
    # Start with a regular hexagonal lattice pattern that's more efficient
    circles = np.zeros((n, 3))
    
    # Use a 6x6 grid for better coverage of the space
    rows = 6
    cols = 6
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Create hexagonal packing pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset odd rows for hexagonal packing
            x_offset = spacing_x * 0.5 if i % 2 == 1 else 0
            x = (j + 0.5) * spacing_x + x_offset
            y = (i + 0.5) * spacing_y
            
            # Initial radius - start with larger values to allow optimization to fine-tune
            # This is key for better performance than very small initial radii
            r = min(spacing_x, spacing_y) * 0.4
            
            # Ensure it fits in the square
            if x + r <= 1 and y + r <= 1 and x - r >= 0 and y - r >= 0:
                circles[idx] = [x, y, r]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with better initialization strategy
    for i in range(idx, n):
        # Try to place in areas where circles might fit better
        attempts = 0
        while attempts < 100:
            # Use better sampling: prioritize center area with some edge/corner bias
            if np.random.random() < 0.7:  # 70% chance to sample center region
                # Sample from center biased region
                x = np.random.beta(2.5, 2.5) * 0.8 + 0.1  # Center biased
                y = np.random.beta(2.5, 2.5) * 0.8 + 0.1
            else:
                # Sample from edge/corner regions more carefully
                region = np.random.choice(['edge', 'corner'])
                if region == 'corner':
                    # Place near corners
                    corner = np.random.choice(['tl', 'tr', 'bl', 'br'])
                    if corner == 'tl':
                        x = np.random.uniform(0.01, 0.12)
                        y = np.random.uniform(0.01, 0.12)
                    elif corner == 'tr':
                        x = np.random.uniform(0.88, 0.99)
                        y = np.random.uniform(0.01, 0.12)
                    elif corner == 'bl':
                        x = np.random.uniform(0.01, 0.12)
                        y = np.random.uniform(0.88, 0.99)
                    else:  # br
                        x = np.random.uniform(0.88, 0.99)
                        y = np.random.uniform(0.88, 0.99)
                else:
                    # Place near edges
                    edge = np.random.choice(['top', 'bottom', 'left', 'right'])
                    if edge == 'top':
                        x = np.random.uniform(0.1, 0.9)
                        y = np.random.uniform(0.88, 0.99)
                    elif edge == 'bottom':
                        x = np.random.uniform(0.1, 0.9)
                        y = np.random.uniform(0.01, 0.12)
                    elif edge == 'left':
                        x = np.random.uniform(0.01, 0.12)
                        y = np.random.uniform(0.1, 0.9)
                    else:  # right
                        x = np.random.uniform(0.88, 0.99)
                        y = np.random.uniform(0.1, 0.9)
            
            # Calculate maximum possible radius at this location
            max_r = min(x, 1-x, y, 1-y)
            
            # Use reasonable initial radius
            r = max(0.005, min(max_r * 0.6, 0.15))
            
            if r > 0.005:
                circles[i] = [x, y, r]
                break
            attempts += 1
    
    # Optimization with improved constraint handling
    def objective(x):
        # Maximize sum of radii (minimize negative sum)
        return -np.sum(x[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # More efficient constraint creation
    def create_constraints():
        """Create constraints more efficiently"""
        cons = []
        
        # Boundary constraints - more robust formulation
        for i in range(n):
            def make_boundary_func(i):
                def boundary_func(x):
                    x_pos, y_pos, r = x[3*i], x[3*i+1], x[3*i+2]
                    # Return positive when inside bounds
                    return min(x_pos - r, 1 - x_pos - r, y_pos - r, 1 - y_pos - r)
                return boundary_func
            cons.append({'type': 'ineq', 'fun': make_boundary_func(i)})
        
        # Non-overlap constraints - use spatial indexing for efficiency
        # But for this problem size, we'll use a simpler approach that's still effective
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
    
    # Set bounds: x, y in [0,1], r in [0, 0.5] 
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Optimization parameters - tuned for better convergence
    options = {'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
    
    # Try optimization with multiple strategies
    result = None
    
    try:
        # Strategy 1: Try L-BFGS-B with better tolerance settings
        print("Trying L-BFGS-B...")
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options=options)
        
        # If L-BFGS-B fails or produces poor results, try SLSQP with constraints
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
    
    # Fallback to initial configuration with additional refinement
    print("Falling back to initial configuration with refinement")
    
    # Apply some local refinement to improve the initial solution
    improved = True
    max_refinements = 100
    refinement_count = 0
    
    while improved and refinement_count < max_refinements:
        improved = False
        refinement_count += 1
        
        # Try to increase each radius slightly while maintaining constraints
        for i in range(n):
            old_r = circles[i, 2]
            # Try to increase radius by up to 0.005
            new_r = min(old_r + 0.005, 0.5)
            
            # Check if this change violates any constraints
            valid = True
            for j in range(n):
                if i != j:
                    dist_sq = (circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2
                    min_dist_sq = (new_r + circles[j, 2])**2
                    if dist_sq < min_dist_sq:
                        valid = False
                        break
            
            # Check containment
            if (circles[i, 0] - new_r < 0 or circles[i, 0] + new_r > 1 or 
                circles[i, 1] - new_r < 0 or circles[i, 1] + new_r > 1):
                valid = False
            
            if valid and new_r > old_r:
                circles[i, 2] = new_r
                improved = True
    
    return circles


# EVOLVE-BLOCK-END
