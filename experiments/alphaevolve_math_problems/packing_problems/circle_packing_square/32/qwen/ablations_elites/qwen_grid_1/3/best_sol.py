# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining smart initialization, gradient-based optimization, 
    and iterative refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using hexagonal packing pattern with good spread
    def initialize_circles():
        # Create a more structured hexagonal grid pattern for better initial placement
        circles = np.zeros((n, 3))
        
        # Use a 6x6 grid with some randomness for good distribution
        rows = 6
        cols = 6
        
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset pattern
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                # Add small randomization to avoid regular patterns
                x += (np.random.random() - 0.5) * spacing_x * 0.2
                y += (np.random.random() - 0.5) * spacing_y * 0.2
                
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Initial radius - start with small values
                r = min(spacing_x, spacing_y) * 0.2
                
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with strategic random placements
        for i in range(idx, n):
            # Place in central area with some variation
            x = 0.3 + np.random.random() * 0.4
            y = 0.3 + np.random.random() * 0.4
            r = 0.02 + np.random.random() * 0.05
            
            # Ensure within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[i] = [x, y, r]
            
        return circles
    
    # Define constraints more carefully for better numerical stability
    def create_constraints():
        cons = []
        
        # Boundary constraints: radius <= x <= 1-radius, radius <= y <= 1-radius
        for i in range(n):
            # These are the constraints: x >= r, y >= r, x <= 1-r, y <= 1-r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})  # x >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})  # y >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})  # 1-x >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})  # 1-y >= r
            
        # Non-overlap constraints: distance >= r1 + r2 (using squared distances for numerical stability)
        for i in range(n):
            for j in range(i+1, n):
                # We want (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
                })
                
        return cons
    
    # Objective function: negative sum of radii (we want to maximize sum)
    def objective(x):
        return -np.sum(x[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Initialize
    circles = initialize_circles()
    
    # Flatten initial guess: [x1, y1, r1, x2, y2, r2, ...]
    x0 = circles.flatten()
    
    # Create bounds: [0.01,0.99] for positions, [0.001,0.4] for radii
    bounds = []
    for i in range(n):
        bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.001, 0.4)])  # x, y, r bounds
    
    # Get constraints
    constraints = create_constraints()
    
    # Optimization with multiple strategies
    try:
        # First attempt with SLSQP - more robust settings
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1500, 'ftol': 1e-9, 'eps': 1e-7, 'iprint': 0}
        )
        
        if result.success:
            optimized = result.x.reshape(-1, 3)
            # Final validation and refinement
            final_result = validate_and_refine(optimized)
            return final_result
        else:
            # Fall back to simpler approach
            return refine_simple(circles)
            
    except Exception as e:
        # Fallback to simpler approach if optimization fails
        return refine_simple(circles)

def validate_and_refine(circles):
    """Refine the solution to ensure constraints are met"""
    n = len(circles)
    
    # Ensure all circles are within bounds and non-overlapping
    for i in range(n):
        # Enforce boundary constraints
        x, y, r = circles[i]
        circles[i] = [max(r, min(1-r, x)), max(r, min(1-r, y)), r]
    
    # Refine by checking overlaps and adjusting radii - more aggressive approach
    for iteration in range(200):  # Even more iterations for better convergence
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Calculate distance squared for numerical stability
                dx = x1 - x2
                dy = y1 - y2
                dist_squared = dx*dx + dy*dy
                min_dist_squared = (r1 + r2)**2
                
                if dist_squared < min_dist_squared:
                    # Overlap detected, reduce radii
                    reduction = (np.sqrt(min_dist_squared) - np.sqrt(dist_squared)) * 0.5  # Even more aggressive
                    if r1 > reduction and r2 > reduction:
                        circles[i][2] -= reduction
                        circles[j][2] -= reduction
                        changed = True
                        
        if not changed:
            break
    
    # Final pass: try to improve radii by pushing them up where possible
    # This is a more sophisticated approach - we'll try to find optimal radii
    for iteration in range(100):
        improved = False
        for i in range(n):
            # Find the maximum possible radius for this circle
            max_radius = min(
                circles[i, 0], 1 - circles[i, 0],  # x bounds
                circles[i, 1], 1 - circles[i, 1]   # y bounds
            )
            
            # Check overlap with all others
            for j in range(n):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist_squared = dx*dx + dy*dy
                    if dist_squared > 0:
                        max_radius = min(max_radius, np.sqrt(dist_squared) - circles[j, 2])
            
            max_radius = max(1e-6, max_radius)
            
            if max_radius > circles[i, 2]:
                # Try to push radius up to maximum while maintaining feasibility
                # But be careful not to cause overlaps
                test_radius = max_radius
                valid = True
                
                # Check if this would cause any overlaps
                for j in range(n):
                    if i != j:
                        dx = circles[i, 0] - circles[j, 0]
                        dy = circles[i, 1] - circles[j, 1]
                        dist_squared = dx*dx + dy*dy
                        if dist_squared < (test_radius + circles[j, 2])**2:
                            valid = False
                            break
                
                if valid:
                    circles[i, 2] = test_radius
                    improved = True
                    
        if not improved:
            break
    
    return np.array(circles)

def refine_simple(initial_circles):
    """Simple refinement approach for fallback"""
    n = len(initial_circles)
    circles = initial_circles.copy()
    
    # Simple iterative improvement with better logic
    for _ in range(500):  # Even more iterations for better exploration
        improved = False
        for i in range(n):
            # Try to slightly increase radius while maintaining constraints
            old_r = circles[i][2]
            test_r = min(old_r * 1.03, 0.4)  # Slightly higher increase rate
            
            # Check if we can safely increase radius
            valid = True
            for j in range(n):
                if i != j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_squared = dx*dx + dy*dy
                    if dist_squared < (test_r + r2)**2:
                        valid = False
                        break
            
            if valid:
                # Check boundary constraints
                x, y, _ = circles[i]
                if x >= test_r and x <= 1-test_r and y >= test_r and y <= 1-test_r:
                    circles[i][2] = test_r
                    improved = True
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
