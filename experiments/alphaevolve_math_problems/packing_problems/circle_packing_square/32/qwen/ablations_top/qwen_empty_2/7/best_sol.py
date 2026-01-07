# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial seeding with optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))
    
    # Stage 1: Generate initial configuration using hexagonal packing heuristic
    # This gives us a reasonable starting point
    initial_config = _generate_initial_configuration(n)
    
    # Stage 2: Optimize using scipy's constrained optimization
    optimized_result = _optimize_circles(initial_config)
    
    # Stage 3: Refine with local optimization
    final_result = _local_refinement(optimized_result)
    
    return final_result

def _generate_initial_configuration(n: int) -> np.ndarray:
    """Generate initial circle positions using hexagonal packing heuristic"""
    # Create a grid-based initial configuration
    circles = np.zeros((n, 3))
    
    # Try to place circles in a hexagonal pattern
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Adjust dimensions to fit in unit square
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    # Fill the grid with circles
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + 1) * spacing_x + offset * spacing_x * 0.5
            y = (i + 1) * spacing_y
            
            # Initial radius - small to allow room for optimization
            r = min(spacing_x, spacing_y) * 0.2
            
            # Ensure we're within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining circles with random positions but valid radii
    for i in range(idx, n):
        # Random position with bounded radius
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = min(x, 1-x, y, 1-y) * 0.3
        circles[i] = [x, y, r]
        
    return circles

def _constraint_violation(circles: np.ndarray) -> float:
    """Calculate total constraint violation"""
    n = len(circles)
    violation = 0.0
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1-r or y < r or y > 1-r:
            violation += abs(min(x-r, r-x, y-r, r-y))
    
    # Check overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1-x2)**2 + (y1-y2)**2
            min_dist_sq = (r1+r2)**2
            if dist_sq < min_dist_sq:
                violation += min_dist_sq - dist_sq
                
    return violation

def _objective(circles: np.ndarray) -> float:
    """Objective function to maximize sum of radii"""
    return -np.sum(circles[:, 2])  # Negative because we minimize

def _constraints(circles: np.ndarray) -> dict:
    """Define all constraints for optimization"""
    n = len(circles)
    cons = []
    
    # Containment constraints: r <= x <= 1-r and r <= y <= 1-r
    def containment_constraint(i):
        def func(x):
            return np.array([x[2*i] - x[2*i+1], 1-x[2*i+1] - x[2*i], 
                           x[2*i+1] - x[2*i+2], 1-x[2*i+2] - x[2*i+1]])
        return func
    
    # Overlap constraints: sqrt((x1-x2)^2 + (y1-y2)^2) >= r1 + r2
    def overlap_constraint(i, j):
        def func(x):
            x1, y1, r1 = x[2*i], x[2*i+1], x[2*i+2]
            x2, y2, r2 = x[2*j], x[2*j+1], x[2*j+2]
            dist_sq = (x1-x2)**2 + (y1-y2)**2
            min_dist_sq = (r1+r2)**2
            return dist_sq - min_dist_sq
        return func
    
    # Add containment constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*i+1]})  # r <= x
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1-x[2*i+1] - x[2*i]})  # x <= 1-r
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*i+2]})  # r <= y
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1-x[2*i+2] - x[2*i+1]})  # y <= 1-r
    
    # Add overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: overlap_constraint(i, j)(x)})
    
    return cons

def _optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circles using scipy minimize with constraints"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = initial_circles[i, 0]  # x
        x0[3*i+1] = initial_circles[i, 1]  # y
        x0[3*i+2] = initial_circles[i, 2]  # r
    
    # Define bounds for optimization
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])  # x, y, r bounds
    
    # Constraints for optimization
    def contain_bounds(x):
        # Ensure circles are contained in unit square
        violations = []
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            violations.append(xi - ri)  # x - r >= 0
            violations.append(1 - xi - ri)  # 1 - x - r >= 0
            violations.append(yi - ri)  # y - r >= 0
            violations.append(1 - yi - ri)  # 1 - y - r >= 0
        return np.array(violations)
    
    def overlap_constraints(x):
        # Ensure no overlaps
        violations = []
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist_sq = (ri+rj)**2
                violations.append(dist_sq - min_dist_sq)
        return np.array(violations)
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: contain_bounds(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
    ]
    
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            lambda x: -np.sum(x[2::3]),  # Maximize sum of radii
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            optimized = initial_circles.copy()
            for i in range(n):
                optimized[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            return optimized
    except Exception:
        pass
    
    return initial_circles

def _local_refinement(circles: np.ndarray) -> np.ndarray:
    """Apply local refinement to improve solution"""
    # Simple gradient-based refinement approach
    n = len(circles)
    refined = circles.copy()
    
    # Try small perturbations to improve objective
    best_sum = np.sum(refined[:, 2])
    improved = True
    iterations = 0
    
    while improved and iterations < 50:
        improved = False
        iterations += 1
        
        # Try small moves for each circle
        for i in range(n):
            original_x, original_y, original_r = refined[i]
            best_move = [original_x, original_y, original_r]
            best_sum_here = original_r
            
            # Try small perturbations
            for dx in [-0.001, -0.0005, 0, 0.0005, 0.001]:
                for dy in [-0.001, -0.0005, 0, 0.0005, 0.001]:
                    for dr in [-0.001, -0.0005, 0, 0.0005, 0.001]:
                        new_x = max(0.001, min(0.999, original_x + dx))
                        new_y = max(0.001, min(0.999, original_y + dy))
                        new_r = max(0.001, min(0.499, original_r + dr))
                        
                        # Check if move is valid (no overlaps with others)
                        valid = True
                        for j in range(n):
                            if i != j:
                                xj, yj, rj = refined[j]
                                dist_sq = (new_x - xj)**2 + (new_y - yj)**2
                                min_dist_sq = (new_r + rj)**2
                                if dist_sq < min_dist_sq:
                                    valid = False
                                    break
                        
                        if valid:
                            # Check containment
                            if new_x >= new_r and new_x <= 1-new_r and new_y >= new_r and new_y <= 1-new_r:
                                # Accept better solution
                                new_sum = best_sum_here + dr
                                if new_sum > best_sum_here:
                                    best_sum_here = new_sum
                                    best_move = [new_x, new_y, new_r]
                                    improved = True
            
            # Update if improvement found
            if improved:
                refined[i] = best_move
                best_sum = np.sum(refined[:, 2])
    
    return refined


# EVOLVE-BLOCK-END
