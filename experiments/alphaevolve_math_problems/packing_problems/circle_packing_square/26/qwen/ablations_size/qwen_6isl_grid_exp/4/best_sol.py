# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import time
import random

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies and optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores 
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 26
    
    # Multi-strategy initialization
    def initialize_hexagonal():
        """Initialize using hexagonal packing pattern"""
        circles = []
        rows = 5
        cols = 6
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5 * (i % 2)) / cols
                y = (i + 0.5) / rows
                # Keep within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                r = 0.08  # Initial radius
                circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    def initialize_grid():
        """Initialize using grid pattern"""
        circles = []
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                # Keep within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                r = 0.06  # Initial radius
                circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    def initialize_random():
        """Initialize using random placement with careful spacing"""
        circles = []
        for i in range(n):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.03, 0.08)
            circles.append([x, y, r])
        
        return np.array(circles)
    
    # Try multiple initialization strategies
    initial_strategies = [initialize_hexagonal, initialize_grid, initialize_random]
    best_solution = None
    best_sum = -np.inf
    
    for strategy in initial_strategies:
        try:
            circles = strategy()
            
            # Project to feasible region
            for i in range(n):
                x, y, r = circles[i]
                # Ensure radius fits within bounds
                r = min(r, x, y, 1-x, 1-y)
                r = max(r, 0.001)
                circles[i] = [x, y, r]
            
            # Refine using optimization
            refined_circles = optimize_solution(circles)
            
            current_sum = np.sum(refined_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution = refined_circles.copy()
                
        except Exception as e:
            continue
    
    # If no good solution found, use fallback
    if best_solution is None:
        circles = initialize_hexagonal()
        best_solution = optimize_solution(circles)
    
    return best_solution

def optimize_solution(initial_circles):
    """Optimize the initial solution using scipy with proper constraints"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    x0 = initial_circles.flatten()
    
    # Objective: maximize sum of radii (minimize negative sum)
    def objective(x):
        radii = x[2::3]  # Every third element starting from index 2
        return -np.sum(radii)
    
    # Constraints
    def constraint_func(x):
        circles_flat = x.reshape(-1, 3)
        positions = circles_flat[:, :2]
        radii = circles_flat[:, 2]
        
        constraints = []
        
        # Boundary constraints (each circle must fit completely in unit square)
        for i in range(n):
            x_pos, y_pos = positions[i]
            r = radii[i]
            # x >= r, 1-x >= r, y >= r, 1-y >= r
            constraints.extend([
                x_pos - r,           # x >= r
                1 - x_pos - r,       # 1-x >= r  
                y_pos - r,           # y >= r
                1 - y_pos - r        # 1-y >= r
            ])
        
        # Non-overlap constraints
        tree = cKDTree(positions)
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = np.sqrt(dx*dx + dy*dy)
                min_distance = radii[i] + radii[j]
                constraints.append(distance - min_distance)  # Distance >= sum of radii
        
        return np.array(constraints)
    
    # Bounds for x, y, r for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Constraints dictionary
    constraints = {'type': 'ineq', 'fun': constraint_func}
    
    try:
        # Use SLSQP optimizer with more iterations
        result = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure feasibility after optimization
            return project_to_feasible(optimized_circles)
    except Exception as e:
        pass
    
    # Fallback to initial solution
    return initial_circles

def project_to_feasible(circles):
    """Project circles to ensure all constraints are satisfied"""
    # Ensure containment constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Clamp radius to fit within bounds
        r = min(r, x, y, 1-x, 1-y)
        r = max(r, 0.001)
        circles[i] = [x, y, r]
    
    # Resolve overlaps iteratively
    tree = cKDTree(circles[:, :2])
    max_iter = 50
    
    for _ in range(max_iter):
        # Find overlapping pairs
        pairs = tree.query_pairs(2.0 * np.max(circles[:, 2]), output_type='ndarray')
        if len(pairs) == 0:
            break
            
        # Resolve overlaps by reducing radii
        resolved = set()
        for i, j in pairs:
            if i in resolved or j in resolved:
                continue
                
            r1, r2 = circles[i, 2], circles[j, 2]
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < (r1 + r2) and distance > 0:
                overlap = (r1 + r2) - distance
                reduction = min(overlap * 0.5, r1 * 0.3, r2 * 0.3)
                
                if reduction > 0:
                    r1_new = max(0.001, r1 - reduction)
                    r2_new = max(0.001, r2 - reduction)
                    circles[i, 2] = r1_new
                    circles[j, 2] = r2_new
                    resolved.add(i)
                    resolved.add(j)
        
        # Rebuild tree for next iteration
        tree = cKDTree(circles[:, :2])
    
    return circles


# EVOLVE-BLOCK-END
