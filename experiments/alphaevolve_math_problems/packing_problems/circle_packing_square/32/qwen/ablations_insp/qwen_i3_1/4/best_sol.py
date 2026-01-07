# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and scipy optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = 32
    
    # Initialize using hexagonal packing pattern (inspired by better approach)
    circles = _initialize_hexagonal_packing(n)
    
    # Optimize using scipy minimize with constraints
    circles_opt = _optimize_circles(circles)
    
    return circles_opt

def _initialize_hexagonal_packing(n: int) -> np.ndarray:
    """Initialize circles in a hexagonal grid pattern with better distribution"""
    # Create a hexagonal grid pattern
    rows = int(math.sqrt(n) * 1.2)
    cols = int(n / rows) + 1
    
    # Adjust dimensions to fit in unit square
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Hexagonal offset
    offset = spacing_x * 0.5
    
    circles = []
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
                
            # Alternate row offset for hexagonal pattern
            x = (j + (i % 2) * 0.5) * spacing_x
            y = i * spacing_y
            
            # Ensure we're within bounds
            if x <= 1 and y <= 1:
                # Initial radius - small enough to fit in grid cell but not too small
                r = min(spacing_x, spacing_y) * 0.4
                
                # Ensure circle fits within unit square
                r = min(r, x, 1-x, y, 1-y)
                
                if r > 0:
                    circles.append([x, y, r])
                    count += 1
                    
        if count >= n:
            break
    
    # Fill remaining slots with random positions if needed
    while len(circles) < n:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = np.random.uniform(0.01, 0.1)
        # Ensure circle fits in square
        r = min(r, x, 1-x, y, 1-y)
        if r > 0:
            circles.append([x, y, r])
    
    return np.array(circles)

def _calculate_objective(circles: np.ndarray) -> float:
    """Calculate negative sum of radii (since we want to maximize)"""
    return -np.sum(circles[:, 2])

def _calculate_constraints(circles: np.ndarray) -> dict:
    """Calculate constraint violations efficiently"""
    n = len(circles)
    constraints = []
    
    # Circle containment constraints (each circle must fit within unit square)
    for i in range(n):
        x, y, r = circles[i]
        # Distance to boundaries must be >= radius
        constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[i][0] - c[i][2]})  # x >= r
        constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[i][1] - c[i][2]})  # y >= r
        constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[i][0] - c[i][2]})  # 1-x >= r
        constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[i][1] - c[i][2]})  # 1-y >= r
    
    # Circle overlap constraints (distance between centers >= sum of radii)
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(c, i=i, j=j):
                x1, y1, r1 = c[i]
                x2, y2, r2 = c[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                return distance - (r1 + r2)
            
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    return constraints

def _optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using scipy minimize"""
    n = len(initial_circles)
    
    # Flatten the array for optimization
    initial_flat = initial_circles.flatten()
    
    # Define bounds: [x1,y1,r1,x2,y2,r2,...]
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds  
        bounds.append((0.001, 0.999))
        # r bounds
        bounds.append((0.001, 0.499))  # Reasonable upper bound
    
    def objective(x_flat):
        # Reshape back to circles array
        circles = x_flat.reshape((n, 3))
        return -np.sum(circles[:, 2])  # Negative because we minimize
    
    def constraint_func(x_flat):
        circles = x_flat.reshape((n, 3))
        constraints = []
        
        # Containment constraints
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([
                x - r,          # x >= r
                y - r,          # y >= r  
                1 - x - r,      # 1-x >= r
                1 - y - r       # 1-y >= r
            ])
        
        # Overlap constraints - use KDTree for efficiency
        # First check if we can optimize this part
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(distance - (r1 + r2))
        
        return np.array(constraints)
    
    # Create constraint dictionary
    constraints = [{'type': 'ineq', 'fun': constraint_func}]
    
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape((n, 3))
            # Ensure all circles still satisfy constraints
            return _validate_and_fix_circles(optimized_circles)
        else:
            # If optimization fails, return initial configuration
            return initial_circles
            
    except Exception:
        # If anything goes wrong, return initial configuration
        return initial_circles

def _validate_and_fix_circles(circles: np.ndarray) -> np.ndarray:
    """Ensure all circles satisfy constraints and fix if necessary"""
    n = len(circles)
    
    # Create a copy to work with
    validated = circles.copy()
    
    # Fix containment issues first
    for i in range(n):
        x, y, r = validated[i]
        # Clamp to valid ranges
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        validated[i] = [x, y, r]
    
    # Handle overlaps using a simple iterative approach
    max_iter = 100
    for _ in range(max_iter):
        changed = False
        # Check all pairs for overlap
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = validated[i]
                x2, y2, r2 = validated[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                # If overlapping
                if distance < (r1 + r2):
                    # Reduce radii to prevent overlap
                    total_radius = r1 + r2
                    reduction = (r1 + r2 - distance) * 0.5 + 1e-6
                    if r1 > reduction and r2 > reduction:
                        validated[i][2] -= reduction * 0.5
                        validated[j][2] -= reduction * 0.5
                        changed = True
        
        if not changed:
            break
    
    return validated


# EVOLVE-BLOCK-END
