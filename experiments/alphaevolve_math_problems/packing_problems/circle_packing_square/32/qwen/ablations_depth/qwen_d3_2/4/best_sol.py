# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses mathematical optimization with proper constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)
    
    # Better initialization using a more strategic approach
    def initialize_better():
        circles = np.zeros((n, 3))
        
        # Use a hexagonal lattice pattern which typically gives better results
        # For 32 circles, we can arrange in roughly a 6x6 grid with some adjustment
        
        # Create a more sophisticated initial layout
        rows = 6
        cols = 6
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for hexagonal packing
                offset = 0.5 * (i % 2)
                x = 0.05 + (j + offset) * spacing_x + np.random.uniform(-0.005, 0.005)
                y = 0.05 + i * spacing_y + np.random.uniform(-0.005, 0.005)
                r = 0.03  # Slightly larger initial radius
                
                # Ensure within bounds
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
        
        return circles
    
    # Create initial configuration
    circles = initialize_better()
    
    # Flatten parameters: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
    def compute_objective(params):
        """Compute negative sum of radii (since we minimize)"""
        circles_flat = params.reshape((n, 3))
        return -np.sum(circles_flat[:, 2])
    
    def compute_constraints(params):
        """Check constraint violations"""
        circles_flat = params.reshape((n, 3))
        positions = circles_flat[:, :2]
        radii = circles_flat[:, 2]
        
        violations = []
        
        # Boundary constraints (each circle must fit completely in unit square)
        for i in range(n):
            x, y, r = positions[i, 0], positions[i, 1], radii[i]
            violations.append(x - r)  # x >= r
            violations.append(1 - x - r)  # 1-x >= r
            violations.append(y - r)  # y >= r
            violations.append(1 - y - r)  # 1-y >= r
        
        # Non-overlap constraints
        for i, j in combinations(range(n), 2):
            x1, y1, r1 = positions[i, 0], positions[i, 1], radii[i]
            x2, y2, r2 = positions[j, 0], positions[j, 1], radii[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            violations.append(distance - (r1 + r2))
            
        return np.array(violations)
    
    # Prepare bounds: (x_min, x_max), (y_min, y_max), (r_min, r_max)
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.49)] * n
    
    # Multi-start optimization to avoid local optima
    best_sum = 0
    best_params = None
    
    # Try multiple random starts to find better solutions
    for start_iter in range(3):
        # Create a slightly different initial configuration for each start
        if start_iter == 0:
            # Use the initialized configuration
            current_params = circles.flatten()
        else:
            # Perturb the current configuration slightly
            current_params = circles.flatten() + np.random.normal(0, 0.01, len(circles.flatten()))
            # Ensure valid bounds
            current_params = np.clip(current_params, 0.001, 0.999)
        
        try:
            # Run optimization
            result = minimize(
                compute_objective,
                current_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-6, 'gtol': 1e-6},
                constraints=[
                    {'type': 'ineq', 'fun': lambda p, i=i: p[i*3] - p[i*3+2]} for i in range(n)  # x >= r
                ] + [
                    {'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i*3] - p[i*3+2]} for i in range(n)  # 1-x >= r
                ] + [
                    {'type': 'ineq', 'fun': lambda p, i=i: p[i*3+1] - p[i*3+2]} for i in range(n)  # y >= r
                ] + [
                    {'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i*3+1] - p[i*3+2]} for i in range(n)  # 1-y >= r
                ] + [
                    {'type': 'ineq', 'fun': lambda p, i=i, j=j: np.sqrt((p[i*3]-p[j*3])**2 + (p[i*3+1]-p[j*3+1])**2) - (p[i*3+2] + p[j*3+2])} 
                    for i in range(n) for j in range(i+1, n)
                ]
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_params = result.x.copy()
                    
        except Exception as e:
            continue
    
    # If no successful optimization found, fall back to the initial configuration
    if best_params is None:
        best_params = circles.flatten()
    
    # Convert back to circles format and finalize
    circles_final = best_params.reshape((n, 3))
    
    # Post-processing: ensure all constraints are met and fix any remaining issues
    for i in range(n):
        r = circles_final[i, 2]
        circles_final[i, 0] = np.clip(circles_final[i, 0], r, 1-r)
        circles_final[i, 1] = np.clip(circles_final[i, 1], r, 1-r)
    
    # Final validation and correction
    max_iter = 50
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
                    dx = (x2 - x1) / distance * overlap * 0.5 if distance > 0.001 else 0.001
                    dy = (y2 - y1) / distance * overlap * 0.5 if distance > 0.001 else 0.001
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
    
    # Final check and ensure all circles are within bounds
    for i in range(n):
        r = circles_final[i, 2]
        circles_final[i, 0] = np.clip(circles_final[i, 0], r, 1-r)
        circles_final[i, 1] = np.clip(circles_final[i, 1], r, 1-r)
    
    return circles_final


# EVOLVE-BLOCK-END
