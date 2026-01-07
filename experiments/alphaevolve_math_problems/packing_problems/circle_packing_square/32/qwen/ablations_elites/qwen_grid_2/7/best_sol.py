# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with mathematical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)
    n = 32
    
    # Generate high-quality hexagonal initial configuration inspired by INSPIRATION 2
    def generate_hexagonal_initial():
        # Create a more systematic hexagonal grid
        rows = 6
        cols = 6
        
        # Calculate spacing to fit well in unit square with padding
        padding = 0.1
        width = 1 - 2 * padding
        height = 1 - 2 * padding
        
        # Determine spacing to fit within bounds
        spacing_x = width / (cols - 1) if cols > 1 else 0.2
        spacing_y = height / (rows - 1) if rows > 1 else 0.2
        
        # Hexagonal packing spacing
        hex_spacing_y = spacing_y * np.sqrt(3) / 2
        
        circles = []
        count = 0
        
        # Create hexagonal grid with offset rows
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Hexagonal offset for odd rows
                x_offset = (i % 2) * spacing_x * 0.5
                x = padding + x_offset + j * spacing_x
                y = padding + i * hex_spacing_y
                
                # Ensure within bounds
                if x <= 1-padding and y <= 1-padding:
                    # Initial radius based on distance to edges
                    max_radius = min(x-padding, 1-padding-x, y-padding, 1-padding-y)
                    r = min(0.1, max_radius * 0.8)  # Use smaller radius to allow room for optimization
                    circles.append([x, y, r])
                    count += 1
            if count >= n:
                break
        
        # Fill remaining positions with strategic random placements
        for i in range(count, n):
            attempts = 0
            while attempts < 100:
                # Place near center with some randomness
                x = random.uniform(0.3, 0.7)
                y = random.uniform(0.3, 0.7)
                # Radius based on distance to edges and nearby circles
                max_radius = min(x, 1-x, y, 1-y)
                r = min(0.1, max_radius * 0.7)
                
                # Check if this position is valid with existing circles
                valid = True
                for k in range(i):
                    cx, cy, cr = circles[k]
                    dist_sq = (x - cx)**2 + (y - cy)**2
                    if dist_sq < (r + cr)**2:
                        valid = False
                        break
                
                if valid:
                    circles.append([x, y, r])
                    break
                attempts += 1
        
        return np.array(circles)
    
    # Generate initial configuration
    circles = generate_hexagonal_initial()
    
    # Objective function (negative because we're minimizing)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Sum of radii (negated for minimization)
    
    # Constraint functions for optimization - more robust than INSPIRATION 1
    def contain_constraints(circles_flat):
        """Ensure all circles are contained within the unit square"""
        n = len(circles_flat) // 3
        constraints = []
        
        for i in range(n):
            # radius must be positive (we use 1e-6 instead of 0 to avoid numerical issues)
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2+i*3] - 1e-6})
            # x coordinate constraints (slightly relaxed to prevent numerical issues)
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[0+i*3] - x[2+i*3] - 1e-6})  # x >= r + epsilon
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[0+i*3] - x[2+i*3] - 1e-6})  # 1-x >= r + epsilon
            # y coordinate constraints
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[1+i*3] - x[2+i*3] - 1e-6})  # y >= r + epsilon
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[1+i*3] - x[2+i*3] - 1e-6})  # 1-y >= r + epsilon
            
        return constraints
    
    def non_overlap_constraints(circles_flat):
        """Ensure no two circles overlap"""
        n = len(circles_flat) // 3
        constraints = []
        
        # Check all pairs of circles for overlap
        for i in range(n):
            for j in range(i+1, n):
                # Distance between centers must be >= sum of radii
                def constraint_func(x, i=i, j=j):
                    xi, yi, ri = x[i*3], x[i*3+1], x[i*3+2]
                    xj, yj, rj = x[j*3], x[j*3+1], x[j*3+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    # We want sqrt(dist_sq) >= ri + rj, so dist_sq >= (ri + rj)^2
                    return dist_sq - (ri + rj)**2
                    
                constraints.append({'type': 'ineq', 'fun': constraint_func})
                
        return constraints
    
    # Flatten the circles array for optimization
    initial_flat = circles.flatten()
    
    # Bounds for variables (x, y, r) - ensure radii are positive and positions are valid
    bounds = []
    for i in range(n):
        # x coordinate bounds (slightly restricted to ensure proper containment)
        bounds.append((1e-6, 1-1e-6))  # x in (0,1)
        # y coordinate bounds  
        bounds.append((1e-6, 1-1e-6))  # y in (0,1)
        # radius bounds (more conservative upper bound)
        bounds.append((1e-6, 0.499))   # r in (0,0.499) - slightly less than 0.5
    
    # Create constraints once
    cons = []
    # Add containment constraints
    for constraint in contain_constraints(initial_flat):
        cons.append(constraint)
    # Add non-overlap constraints
    for constraint in non_overlap_constraints(initial_flat):
        cons.append(constraint)
    
    # Multi-start optimization with different strategies
    best_circles = circles.copy()
    best_radius_sum = np.sum(best_circles[:, 2])
    
    # Strategy 1: Direct optimization from initial state
    try:
        result = minimize(objective, initial_flat, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if result.success:
            optimized_flat = result.x
            optimized_circles = optimized_flat.reshape(-1, 3)
            new_radius_sum = np.sum(optimized_circles[:, 2])
            if new_radius_sum > best_radius_sum:
                best_circles = optimized_circles
                best_radius_sum = new_radius_sum
    except Exception as e:
        pass
    
    # Strategy 2: Several restarts with different perturbations
    for restart in range(3):
        # Create a perturbed version of the current best
        perturbed = best_circles.copy()
        
        # Apply different perturbation strengths
        perturbation_strength = 0.02 + restart * 0.01  # Gradually decrease strength
        
        for i in range(n):
            if random.random() < 0.25:  # 25% chance to perturb each circle
                # Add more substantial perturbations for early restarts
                perturbed[i, 0] += random.uniform(-perturbation_strength, perturbation_strength)
                perturbed[i, 1] += random.uniform(-perturbation_strength, perturbation_strength)
                # Keep within bounds
                perturbed[i, 0] = max(0.01, min(0.99, perturbed[i, 0]))
                perturbed[i, 1] = max(0.01, min(0.99, perturbed[i, 1]))
        
        # Apply optimization to the perturbed version
        perturbed_flat = perturbed.flatten()
        
        try:
            result = minimize(objective, perturbed_flat, method='SLSQP', bounds=bounds, constraints=cons,
                             options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if result.success:
                refined_flat = result.x
                refined = refined_flat.reshape(-1, 3)
                
                # Check if this improved the solution
                new_radius_sum = np.sum(refined[:, 2])
                if new_radius_sum > best_radius_sum:
                    best_circles = refined
                    best_radius_sum = new_radius_sum
        except:
            pass
    
    # Strategy 3: Physics-inspired refinement (like INSPIRATION 1)
    # This adds a post-processing step to improve the final result
    refined_circles = best_circles.copy()
    max_iterations = 500
    
    for iteration in range(max_iterations):
        changed = False
        positions = refined_circles[:, :2]
        radii = refined_circles[:, 2]
        
        # Repulsion forces for overlapping circles (similar to INSPIRATION 1)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = refined_circles[i]
                x2, y2, r2 = refined_circles[j]
                
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < (r1 + r2) and dist > 1e-10:
                    # Repel circles
                    overlap = (r1 + r2) - dist
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                    
                    move_amount = overlap * 0.3  # Smaller movement factor
                    refined_circles[i, 0] -= dx_norm * move_amount
                    refined_circles[i, 1] -= dy_norm * move_amount
                    refined_circles[j, 0] += dx_norm * move_amount
                    refined_circles[j, 1] += dy_norm * move_amount
                    
                    changed = True
        
        # Boundary constraints and radius adjustments
        for i in range(n):
            x, y, r = refined_circles[i]
            
            # Adjust radius to fit within bounds
            max_radius = min(x, 1-x, y, 1-y)
            if r > max_radius:
                r = max_radius
                changed = True
            
            # Ensure minimum radius
            if r < 0.001:
                r = 0.001
                changed = True
            
            refined_circles[i, 2] = r
            
            # Keep positions within bounds
            refined_circles[i, 0] = max(r, min(1-r, x))
            refined_circles[i, 1] = max(r, min(1-r, y))
        
        # Stop early if no changes
        if not changed:
            break
    
    # Final cleanup and validation
    final_circles = refined_circles.copy()
    for i in range(n):
        x, y, r = final_circles[i]
        # Ensure boundary constraints
        final_circles[i, 0] = max(r, min(1-r, x))
        final_circles[i, 1] = max(r, min(1-r, y))
        # Ensure radius constraints
        final_circles[i, 2] = max(0.001, min(0.499, r))
    
    return final_circles


# EVOLVE-BLOCK-END
