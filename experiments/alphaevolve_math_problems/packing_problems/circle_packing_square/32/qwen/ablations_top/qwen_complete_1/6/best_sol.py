# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with nonlinear optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Better hexagonal initialization inspired by INSPIRATION 2
    def initialize_better_hexagonal():
        circles = np.zeros((n, 3))
        
        # Use a more systematic hexagonal arrangement
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        if rows * cols < n:
            rows += 1
            
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Adjust spacing for better packing
        radius_guess = min(spacing_x, spacing_y) * 0.4
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                # Ensure we're within bounds
                x = max(radius_guess, min(1-radius_guess, x))
                y = max(radius_guess, min(1-radius_guess, y))
                
                circles[idx] = [x, y, radius_guess]
                idx += 1
            if idx >= n:
                break
                
        return circles
    
    # Objective function - maximize sum of radii
    def objective(circles_flat):
        # Return negative because we want to maximize
        return -np.sum(circles_flat[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Constraint functions with proper closures (inspired by INSPIRATION 2)
    def create_containment_constraints():
        constraints = []
        for i in range(n):
            # x >= r
            def x_min(c, i=i):
                return c[3*i] - c[3*i+2]
            constraints.append({'type': 'ineq', 'fun': x_min})
            
            # x <= 1-r
            def x_max(c, i=i):
                return 1 - c[3*i] - c[3*i+2]
            constraints.append({'type': 'ineq', 'fun': x_max})
            
            # y >= r
            def y_min(c, i=i):
                return c[3*i+1] - c[3*i+2]
            constraints.append({'type': 'ineq', 'fun': y_min})
            
            # y <= 1-r
            def y_max(c, i=i):
                return 1 - c[3*i+1] - c[3*i+2]
            constraints.append({'type': 'ineq', 'fun': y_max})
        return constraints
    
    def create_nonoverlap_constraints():
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                # Distance between centers must be >= sum of radii
                def overlap_constraint(c, i=i, j=j):
                    x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                    x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                    dist_sq = (x1-x2)**2 + (y1-y2)**2
                    return dist_sq - (r1+r2)**2
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        return constraints
    
    # Generate initial configuration
    initial_circles = initialize_better_hexagonal()
    initial_flat = initial_circles.flatten()
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # x in [r, 1-r], y in [r, 1-r], r in [0.001, 0.5]
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.499))  # r
    
    # Create constraints
    constraints = []
    constraints.extend(create_containment_constraints())
    constraints.extend(create_nonoverlap_constraints())
    
    # Run optimization with multiple restarts for better results (inspired by INSPIRATION 2)
    best_result = None
    best_sum = -np.inf
    
    # Try multiple random restarts with enhanced strategy
    for restart in range(5):  # Increased from 3 to 5 restarts
        # Use fixed seed for reproducibility
        np.random.seed(restart * 1000 + 42)
        
        # Perturb the initial solution with different perturbation strengths
        perturbed_initial = initial_flat.copy()
        for i in range(len(perturbed_initial)):
            if i % 3 in [0, 1]:  # x and y coordinates
                # Add larger perturbation for better exploration
                perturbed_initial[i] += np.random.normal(0, 0.025 if restart < 2 else 0.01)
                perturbed_initial[i] = np.clip(perturbed_initial[i], 0.001, 0.999)
            elif i % 3 == 2:  # radius
                # Add smaller perturbation for radius
                perturbed_initial[i] += np.random.normal(0, 0.015 if restart < 2 else 0.005)
                perturbed_initial[i] = np.clip(perturbed_initial[i], 0.001, 0.499)
        
        try:
            result = minimize(
                objective,
                perturbed_initial,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6},
                tol=1e-6
            )
            
            if result.success:
                # Calculate sum of radii for this result
                total_radius = -objective(result.x)  # Negative because objective returns negative sum
                if total_radius > best_sum:
                    best_sum = total_radius
                    best_result = result
        except Exception:
            continue
    
    # Use best result or fallback to initial
    if best_result is not None and best_result.success:
        final_flat = best_result.x
    else:
        final_flat = initial_flat
    
    # Convert back to circles array
    final_circles = np.zeros((n, 3))
    for i in range(n):
        final_circles[i] = [final_flat[3*i], final_flat[3*i+1], final_flat[3*i+2]]
    
    # Final refinement with local search to improve radii (inspired by INSPIRATION 2)
    def compute_distance(i, j, circles):
        """Compute distance between circle centers"""
        x1, y1 = circles[i, 0], circles[i, 1]
        x2, y2 = circles[j, 0], circles[j, 1]
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def compute_max_radius(circles, i):
        """Compute maximum possible radius for circle i without violating constraints"""
        x, y = circles[i, 0], circles[i, 1]
        # Maximum radius based on boundaries
        max_r = min(x, 1-x, y, 1-y)
        
        # Check non-overlap constraints with all other circles
        for j in range(n):
            if i != j:
                dist = compute_distance(i, j, circles)
                max_r = min(max_r, dist - circles[j, 2])
        
        return max_r
    
    # Iteratively improve by trying to increase radii (increased iterations)
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try to increase each radius
        for i in range(n):
            current_radius = final_circles[i, 2]
            max_possible_radius = compute_max_radius(final_circles, i)
            
            # Only try to increase if beneficial
            if max_possible_radius > current_radius + 1e-6:
                # Try to set radius to max possible (but keep within bounds)
                new_radius = min(max_possible_radius, 0.499)
                
                # Make sure this doesn't violate any constraints
                valid = True
                for j in range(n):
                    if i != j:
                        dist = compute_distance(i, j, final_circles)
                        if dist < (new_radius + final_circles[j, 2]):
                            valid = False
                            break
                
                if valid:
                    final_circles[i, 2] = new_radius
                    improved = True
    
    # Final constraint enforcement
    for i in range(n):
        x, y, r = final_circles[i]
        # Ensure circles stay within bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        r = np.clip(r, 0.001, min(x, 1-x, y, 1-y))
        final_circles[i] = [x, y, r]
    
    return final_circles


# EVOLVE-BLOCK-END
