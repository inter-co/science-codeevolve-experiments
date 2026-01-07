# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies with local optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 26
    best_sum = 0
    best_circles = None
    
    # Try multiple random initializations to avoid local optima
    num_starts = 20
    
    for start_idx in range(num_starts):
        # Use different initialization strategies based on start_idx
        circles = np.zeros((n, 3))
        
        # Strategy 1: Hexagonal grid initialization (like INSPIRATION 2)
        if start_idx < 12:
            # Create a hexagonal packing pattern that fits in the unit square
            rows = 5
            cols = 5
            row_spacing = 0.18
            col_spacing = 0.18 * math.sqrt(3)/2
            
            idx = 0
            for row in range(rows):
                for col in range(cols):
                    if idx >= n:
                        break
                    # Offset odd rows for hexagonal packing
                    x_offset = col * col_spacing
                    if row % 2 == 1:
                        x_offset += col_spacing / 2
                    
                    # Ensure we stay within bounds
                    x = x_offset + 0.05
                    y = row * row_spacing + 0.05
                    
                    # Make sure it's within the unit square
                    if x <= 0.95 and y <= 0.95:
                        # Initial radius - small enough to allow optimization
                        r = min(0.05, 0.5 * col_spacing, 0.5 * row_spacing)
                        circles[idx] = [x, y, r]
                        idx += 1
                if idx >= n:
                    break
            
            # Fill remaining positions with small circles near center
            for i in range(idx, n):
                # Add slight randomness to avoid perfect symmetry
                noise = random.uniform(-0.01, 0.01)
                circles[i] = [0.5 + noise, 0.5 + noise, 0.01]
        
        # Strategy 2: Grid-based with random perturbation
        elif start_idx < 16:
            # Create a more uniform grid pattern with some randomness
            grid_size = int(math.ceil(math.sqrt(n)))
            spacing = 1.0 / (grid_size + 1)
            
            # Place circles in a grid pattern with slight randomness
            count = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if count >= n:
                        break
                    x = (j + 0.5) * spacing + random.uniform(-spacing/4, spacing/4)
                    y = (i + 0.5) * spacing + random.uniform(-spacing/4, spacing/4)
                    # Ensure within bounds
                    x = max(0.05, min(0.95, x))
                    y = max(0.05, min(0.95, y))
                    circles[count] = [x, y, 0.05]
                    count += 1
                    
            # Fill remaining with random points
            while count < n:
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                circles[count] = [x, y, 0.05]
                count += 1
        
        # Strategy 3: Random initialization with better distribution
        else:
            # Generate random points with some clustering avoidance
            np.random.seed(start_idx)  # Different seed for each start
            
            # Generate points with more even distribution
            for i in range(n):
                # Distribute points more evenly using a combination of random and structured approach
                if i < n // 2:
                    # First half: structured approach
                    x = 0.1 + 0.8 * (i % 5) / 4.0 + random.uniform(-0.05, 0.05)
                    y = 0.1 + 0.8 * (i // 5) / 4.0 + random.uniform(-0.05, 0.05)
                else:
                    # Second half: pure random
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                circles[i] = [x, y, 0.05]
        
        # Define objective function (negative because we minimize for maximization)
        def objective(x):
            return -np.sum(x[2::3])  # Sum of all radii (negated for minimization)
        
        # Define constraint functions
        def boundary_constraint(x):
            """Ensure all circles stay within unit square boundaries"""
            constraints = []
            for i in range(n):
                xi, yi, ri = x[3*i:3*i+3]
                # Circle must be fully inside square with radius ri
                constraints.extend([
                    xi - ri,      # x >= r
                    yi - ri,      # y >= r  
                    1 - xi - ri,  # 1-x >= r
                    1 - yi - ri   # 1-y >= r
                ])
            return np.array(constraints)
        
        def overlap_constraint(x):
            """Ensure no two circles overlap"""
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    xi, yi, ri = x[3*i:3*i+3]
                    xj, yj, rj = x[3*j:3*j+3]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    # Distance squared must be >= (ri + rj)^2 for no overlap
                    constraints.append(dist_sq - (ri + rj)**2)
            return np.array(constraints)
        
        # Set up bounds for optimization (x, y, r) for each circle
        bounds = []
        for i in range(n):
            # Bounds: x in [0.001, 0.999], y in [0.001, 0.999], r in [0.001, 0.499]
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Set up constraints for scipy.optimize
        constraints = [
            {'type': 'ineq', 'fun': boundary_constraint},
            {'type': 'ineq', 'fun': overlap_constraint}
        ]
        
        # Flatten initial guess
        x0 = circles.flatten()
        
        # Optimize using SLSQP method with more aggressive settings
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False},
                callback=None
            )
            
            if result.success:
                circles_opt = result.x.reshape(-1, 3)
                current_sum = np.sum(circles_opt[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles_opt.copy()
            else:
                # Even if optimization fails, keep the initial configuration if it's better
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
                    
        except Exception as e:
            # If optimization fails due to numerical issues, keep the initial configuration
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
    
    # Final refinement with local optimization if we have a good solution
    if best_circles is not None:
        # Apply a few rounds of local optimization to fine-tune
        for _ in range(3):
            # Create a refined version with small adjustments
            refined_circles = best_circles.copy()
            for i in range(n):
                # Try small adjustments to positions and radii
                x, y, r = refined_circles[i]
                # Slightly adjust position and radius
                refined_circles[i] = [x + random.uniform(-0.005, 0.005), 
                                    y + random.uniform(-0.005, 0.005),
                                    max(0.001, min(0.499, r + random.uniform(-0.002, 0.002)))]
            
            # Re-optimize the refined solution
            try:
                # Same optimization process but starting from refined solution
                def objective_refined(x):
                    return -np.sum(x[2::3])
                
                def boundary_constraint_refined(x):
                    constraints = []
                    for i in range(n):
                        xi, yi, ri = x[3*i:3*i+3]
                        constraints.extend([
                            xi - ri,
                            yi - ri,
                            1 - xi - ri,
                            1 - yi - ri
                        ])
                    return np.array(constraints)
                
                def overlap_constraint_refined(x):
                    constraints = []
                    for i in range(n):
                        for j in range(i+1, n):
                            xi, yi, ri = x[3*i:3*i+3]
                            xj, yj, rj = x[3*j:3*j+3]
                            dist_sq = (xi - xj)**2 + (yi - yj)**2
                            constraints.append(dist_sq - (ri + rj)**2)
                    return np.array(constraints)
                
                bounds_refined = []
                for i in range(n):
                    bounds_refined.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                
                constraints_refined = [
                    {'type': 'ineq', 'fun': boundary_constraint_refined},
                    {'type': 'ineq', 'fun': overlap_constraint_refined}
                ]
                
                result = minimize(
                    objective_refined,
                    refined_circles.flatten(),
                    method='SLSQP',
                    bounds=bounds_refined,
                    constraints=constraints_refined,
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
                )
                
                if result.success:
                    circles_opt = result.x.reshape(-1, 3)
                    current_sum = np.sum(circles_opt[:, 2])
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_circles = circles_opt.copy()
                        
            except:
                pass  # Continue with current best if refinement fails
    
    # Return the best solution found
    if best_circles is not None:
        return best_circles
    else:
        # Fallback to a standard hexagonal configuration
        circles = np.zeros((n, 3))
        rows = 5
        cols = 5
        row_spacing = 0.18
        col_spacing = 0.18 * math.sqrt(3)/2
        
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                x_offset = col * col_spacing
                if row % 2 == 1:
                    x_offset += col_spacing / 2
                
                x = x_offset + 0.05
                y = row * row_spacing + 0.05
                
                if x <= 0.95 and y <= 0.95:
                    r = min(0.05, 0.5 * col_spacing, 0.5 * row_spacing)
                    circles[idx] = [x, y, r]
                    idx += 1
            if idx >= n:
                break
        
        for i in range(idx, n):
            circles[i] = [0.5, 0.5, 0.01]
            
        return circles


# EVOLVE-BLOCK-END
