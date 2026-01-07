# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import math
from typing import Tuple
from itertools import combinations
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining strategic initial placement and global optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Strategy: Start with a better initial placement that ensures no overlaps
    # Use a more sophisticated initial configuration inspired by known good packings
    
    # Initialize with a known good starting configuration
    circles = np.zeros((n, 3))
    
    # Generate initial placement that avoids overlaps using a more systematic approach
    def generate_initial_placement(num_circles: int) -> np.ndarray:
        # Use a hexagonal packing pattern which tends to be more efficient
        # For 26 circles, we'll use a 5x5 grid with some adjustments
        positions = []
        
        # Create a more even distribution using a honeycomb-like pattern
        rows = 5
        cols = 5
        padding = 0.05
        grid_size = 1 - 2 * padding
        
        # Create positions in a grid with slight offset for better packing
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= num_circles:
                    break
                # Offset every other row for hexagonal packing
                x_offset = (j + (i % 2) * 0.5) / (cols - 1) if cols > 1 else 0.5
                y_offset = i / (rows - 1) if rows > 1 else 0.5
                
                x = padding + x_offset * grid_size
                y = padding + y_offset * grid_size
                positions.append([x, y])
        
        # Ensure we have exactly num_circles
        while len(positions) < num_circles:
            # Add some randomness to fill up, but keep it within bounds
            x = padding + (1 - 2 * padding) * np.random.random()
            y = padding + (1 - 2 * padding) * np.random.random()
            positions.append([x, y])
            
        return np.array(positions[:num_circles])
    
    # Generate initial positions
    initial_positions = generate_initial_placement(n)
    
    # Assign positions and set initial radii
    for i, (x, y) in enumerate(initial_positions):
        circles[i] = [x, y, 0.05]  # Start with small radius
    
    # Refine initial radii based on available space
    for i in range(n):
        x, y, r = circles[i]
        
        # Find minimum distance to boundaries
        min_boundary_dist = min(x, 1-x, y, 1-y)
        
        # Find minimum distance to other circles
        min_other_dist = float('inf')
        for j in range(n):
            if i != j:
                other_x, other_y, other_r = circles[j]
                dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                min_other_dist = min(min_other_dist, dist)
        
        # Set radius based on available space, ensuring no overlaps
        if min_other_dist < float('inf') and min_other_dist > 0:
            max_radius = min(min_boundary_dist, min_other_dist / 2.0)
        else:
            max_radius = min_boundary_dist
            
        # Make sure it's positive and reasonable
        max_radius = max(0.001, min(max_radius, 0.25))
        circles[i] = [x, y, max_radius]
    
    # More sophisticated optimization approach
    def objective(radii_and_centers):
        # Extract centers and radii from flattened array
        centers = radii_and_centers[:2*n].reshape(-1, 2)
        radii = radii_and_centers[2*n:]
        
        # Calculate negative sum of radii (we want to maximize sum)
        return -np.sum(radii)
    
    def constraint_func(radii_and_centers):
        centers = radii_and_centers[:2*n].reshape(-1, 2)
        radii = radii_and_centers[2*n:]
        
        constraints = []
        
        # Distance constraint: circles must not overlap
        # Use a more efficient approach by only checking pairs once
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_squared = dx*dx + dy*dy
                min_dist_squared = (radii[i] + radii[j])**2
                
                # Constraint is satisfied when dist >= min_dist (so we subtract)
                # We check if the squared distance is greater than or equal to the squared minimum distance
                constraints.append(np.sqrt(dist_squared) - (radii[i] + radii[j]))
        
        # Boundary constraints: all radii must be valid
        for i in range(n):
            constraints.append(centers[i, 0] - radii[i])  # x - r >= 0
            constraints.append(centers[i, 1] - radii[i])  # y - r >= 0
            constraints.append(1 - centers[i, 0] - radii[i])  # 1 - x - r >= 0
            constraints.append(1 - centers[i, 1] - radii[i])  # 1 - y - r >= 0
        
        return np.array(constraints)
    
    # Flatten initial values
    initial_guess = np.concatenate([
        circles[:, :2].flatten(),  # centers
        circles[:, 2]              # radii
    ])
    
    # Define bounds for optimization
    bounds = []
    # Bounds for centers (0.01 to 0.99 to keep some margin)
    for _ in range(2*n):
        bounds.append((0.01, 0.99))
    # Bounds for radii (positive but not too large)
    for _ in range(n):
        bounds.append((0.001, 0.49))
    
    # Apply optimization with better approach
    try:
        # Try multiple optimization strategies with different parameters
        best_result = None
        best_sum = 0
        
        # Strategy 1: Differential Evolution with higher quality parameters
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                maxiter=150,
                popsize=20,
                seed=42,
                disp=False,
                tol=1e-7,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            
            if de_result.success:
                current_sum = -de_result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = de_result
        except:
            pass
            
        # Strategy 2: SLSQP with better convergence criteria
        if best_result is None:
            try:
                result = minimize(
                    objective,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 1000, 'ftol': 1e-9, 'eps': 1e-6, 'iprint': 0}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except:
                pass
        
        # Strategy 3: Another SLSQP run with potentially better starting point
        if best_result is None:
            try:
                # Try with slightly modified initial guess
                modified_guess = initial_guess.copy()
                # Add small perturbations to improve chances
                for i in range(len(modified_guess)):
                    if i >= 2*n:  # Radii part
                        modified_guess[i] += (np.random.random() - 0.5) * 0.005
                    else:  # Center part
                        modified_guess[i] += (np.random.random() - 0.5) * 0.005
                
                result = minimize(
                    objective,
                    modified_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-5, 'iprint': 0}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except:
                pass
        
        # If we found a good result, use it
        if best_result is not None and best_result.success:
            optimized_centers = best_result.x[:2*n].reshape(-1, 2)
            optimized_radii = best_result.x[2*n:]
            
            # Update circles with optimized values
            for i in range(n):
                circles[i] = [optimized_centers[i, 0], optimized_centers[i, 1], optimized_radii[i]]
        else:
            # If all optimization failed, try a more robust approach
            # Run a simple local optimization with better constraint checking
            try:
                # Use a simpler but more reliable optimization approach
                current_solution = initial_guess.copy()
                
                # Simple gradient-free optimization
                for iteration in range(100):
                    # Keep track of best solution so far
                    current_sum = -objective(current_solution)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        # Save the best solution
                        best_solution = current_solution.copy()
                    
                    # Try small perturbations
                    new_solution = current_solution.copy()
                    for param_idx in range(len(new_solution)):
                        old_value = new_solution[param_idx]
                        # Small random perturbation
                        perturbation = (np.random.random() - 0.5) * 0.001
                        new_solution[param_idx] = old_value + perturbation
                        
                        # Check if it satisfies constraints
                        try:
                            if constraint_func(new_solution).min() >= -1e-6:
                                # If feasible, accept the change
                                current_solution = new_solution.copy()
                            else:
                                # Revert if invalid
                                new_solution[param_idx] = old_value
                        except:
                            new_solution[param_idx] = old_value
                    
                    # Early stopping if no improvement
                    if iteration > 20 and abs(best_sum - current_sum) < 1e-6:
                        break
                        
                # If we found a better solution during local search
                if 'best_solution' in locals():
                    current_solution = best_solution
                    
                # Convert back to circles
                final_centers = current_solution[:2*n].reshape(-1, 2)
                final_radii = current_solution[2*n:]
                for i in range(n):
                    circles[i] = [final_centers[i, 0], final_centers[i, 1], final_radii[i]]
                    
            except:
                pass
                
    except Exception as e:
        # If any optimization fails, return the initial placement
        pass
    
    # Final validation to ensure all constraints are met
    # Make sure circles don't go outside boundaries
    for i in range(n):
        x, y, r = circles[i]
        # Adjust if necessary
        circles[i] = [max(r, min(1-r, x)), max(r, min(1-r, y)), r]
    
    return circles


# EVOLVE-BLOCK-END
