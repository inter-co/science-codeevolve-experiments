# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    # Better initialization using more systematic approach - inspired by INSPIRATION 1
    def initialize_better():
        circles = []
        
        # Create a more refined hexagonal grid pattern
        rows = 6
        cols = 6
        
        spacing_x = 1.0 / (cols + 1)
        spacing_y = spacing_x * math.sqrt(3) / 2
        radius_estimate = spacing_x * 0.4
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Add slight randomness to avoid perfect grid
                x += random.uniform(-spacing_x*0.1, spacing_x*0.1)
                y += random.uniform(-spacing_y*0.1, spacing_y*0.1)
                circles.append([x, y, radius_estimate])
                idx += 1
        
        # Fill remaining circles with random positions
        for i in range(idx, n):
            circles.append([
                random.uniform(0.05, 0.95),
                random.uniform(0.05, 0.95),
                random.uniform(0.01, 0.05)
            ])
        
        return np.array(circles)
    
    # Optimized constraint checking using vectorization - inspired by INSPIRATION 1
    def constraint_containment(params):
        circles = params.reshape(-1, 3)
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Vectorized constraint evaluation for all boundary constraints
        # Left boundary: x - r >= 0
        left_constraint = positions[:, 0] - radii
        # Right boundary: 1 - x - r >= 0  
        right_constraint = 1 - positions[:, 0] - radii
        # Bottom boundary: y - r >= 0
        bottom_constraint = positions[:, 1] - radii
        # Top boundary: 1 - y - r >= 0
        top_constraint = 1 - positions[:, 1] - radii
        
        return np.concatenate([left_constraint, right_constraint, bottom_constraint, top_constraint])
    
    def constraint_nonoverlap(params):
        circles = params.reshape(-1, 3)
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Vectorized non-overlap constraint checking using cdist for efficiency - inspired by INSPIRATION 1
        distances = cdist(positions, positions)
        constraints = []
        
        # Only compute upper triangle to avoid double counting
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dist_sq = distances[i, j]**2
                min_dist_sq = (radii[i] + radii[j])**2
                
                # We want dist_sq >= min_dist_sq (non-overlap constraint)
                # So we want: dist_sq - min_dist_sq >= 0
                constraints.append(dist_sq - min_dist_sq)
                
        return np.array(constraints)
    
    # Compute objective function (negative because we want to maximize)
    def objective(params):
        circles = params.reshape(-1, 3)
        # Sum of radii (we want to maximize this, so return negative)
        return -np.sum(circles[:, 2])
    
    # Enhanced post-processing with better refinement strategies - inspired by INSPIRATION 1
    def post_process(circles):
        # Try to improve the solution through multiple refinement passes
        improved = True
        iterations = 0
        max_iterations = 3  # Reduced iterations to save time
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # First, try to increase radii globally with better search strategy
            for i in range(len(circles)):
                old_x, old_y, old_r = circles[i]
                best_r = old_r
                
                # Try to increase radius while staying within bounds
                step_size = 0.001
                max_increase = 0.01
                
                test_r = old_r + step_size
                while test_r <= old_r + max_increase:
                    # Check if this radius is valid
                    valid = True
                    for j in range(len(circles)):
                        if i != j:
                            dx = old_x - circles[j, 0]
                            dy = old_y - circles[j, 1]
                            dist_sq = dx*dx + dy*dy
                            min_dist_sq = (test_r + circles[j, 2])**2
                            if dist_sq < min_dist_sq:
                                valid = False
                                break
                    
                    # Check boundary constraints
                    if old_x - test_r < 0 or old_x + test_r > 1 or old_y - test_r < 0 or old_y + test_r > 1:
                        valid = False
                    
                    if valid:
                        best_r = test_r
                        improved = True
                    else:
                        break
                    test_r += step_size
                
                # Apply the best radius found
                circles[i, 2] = best_r
            
            # Then try small positional adjustments with more thorough search
            for i in range(len(circles)):
                old_x, old_y, old_r = circles[i]
                best_x, best_y = old_x, old_y
                best_sum = np.sum(circles[:, 2])
                
                # Try more adjustments to position for better results
                moves = [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001), 
                         (0.0005, 0.0005), (-0.0005, -0.0005), (0.0005, -0.0005), (-0.0005, 0.0005)]
                for dx, dy in moves:
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    # Check bounds
                    if new_x - old_r < 0 or new_x + old_r > 1 or new_y - old_r < 0 or new_y + old_r > 1:
                        continue
                    
                    # Temporarily adjust position
                    circles[i, 0] = new_x
                    circles[i, 1] = new_y
                    
                    # Check if still valid (no overlaps)
                    valid = True
                    for j in range(len(circles)):
                        if i != j:
                            dx_ij = circles[i, 0] - circles[j, 0]
                            dy_ij = circles[i, 1] - circles[j, 1]
                            dist_sq = dx_ij*dx_ij + dy_ij*dy_ij
                            min_dist_sq = (circles[i, 2] + circles[j, 2])**2
                            if dist_sq < min_dist_sq:
                                valid = False
                                break
                    
                    if valid:
                        current_sum = np.sum(circles[:, 2])
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_x, best_y = new_x, new_y
                            improved = True
                    
                    # Restore original position
                    circles[i, 0] = old_x
                    circles[i, 1] = old_y
                
                # Apply best adjustment if found
                circles[i, 0] = best_x
                circles[i, 1] = best_y
        
        return circles
    
    # Initialize with better initialization
    initial_circles = initialize_better()
    
    # Extract initial parameters (x, y, r) for each circle
    initial_params = initial_circles.flatten()
    
    # Optimize using scipy's minimize with constraints
    try:
        # Set up bounds for variables (x, y, r)
        bounds = []
        for i in range(n):
            # Bounds for x and y positions (can't go outside square)
            bounds.extend([(0.001, 0.999), (0.001, 0.999)])
            # Bounds for radii (must be positive and reasonable)
            bounds.append((0.001, 0.499))
        
        # Define constraints
        cons = [
            {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
            {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
        ]
        
        # Optimize with multiple attempts for better results
        best_result = None
        best_sum = -float('inf')
        
        # Try multiple optimization runs with different initializations - reduced attempts for time
        for attempt in range(3):  # Reduced attempts to stay within time limit
            # Slightly perturb initial solution for diversity
            perturbed_params = initial_params.copy()
            if attempt > 0:
                # Add small random perturbations
                for i in range(0, len(perturbed_params), 3):
                    perturbed_params[i] += random.uniform(-0.005, 0.005)  # x
                    perturbed_params[i+1] += random.uniform(-0.005, 0.005)  # y
                    perturbed_params[i+2] += random.uniform(-0.002, 0.002)  # r
            
            result = minimize(
                objective,
                perturbed_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 150, 'ftol': 1e-6}  # Reduced iterations to meet time limit
            )
            
            if result.success:
                current_sum = -objective(result.x)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        
        if best_result is not None and best_result.success:
            optimized_circles = best_result.x.reshape(-1, 3)
            # Apply post-processing to fine-tune
            final_circles = post_process(optimized_circles)
            return final_circles
        else:
            # If optimization fails, return the initial configuration with post-processing
            return post_process(initial_circles)
            
    except Exception as e:
        # Fallback to initial configuration with post-processing if optimization fails
        return post_process(initial_circles)


# EVOLVE-BLOCK-END
