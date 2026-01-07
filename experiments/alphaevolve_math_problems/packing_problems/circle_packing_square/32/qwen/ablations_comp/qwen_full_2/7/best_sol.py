# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: hexagonal packing initialization + constrained optimization refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better hexagonal initialization (based on INSPIRATION 2)
    def create_hexagonal_pattern():
        # Arrange circles in a hexagonal grid pattern
        rows = 6
        cols = 6
        circles = []
        
        # Hexagonal packing parameters - use a more precise approach
        radius_guess = 0.08  # Initial guess for radius
        spacing = 2 * radius_guess  # Center-to-center distance
        
        # Create hexagonal grid with better positioning
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing + (i % 2) * spacing/2
                y = 0.1 + i * spacing * math.sqrt(3)/2
                # Ensure circles are fully within bounds
                if (x - radius_guess >= 0 and x + radius_guess <= 1 and 
                    y - radius_guess >= 0 and y + radius_guess <= 1):
                    circles.append([x, y, radius_guess])
        
        # Fill remaining positions with random circles
        while len(circles) < n:
            x = np.random.uniform(radius_guess, 1-radius_guess)
            y = np.random.uniform(radius_guess, 1-radius_guess)
            circles.append([x, y, radius_guess])
            
        return np.array(circles[:n])
    
    # Constraint functions (cleaner and more precise)
    def constraint_containment(params):
        """Ensure all circles are fully contained in the unit square"""
        circles = params.reshape(-1, 3)
        constraints = []
        
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Circle must be within bounds with a safety margin
            constraints.append(x - r)  # x - r >= 0
            constraints.append(y - r)  # y - r >= 0  
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        return np.array(constraints)
    
    def constraint_nonoverlap(params):
        """Ensure no two circles overlap"""
        circles = params.reshape(-1, 3)
        constraints = []
        
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Distance between centers must be >= sum of radii
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # We want dist_sq >= min_dist_sq, so we return dist_sq - min_dist_sq
                constraints.append(dist_sq - min_dist_sq)
                
        return np.array(constraints)
    
    # Objective function (negative because we want to maximize)
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Multi-stage optimization approach
    best_result = None
    best_sum = -float('inf')
    
    # Try multiple initializations to get better results
    for attempt in range(3):
        # Create initial configuration
        initial_circles = create_hexagonal_pattern()
        initial_flat = initial_circles.flatten()
        
        # Stage 1: SLSQP optimization with proper constraints
        try:
            result = minimize(
                objective,
                initial_flat,
                method='SLSQP',
                bounds=[(1e-6, 1-1e-6) for _ in range(3*n)],  # Tighter bounds
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
                    {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
                ],
                options={'maxiter': 800, 'ftol': 1e-7, 'disp': False}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                optimized_sum = np.sum(optimized_circles[:, 2])
                if optimized_sum > best_sum:
                    best_sum = optimized_sum
                    best_result = result.x.copy()
        except Exception:
            pass
    
    # If no optimization worked, use the initial configuration
    if best_result is None:
        best_result = initial_flat.copy()
    
    # Convert back to circles array
    final_circles = best_result.reshape(-1, 3)
    
    # Enhanced local search refinement (from INSPIRATION 2 but improved)
    def local_search(circles):
        best_circles = circles.copy()
        best_sum = np.sum(circles[:, 2])
        
        # More thorough local search with multiple passes
        for iteration in range(300):  # Increased iterations
            improved = False
            # Randomize order for better exploration
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                best_pos = best_circles[i, :2].copy()
                
                # Try larger range of perturbations for better exploration
                step_sizes = [-0.015, -0.01, -0.005, -0.002, 0, 0.002, 0.005, 0.01, 0.015]
                for dx in step_sizes:
                    for dy in step_sizes:
                        test_x = best_circles[i, 0] + dx
                        test_y = best_circles[i, 1] + dy
                        
                        # Check if position is valid
                        if (test_x - best_circles[i, 2] >= 0 and 
                            test_x + best_circles[i, 2] <= 1 and 
                            test_y - best_circles[i, 2] >= 0 and 
                            test_y + best_circles[i, 2] <= 1):
                            
                            # Check overlaps with other circles
                            valid = True
                            for j in range(n):
                                if i != j:
                                    dx_ij = test_x - best_circles[j, 0]
                                    dy_ij = test_y - best_circles[j, 1]
                                    dist = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                                    if dist < best_circles[i, 2] + best_circles[j, 2]:
                                        valid = False
                                        break
                            
                            if valid:
                                # Temporarily update and evaluate
                                old_x, old_y = best_circles[i, 0], best_circles[i, 1]
                                best_circles[i, 0], best_circles[i, 1] = test_x, test_y
                                
                                # Recalculate sum of radii
                                new_sum = np.sum(best_circles[:, 2])
                                
                                if new_sum > best_sum:
                                    best_sum = new_sum
                                    best_pos[0], best_pos[1] = test_x, test_y
                                    improved = True
                                
                                # Restore original position
                                best_circles[i, 0], best_circles[i, 1] = old_x, old_y
                
                if improved:
                    best_circles[i, 0], best_circles[i, 1] = best_pos[0], best_pos[1]
        
        return best_circles
    
    # Apply local search refinement
    refined_circles = local_search(final_circles)
    
    return refined_circles


# EVOLVE-BLOCK-END
