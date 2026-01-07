# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: hexagonal packing initialization + constrained optimization refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    n = 32
    
    # Initial configuration: hexagonal packing pattern with better spacing
    def create_hexagonal_pattern():
        # Arrange circles in a hexagonal grid pattern
        rows = 6
        cols = 6
        circles = []
        
        # Hexagonal packing parameters - better spacing
        radius_guess = 0.08  # Initial guess for radius
        spacing = 2 * radius_guess  # Center-to-center distance
        
        # Create hexagonal grid with more careful placement
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing + (i % 2) * spacing/2
                y = 0.1 + i * spacing * math.sqrt(3)/2
                # Ensure circles are fully within bounds
                if x - radius_guess >= 0 and x + radius_guess <= 1 and \
                   y - radius_guess >= 0 and y + radius_guess <= 1:
                    circles.append([x, y, radius_guess])
        
        # If we don't have enough circles, fill with remaining ones
        while len(circles) < n:
            # Place remaining circles randomly but still within bounds
            x = np.random.uniform(radius_guess, 1-radius_guess)
            y = np.random.uniform(radius_guess, 1-radius_guess)
            circles.append([x, y, radius_guess])
            
        return np.array(circles[:n])
    
    # Constraint functions (similar to INSPIRATION 1 but improved)
    def constraint_containment(circles_flat):
        """Ensure all circles are fully contained in the unit square"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Circle must be within bounds (with numerical safety margin)
            constraints.append(x - r - 1e-8)  # x - r >= 0
            constraints.append(y - r - 1e-8)  # y - r >= 0
            constraints.append(1 - x - r - 1e-8)  # 1 - x - r >= 0
            constraints.append(1 - y - r - 1e-8)  # 1 - y - r >= 0
            
        return np.array(constraints)
    
    def constraint_nonoverlap(circles_flat):
        """Ensure no two circles overlap with numerical tolerance"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Use spatial indexing for better performance on large numbers
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Distance between centers must be >= sum of radii
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # Add small tolerance to prevent numerical issues
                constraints.append(dist_sq - min_dist_sq - 1e-10)
                
        return np.array(constraints)
    
    # Objective function (negative because we want to maximize)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Enhanced optimization with multiple strategies
    def optimize_with_refinement(initial_flat):
        # First optimization pass with SLSQP
        try:
            result = minimize(
                objective,
                initial_flat,
                method='SLSQP',
                bounds=[(1e-8, 1-1e-8) for _ in range(3*n)],  # Tighter bounds
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
                    {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
                ],
                options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success:
                final_circles = result.x.reshape(-1, 3)
                return final_circles
        except Exception as e:
            # Log error for debugging but continue
            pass
        
        # Fallback: try L-BFGS-B if SLSQP fails
        try:
            result = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=[(1e-8, 1-1e-8) for _ in range(3*n)],
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                final_circles = result.x.reshape(-1, 3)
                return final_circles
        except Exception as e:
            # Log error for debugging but continue
            pass
        
        # Second fallback: try Trust-Constr if available
        try:
            result = minimize(
                objective,
                initial_flat,
                method='trust-constr',
                bounds=[(1e-8, 1-1e-8) for _ in range(3*n)],
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                final_circles = result.x.reshape(-1, 3)
                return final_circles
        except Exception as e:
            # Log error for debugging but continue
            pass
        
        # Return initial configuration as last resort
        return initial_flat.reshape(-1, 3)
    
    # Create initial configuration
    initial_circles = create_hexagonal_pattern()
    initial_flat = initial_circles.flatten()
    
    # Apply optimization with refinement
    final_circles = optimize_with_refinement(initial_flat)
    
    # Post-processing: fine-tune using local search (improved version)
    def local_search(circles):
        # Try small adjustments to improve packing
        best_circles = circles.copy()
        best_sum = np.sum(circles[:, 2])
        
        # Try small position adjustments with more thorough search
        max_iterations = 300  # Increased iterations for better search
        for iter_num in range(max_iterations):
            improved = False
            # Randomize order of circle updates for better exploration
            circle_indices = list(range(n))
            random.shuffle(circle_indices)
            
            for i in circle_indices:
                # Try small perturbations with more granular steps
                best_pos = best_circles[i, :2].copy()
                best_radius = best_circles[i, 2]
                
                # Try a wider range of perturbations for better exploration
                step_sizes = [-0.02, -0.01, -0.005, -0.002, 0, 0.002, 0.005, 0.01, 0.02]
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
                                
                                # Recalculate sum of radii (we're trying to maximize)
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
