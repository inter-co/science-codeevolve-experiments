# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with numerical optimization and 
    local refinement for improved results.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Enhanced hexagonal initialization inspired by INSPIRATION 2
    def initialize_hexagonal_layout():
        circles = np.zeros((n, 3))
        
        # Create a more systematic hexagonal pattern
        rows = 6
        cols = 6
        
        # Calculate spacing for hexagonal packing
        radius_estimate = 0.08
        spacing_x = 2 * radius_estimate
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        # Ensure we stay within bounds
        max_rows = int((1.0 - 2*radius_estimate) / spacing_y) + 1
        max_cols = int((1.0 - 2*radius_estimate) / spacing_x) + 1
        
        rows = min(rows, max_rows)
        cols = min(cols, max_cols)
        
        # Create hexagonal grid with offset rows
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = spacing_x / 2.0 if i % 2 == 1 else 0.0
                x = 0.1 + j * spacing_x + x_offset
                y = 0.1 + i * spacing_y
                
                # Ensure circle fits within bounds
                if x + radius_estimate <= 0.9 and y + radius_estimate <= 0.9:
                    circles[idx] = [x, y, radius_estimate * 0.95]
                    idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with random placements
        while idx < n:
            circles[idx] = [
                np.random.uniform(0.1, 0.9),
                np.random.uniform(0.1, 0.9),
                np.random.uniform(0.02, 0.08)
            ]
            idx += 1
            
        return circles
    
    # Improved constraint functions with better numerical handling
    def create_constraints():
        """Create constraint functions for optimization"""
        # For better performance, we'll create a single constraint function
        # that evaluates all constraints at once
        
        def boundary_constraint(vars):
            # vars is flattened array [x1, y1, r1, x2, y2, r2, ...]
            constraints = []
            for i in range(n):
                x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
                # Ensure circle is fully within bounds with safety margin
                constraints.extend([
                    x - r - 1e-6,      # x - r >= 0
                    y - r - 1e-6,      # y - r >= 0
                    1 - x - r - 1e-6,  # 1 - x - r >= 0
                    1 - y - r - 1e-6   # 1 - y - r >= 0
                ])
            return np.array(constraints)
        
        def overlap_constraint(vars):
            # vars is flattened array [x1, y1, r1, x2, y2, r2, ...]
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                    x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                    # Distance between centers must be >= sum of radii
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist_sq = (r1 + r2)**2
                    # Add small tolerance to prevent numerical issues
                    constraints.append(dist_sq - min_dist_sq - 1e-8)
            return np.array(constraints)
        
        return boundary_constraint, overlap_constraint
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(vars):
        # Sum of all radii (every third element starting from index 2)
        return -np.sum(vars[2::3])  # Negative because minimize
    
    # Enhanced optimization with fallback strategies
    def optimize_with_fallback(initial_vars):
        # First attempt with SLSQP
        try:
            result = minimize(
                objective,
                initial_vars,
                method='SLSQP',
                bounds=[(1e-6, 1-1e-6) for _ in range(3*n)],
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: create_constraints()[0](x)},
                    {'type': 'ineq', 'fun': lambda x: create_constraints()[1](x)}
                ],
                options={'maxiter': 1000, 'ftol': 1e-7, 'eps': 1e-6},
                tol=1e-7
            )
            
            if result.success:
                return result.x
        except Exception:
            pass
        
        # Fallback to L-BFGS-B if SLSQP fails
        try:
            result = minimize(
                objective,
                initial_vars,
                method='L-BFGS-B',
                bounds=[(1e-6, 1-1e-6) for _ in range(3*n)],
                options={'maxiter': 1000, 'ftol': 1e-7}
            )
            
            if result.success:
                return result.x
        except Exception:
            pass
        
        # If both fail, return initial values
        return initial_vars
    
    # Local search refinement to improve results
    def local_search(circles):
        best_circles = circles.copy()
        best_sum = np.sum(circles[:, 2])
        
        # Try small position adjustments
        for iter_num in range(100):
            improved = False
            for i in range(n):
                # Save current state
                original_x, original_y, original_r = best_circles[i]
                
                # Try small perturbations
                best_pos = [original_x, original_y]
                best_radius = original_r
                
                # Test several small moves
                moves = [(-0.005, -0.005), (-0.005, 0), (-0.005, 0.005),
                         (0, -0.005), (0, 0.005),
                         (0.005, -0.005), (0.005, 0), (0.005, 0.005)]
                
                for dx, dy in moves:
                    test_x = original_x + dx
                    test_y = original_y + dy
                    
                    # Check if position is valid
                    if (test_x - original_r >= 0 and 
                        test_x + original_r <= 1 and 
                        test_y - original_r >= 0 and 
                        test_y + original_r <= 1):
                        
                        # Check overlaps with other circles
                        valid = True
                        for j in range(n):
                            if i != j:
                                dx_ij = test_x - best_circles[j, 0]
                                dy_ij = test_y - best_circles[j, 1]
                                dist = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                                if dist < original_r + best_circles[j, 2]:
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
    
    # Main optimization process
    # Initialize
    circles = initialize_hexagonal_layout()
    initial_vars = circles.flatten()
    
    # Optimize
    optimized_vars = optimize_with_fallback(initial_vars)
    
    # Convert back to circles format
    final_circles = optimized_vars.reshape(-1, 3)
    
    # Apply local search refinement
    refined_circles = local_search(final_circles)
    
    return refined_circles


# EVOLVE-BLOCK-END
