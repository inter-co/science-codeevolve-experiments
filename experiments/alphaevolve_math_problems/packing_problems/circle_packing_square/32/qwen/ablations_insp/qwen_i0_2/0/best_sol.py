# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using hexagonal packing pattern inspired by known solutions
    def initialize_hexagonal():
        circles = np.zeros((n, 3))
        
        # Create a more sophisticated hexagonal packing pattern
        # We'll try to arrange in roughly 5 rows with alternating offsets
        rows = 5
        cols_per_row = [6, 5, 6, 5, 6]  # Alternating pattern
        
        # Determine spacing based on ideal hexagonal packing
        # For a hexagon with side length s, the distance between centers is s
        # In our case, we want to fit 5 rows with 5-6 columns each
        spacing_x = 0.8  # Leave margin for edge constraints
        spacing_y = 0.8
        
        # Adjust spacing so that we can fit within unit square
        actual_cols = max(cols_per_row)
        actual_rows = rows
        
        # Scale spacing to fit nicely
        max_width = actual_cols * spacing_x
        max_height = actual_rows * spacing_y
        
        # Scale down to fit in unit square
        scale_x = 0.95 / max_width if max_width > 0 else 1.0
        scale_y = 0.95 / max_height if max_height > 0 else 1.0
        scale = min(scale_x, scale_y)
        
        spacing_x *= scale
        spacing_y *= scale
        
        # Center the pattern
        offset_x = (1.0 - actual_cols * spacing_x) / 2.0
        offset_y = (1.0 - actual_rows * spacing_y) / 2.0
        
        idx = 0
        for i in range(rows):
            row_cols = cols_per_row[i]
            # Offset every other row
            x_offset = spacing_x * 0.5 if i % 2 == 1 else 0.0
            
            for j in range(row_cols):
                if idx >= n:
                    break
                x = offset_x + j * spacing_x + x_offset
                y = offset_y + i * spacing_y
                
                # Calculate maximum possible radius at this position
                max_r = min(x, 1-x, y, 1-y)
                # Use a more aggressive but safe initial radius
                r = max_r * 0.45
                
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
                
        # Fill remaining slots with carefully placed circles
        # Use a more systematic approach for remaining positions
        for i in range(idx, n):
            # Try to place in corners or along edges
            corner_positions = [
                (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),
                (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5)
            ]
            
            if i < len(corner_positions):
                x, y = corner_positions[i]
            else:
                # Random edge positions
                edge = np.random.choice(['top', 'bottom', 'left', 'right'])
                if edge == 'top':
                    x = np.random.uniform(0.1, 0.9)
                    y = 0.95
                elif edge == 'bottom':
                    x = np.random.uniform(0.1, 0.9)
                    y = 0.05
                elif edge == 'left':
                    x = 0.05
                    y = np.random.uniform(0.1, 0.9)
                else:  # right
                    x = 0.95
                    y = np.random.uniform(0.1, 0.9)
            
            # Calculate max radius at this position
            max_r = min(x, 1-x, y, 1-y)
            r = max_r * 0.35
            
            circles[i] = [x, y, r]
            
        return circles
    
    # More efficient constraint checking with early termination
    def check_constraints(circles):
        """Check if all constraints are satisfied"""
        # Check boundary constraints first
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlap constraints with early termination
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            for j in range(i+1, len(circles)):
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                if dist_sq < (r1+r2)**2:
                    return False
                    
        return True
    
    # Optimized objective function
    def objective(circles_flat):
        # Reshape flat array back to circles array
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we minimize
    
    # Optimized constraint functions
    def constraint_func(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Boundary constraints: each circle must be fully contained
        for i in range(n):
            x, y, r = circles[i]
            # x - r >= 0 and 1 - x - r >= 0 and y - r >= 0 and 1 - y - r >= 0
            constraints.append(x - r)  # x - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        # Overlap constraints: distance >= sum of radii
        # Use more efficient pairwise checking
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                overlap = dist_sq - (r1+r2)**2
                constraints.append(overlap)  # Should be >= 0
                
        return np.array(constraints)
    
    # Enhanced local refinement
    def local_refinement(circles):
        """Try to improve the solution through local adjustments"""
        improved = True
        iterations = 0
        max_iterations = 20
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try to increase radii where possible
            for i in range(n):
                x, y, r = circles[i]
                old_r = r
                max_r = min(x, 1-x, y, 1-y)
                
                # Try to increase radius by small amounts
                test_r = min(max_r, r + 0.005)
                
                if test_r > r:
                    # Check if this change maintains all constraints
                    temp_circles = circles.copy()
                    temp_circles[i, 2] = test_r
                    
                    # Check all neighbor constraints efficiently
                    valid = True
                    for j in range(n):
                        if i != j:
                            x1, y1, r1 = temp_circles[i]
                            x2, y2, r2 = temp_circles[j]
                            dist_sq = (x1-x2)**2 + (y1-y2)**2
                            if dist_sq < (r1+r2)**2:
                                valid = False
                                break
                    
                    if valid:
                        circles[i, 2] = test_r
                        improved = True
            
            # Try to slightly adjust positions to allow larger radii
            for i in range(n):
                x, y, r = circles[i]
                # Try small adjustments to position
                best_x, best_y = x, y
                best_r = r
                best_valid = True
                
                # Test small position adjustments
                adjustments = [(0, 0), (-0.005, 0), (0.005, 0), (0, -0.005), (0, 0.005)]
                for dx, dy in adjustments:
                    new_x = x + dx
                    new_y = y + dy
                    if 0 < new_x < 1 and 0 < new_y < 1:
                        # Check if we can increase radius at new position
                        new_max_r = min(new_x, 1-new_x, new_y, 1-new_y)
                        if new_max_r > r:
                            # Check if still valid with neighbors
                            temp_circles = circles.copy()
                            temp_circles[i, 0] = new_x
                            temp_circles[i, 1] = new_y
                            temp_circles[i, 2] = new_max_r
                            
                            valid = True
                            for j in range(n):
                                if i != j:
                                    x1, y1, r1 = temp_circles[i]
                                    x2, y2, r2 = temp_circles[j]
                                    dist_sq = (x1-x2)**2 + (y1-y2)**2
                                    if dist_sq < (r1+r2)**2:
                                        valid = False
                                        break
                            
                            if valid and new_max_r > best_r:
                                best_x, best_y, best_r = new_x, new_y, new_max_r
                                
                if best_r > r:
                    circles[i, 0] = best_x
                    circles[i, 1] = best_y
                    circles[i, 2] = best_r
                    improved = True
                    
        return circles
    
    # Multiple restart strategy
    best_result = None
    best_sum = 0
    
    # Try multiple initializations and optimization runs
    for restart in range(5):
        # Initialize with better configuration
        circles = initialize_hexagonal()
        
        # Flatten for optimization
        circles_flat = circles.flatten()
        
        # Optimize using scipy minimize with constraints
        try:
            # Create bounds for each variable (x, y, r) for each circle
            bounds = []
            for i in range(n):
                # x bounds: r <= x <= 1-r, y bounds: r <= y <= 1-r, r bounds: 0 < r <= 0.5
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            # Set up constraints
            cons = {
                'type': 'ineq',
                'fun': constraint_func
            }
            
            # Perform optimization with better settings
            result = minimize(
                objective,
                circles_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-4, 'disp': False}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                # Local refinement
                refined_circles = local_refinement(optimized_circles.copy())
                
                # Check if this is better
                current_sum = np.sum(refined_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = refined_circles.copy()
            else:
                # Even if optimization fails, do local refinement on initial
                refined_circles = local_refinement(circles.copy())
                current_sum = np.sum(refined_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = refined_circles.copy()
                    
        except Exception as e:
            # If optimization fails, do local refinement on initial
            refined_circles = local_refinement(circles.copy())
            current_sum = np.sum(refined_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = refined_circles.copy()
    
    # If we didn't get anything, return the initial configuration
    if best_result is None:
        circles = initialize_hexagonal()
        best_result = local_refinement(circles.copy())
    
    return best_result


# EVOLVE-BLOCK-END
