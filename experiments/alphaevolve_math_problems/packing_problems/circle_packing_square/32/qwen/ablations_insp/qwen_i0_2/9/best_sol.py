# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations

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
    
    # Improved initialization using a more systematic approach
    def initialize_better():
        circles = np.zeros((n, 3))
        
        # Try a more structured approach: place in a grid-like pattern with varying radii
        # First, determine a good distribution
        rows = 5
        cols = 7
        if rows * cols < n:
            rows = 6
            cols = 6
            
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        idx = 0
        # Place circles in a structured pattern
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Position with slight offset to avoid perfect grid alignment
                x_offset = 0.0 if i % 2 == 0 else spacing_x * 0.25
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                
                # Calculate maximum possible radius at this position
                max_r = min(x, 1-x, y, 1-y)
                # Start with a reasonable fraction of max radius
                r = max_r * 0.4
                
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
                
        # Fill remaining slots with carefully placed circles
        # Try to place remaining circles near the edges to utilize space better
        for i in range(idx, n):
            # Place near edges to maximize utilization
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
            r = max_r * 0.3
            
            circles[i] = [x, y, r]
            
        return circles
    
    # More efficient constraint checking
    def check_constraints(circles):
        """Check if all constraints are satisfied"""
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlap constraints
        for i, j in combinations(range(len(circles)), 2):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1-x2)**2 + (y1-y2)**2
            if dist_sq < (r1+r2)**2:
                return False
                
        return True
    
    # Objective function to maximize sum of radii
    def objective(circles_flat):
        # Reshape flat array back to circles array
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we minimize
    
    # Constraint functions
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
        for i, j in combinations(range(n), 2):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1-x2)**2 + (y1-y2)**2
            overlap = dist_sq - (r1+r2)**2
            constraints.append(overlap)  # Should be >= 0
                
        return np.array(constraints)
    
    # Initialize with better configuration
    circles = initialize_better()
    
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
            options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4, 'disp': False}
        )
        
        if result.success:
            circles = result.x.reshape(-1, 3)
        else:
            # Fallback to initial configuration if optimization fails
            pass
            
    except Exception as e:
        # If optimization fails, return the initial configuration
        pass
    
    # Final validation and refinement
    circles = circles.reshape(-1, 3)
    
    # Ensure all circles are within bounds and adjust if necessary
    for i in range(n):
        x, y, r = circles[i]
        # Ensure boundaries
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]
    
    # Additional refinement: try to slightly improve by adjusting radii
    # This helps to better utilize space when optimization might have gotten stuck
    if check_constraints(circles):
        # Try to increase some radii slightly if possible
        improved = True
        iterations = 0
        while improved and iterations < 10:
            improved = False
            iterations += 1
            for i in range(n):
                # Try to increase radius while maintaining constraints
                x, y, r = circles[i]
                old_r = r
                max_r = min(x, 1-x, y, 1-y)
                
                # Try increasing radius by small amount
                test_r = min(max_r, r + 0.001)
                
                if test_r > r:
                    # Check if this change maintains all constraints
                    temp_circles = circles.copy()
                    temp_circles[i, 2] = test_r
                    
                    # Check if this works with neighbors
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
                        
    return circles


# EVOLVE-BLOCK-END
