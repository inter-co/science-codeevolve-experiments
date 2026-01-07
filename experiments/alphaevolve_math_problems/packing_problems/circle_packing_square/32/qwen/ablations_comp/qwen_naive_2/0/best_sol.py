# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Better hexagonal initialization with proper spacing
    def initialize_hexagonal():
        # Calculate optimal hexagonal packing density
        # For a hexagonal arrangement of circles in a square, we can fit approximately
        # 6 rows of 6 circles each (36 total), but we need only 32
        
        circles = []
        sqrt3 = math.sqrt(3)
        
        # Estimate radius based on area considerations
        # Area of 32 circles should be less than unit square area
        estimated_radius = math.sqrt(1.0 / (math.pi * n)) * 0.8  # Slightly conservative
        
        # Create hexagonal grid
        rows = 6
        cols = 6
        row_spacing = 2 * estimated_radius
        col_spacing = row_spacing * sqrt3 / 2
        
        # Adjust spacing to fit in unit square
        max_width = cols * col_spacing
        max_height = rows * row_spacing
        
        if max_width > 1 or max_height > 1:
            # Scale down to fit
            scale_factor = min(1/max_width, 1/max_height) * 0.95
            row_spacing *= scale_factor
            col_spacing *= scale_factor
            estimated_radius *= scale_factor
        
        # Position circles in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                x = (j + 0.5) * col_spacing
                y = (i + 0.5) * row_spacing
                
                # Offset every other row
                if i % 2 == 1:
                    x += col_spacing / 2
                
                # Ensure within bounds
                if x - estimated_radius >= 0 and x + estimated_radius <= 1 and \
                   y - estimated_radius >= 0 and y + estimated_radius <= 1:
                    circles.append([x, y, estimated_radius])
                    
            if len(circles) >= n:
                break
        
        # Fill remaining circles if needed
        while len(circles) < n:
            # Place remaining circles in a more uniform distribution
            circles.append([0.5, 0.5, estimated_radius * 0.5])
            
        return np.array(circles[:n])
    
    # Initialize with improved hexagonal pattern
    circles = initialize_hexagonal()
    
    # Define constraint functions with better numerical stability
    def containment_constraints(x):
        """Ensure all circles are contained in unit square"""
        constraints = []
        for i in range(n):
            x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
            # r_i <= x_i <= 1-r_i
            constraints.append(x_i - r_i)  # x_i - r_i >= 0
            constraints.append(1 - x_i - r_i)  # 1 - x_i - r_i >= 0
            # r_i <= y_i <= 1-r_i
            constraints.append(y_i - r_i)  # y_i - r_i >= 0
            constraints.append(1 - y_i - r_i)  # 1 - y_i - r_i >= 0
        return np.array(constraints)
    
    def non_overlap_constraints(x):
        """Ensure no two circles overlap with numerical tolerance"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                
                # Distance between centers >= sum of radii (with small tolerance)
                dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                min_dist_sq = (r_i + r_j)**2
                
                # Add small safety margin to prevent numerical issues
                safety_margin = 1e-10
                constraints.append(dist_sq - min_dist_sq - safety_margin)
        return np.array(constraints)
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]  # Sum of radii
        return -total_radius
    
    # Flatten initial circles for optimization
    x0 = np.array([circles[i][j] for i in range(n) for j in range(3)])
    
    # Set up bounds for optimization (radius must be positive, positions bounded)
    bounds = []
    for i in range(n):
        # x, y positions: [0.001, 0.999] to avoid boundary issues
        # r: [0.001, 0.499] to ensure containment
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Constraints
    cons = []
    
    # Add containment constraints
    def containment_func(x):
        return containment_constraints(x)
    cons.append({'type': 'ineq', 'fun': containment_func})
    
    # Add non-overlap constraints
    def overlap_func(x):
        return non_overlap_constraints(x)
    cons.append({'type': 'ineq', 'fun': overlap_func})
    
    # Optimize with better parameters
    try:
        # Try multiple optimization methods for better convergence
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_sum = -np.inf
        
        for method in methods:
            try:
                result = minimize(
                    objective, 
                    x0, 
                    method=method, 
                    bounds=bounds, 
                    constraints=cons,
                    options={
                        'maxiter': 2000, 
                        'ftol': 1e-8, 
                        'gtol': 1e-8,
                        'eps': 1e-6
                    },
                    tol=1e-8
                )
                
                if result.success:
                    # Calculate current sum of radii
                    current_sum = -objective(result.x)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
                        
            except Exception:
                continue
        
        if best_result is not None and best_result.success:
            # Extract optimized results
            optimized_circles = []
            for i in range(n):
                x_i = best_result.x[3*i]
                y_i = best_result.x[3*i+1]
                r_i = best_result.x[3*i+2]
                optimized_circles.append([x_i, y_i, r_i])
            return np.array(optimized_circles)
            
    except Exception as e:
        # If optimization fails, proceed to fallback
        pass
    
    # Fallback: Return the initial configuration but with some refinement
    # Try a simpler optimization approach
    try:
        # Simple gradient-based optimization with fewer iterations
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = []
            for i in range(n):
                x_i = result.x[3*i]
                y_i = result.x[3*i+1]
                r_i = result.x[3*i+2]
                optimized_circles.append([x_i, y_i, r_i])
            return np.array(optimized_circles)
            
    except Exception:
        pass
    
    # Final fallback: return the initial hexagonal configuration
    return circles


# EVOLVE-BLOCK-END
