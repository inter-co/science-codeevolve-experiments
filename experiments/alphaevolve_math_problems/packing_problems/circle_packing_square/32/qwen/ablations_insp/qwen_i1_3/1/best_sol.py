# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: hexagonal grid initialization + constrained optimization + refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using hexagonal grid pattern
    def initialize_hexagonal_grid():
        # Create a hexagonal grid pattern that fits in the unit square
        # We'll use a 6x6 grid of points (36 total) and take the first 32
        rows = 6
        cols = 6
        
        # Hexagon parameters - tuned for 32 circles
        side_length = 0.18  # Adjusted for better packing
        hex_height = side_length * math.sqrt(3) / 2
        hex_width = side_length
        
        # Generate hexagonal grid points
        points = []
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5 * (i % 2)) * hex_width + 0.05  # Add offset to center
                y = i * hex_height + 0.05
                if x <= 0.95 and y <= 0.95:  # Leave margin for boundary constraints
                    points.append([x, y])
        
        # Take first 32 points
        points = points[:n]
        
        # Initialize with equal small radii
        circles = np.zeros((n, 3))
        for i, (x, y) in enumerate(points):
            circles[i] = [x, y, 0.03]  # Start with slightly larger initial radius
            
        return circles
    
    # Constraint functions for scipy optimization
    def constraint_bounds(circles_flat):
        """Ensure all circles are contained within unit square"""
        constraints = []
        for i in range(n):
            x, y, r = circles_flat[3*i:3*i+3]
            # r <= x <= 1-r and r <= y <= 1-r
            constraints.append(x - r)  # x - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
        return np.array(constraints)
    
    def constraint_overlaps(circles_flat):
        """Ensure no two circles overlap"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_flat[3*i:3*i+3]
                x2, y2, r2 = circles_flat[3*j:3*j+3]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                constraints.append(distance_sq - min_distance_sq)  # Should be >= 0
        return np.array(constraints)
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Negative because minimize
    
    # Initialize
    circles = initialize_hexagonal_grid()
    
    # Flatten for optimization
    initial_guess = circles.flatten()
    
    # Define bounds for variables (x, y, r for each circle)
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
    
    # Define constraints
    def constraint_func(circles_flat):
        return np.concatenate([constraint_bounds(circles_flat), constraint_overlaps(circles_flat)])
    
    # Run optimization with bounds using SLSQP (more robust than L-BFGS-B for constrained problems)
    try:
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, 
                         constraints={'type': 'ineq', 'fun': constraint_func},
                         options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
        else:
            # Fallback to initial configuration if optimization fails
            optimized_circles = circles
            
    except Exception:
        # If optimization fails, return initial configuration
        optimized_circles = circles
    
    # Final validation and refinement
    # Ensure all circles fit properly within the unit square
    for i in range(n):
        x, y, r = optimized_circles[i]
        # Adjust radius to fit within bounds
        r = min(r, x, 1-x, y, 1-y)
        optimized_circles[i] = [x, y, r]
    
    # More aggressive refinement with local search
    def refine_solution():
        improved = True
        iterations = 0
        max_iterations = 20
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try to increase radii and adjust positions
            for i in range(n):
                best_r = optimized_circles[i, 2]
                best_x, best_y = optimized_circles[i, 0], optimized_circles[i, 1]
                
                # Try different adjustments
                adjustments = [
                    (0, 0, 0.001),   # Small radius increase
                    (0, 0, 0.002),   # Medium radius increase
                    (-0.005, 0, 0),  # Move left
                    (0.005, 0, 0),   # Move right
                    (0, -0.005, 0),  # Move down
                    (0, 0.005, 0),   # Move up
                    (-0.003, -0.003, 0.001),  # Diagonal move + radius increase
                ]
                
                for dx, dy, dr in adjustments:
                    new_x = optimized_circles[i, 0] + dx
                    new_y = optimized_circles[i, 1] + dy
                    new_r = optimized_circles[i, 2] + dr
                    
                    # Check bounds
                    if (0 <= new_x <= 1 and 0 <= new_y <= 1 and 
                        new_r > 0 and new_r <= 0.5 and
                        new_x - new_r >= 0 and new_x + new_r <= 1 and
                        new_y - new_r >= 0 and new_y + new_r <= 1):
                        
                        # Check overlap with other circles
                        valid = True
                        for j in range(n):
                            if i != j:
                                dist = math.sqrt((new_x - optimized_circles[j, 0])**2 + 
                                                (new_y - optimized_circles[j, 1])**2)
                                if dist < (new_r + optimized_circles[j, 2]):
                                    valid = False
                                    break
                        
                        if valid and new_r > best_r:
                            best_r = new_r
                            best_x, best_y = new_x, new_y
                            improved = True
                
                optimized_circles[i] = [best_x, best_y, best_r]
    
    refine_solution()
    
    return optimized_circles


# EVOLVE-BLOCK-END
