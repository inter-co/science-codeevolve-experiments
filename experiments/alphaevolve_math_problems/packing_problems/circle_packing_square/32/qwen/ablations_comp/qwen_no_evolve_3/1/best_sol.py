# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a grid-based heuristic
    def initialize_grid():
        # Create a grid pattern to get initial good placement
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        
        # Adjust grid to fit within unit square with margin
        margin = 0.05
        grid_size = min(1 - 2*margin, 1 - 2*margin)
        cell_width = grid_size / cols
        cell_height = grid_size / rows
        
        circles = []
        radius_guess = min(cell_width, cell_height) / 4.0
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = margin + (j + 0.5) * cell_width
                y = margin + (i + 0.5) * cell_height
                circles.append([x, y, radius_guess])
        
        # Ensure we have exactly n circles
        while len(circles) < n:
            circles.append([0.5, 0.5, radius_guess])
            
        return np.array(circles[:n])
    
    # Constraint functions for optimization
    def constraint_containment(circles_flat):
        """Ensure all circles are contained within the unit square"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Circle must be contained in unit square
            constraints.append(x - r)  # x - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        return np.array(constraints)
    
    def constraint_nonoverlap(circles_flat):
        """Ensure no two circles overlap"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Check pairwise distances
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Distance between centers minus sum of radii must be >= 0
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                constraints.append(dist - (r1 + r2))
                
        return np.array(constraints)
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -sum(circle[2] for circle in circles)
    
    # Set up initial configuration
    initial_circles = initialize_grid()
    
    # Flatten for optimization
    initial_flat = initial_circles.flatten()
    
    # Define bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0.001, 0.999))  # Small buffer to prevent boundary issues
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds (must be positive and respect containment)
        bounds.append((0.001, 0.499))
    
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
    ]
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial configuration if optimization fails
            return initial_circles
            
    except Exception as e:
        # Fallback to initial configuration if anything goes wrong
        return initial_circles


# EVOLVE-BLOCK-END
