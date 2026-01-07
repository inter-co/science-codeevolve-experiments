# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal grid initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Create initial hexagonal grid layout with better spacing and more careful initialization
    def create_hexagonal_layout():
        # Arrange circles in a roughly hexagonal pattern
        rows = 6
        cols = 6
        spacing_x = 0.9 / cols  # Leave some margin
        spacing_y = 0.9 / rows
        
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.05 + (j + 1) * spacing_x
                y = 0.05 + (i + 1) * spacing_y
                # Offset every other row for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x / 2
                # Start with smaller initial radius to allow more room for optimization
                circles.append([x, y, min(spacing_x, spacing_y) * 0.25])  
                
        # Fill remaining positions if needed with center positions
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.05])
            
        return np.array(circles[:n])
    
    # Initialize with hexagonal layout
    initial_circles = create_hexagonal_layout()
    
    # Flatten for optimization
    initial_flat = initial_circles.flatten()
    
    # Objective function: negative sum of radii (we want to maximize sum of radii)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Helper function to create constraints with proper closure
    def make_containment_constraint(i):
        def constraint(circles_flat):
            x, y, r = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
            # Circle must be contained: r <= x <= 1-r and r <= y <= 1-r
            return [x - r, 1 - x - r, y - r, 1 - y - r]
        return constraint
    
    def make_overlap_constraint(i, j):
        def constraint(circles_flat):
            x1, y1, r1 = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
            x2, y2, r2 = circles_flat[3*j], circles_flat[3*j+1], circles_flat[3*j+2]
            # Distance between centers minus sum of radii must be >= 0
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            return [dist - r1 - r2]
        return constraint
    
    # Constraints using proper constraint definitions
    cons = []
    
    # Add containment constraints for all circles
    for i in range(n):
        # Each containment constraint returns 4 values (for x, y, r bounds)
        cons.append({'type': 'ineq', 'fun': make_containment_constraint(i)})
    
    # Add overlap constraints for all pairs of circles
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
    
    # Bounds: x, y in [r, 1-r], r in [0.001, 0.499]
    bounds = []
    for i in range(n):
        bounds.extend([
            (0.001, 0.999),  # x coordinate
            (0.001, 0.999),  # y coordinate
            (0.001, 0.499)   # radius (max possible without overlap)
        ])
    
    # Optimize using SLSQP method with more iterations and better tolerance
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-8}
        )
        
        if result.success:
            optimized_flat = result.x
            circles = optimized_flat.reshape((n, 3))
        else:
            # Fallback to initial configuration if optimization fails
            circles = initial_circles
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        circles = initial_circles
    
    # Ensure final constraints are met
    for i in range(n):
        # Ensure radius is positive and within bounds
        circles[i, 2] = max(0.001, min(0.499, circles[i, 2]))
        # Ensure circle is contained
        circles[i, 0] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 0]))
        circles[i, 1] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 1]))
    
    return circles


# EVOLVE-BLOCK-END
