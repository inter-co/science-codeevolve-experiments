# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a hexagonal lattice pattern for good starting configuration
    def initialize_hexagonal_layout():
        # Try to arrange in roughly a 5x5 grid pattern with hexagonal offset
        circles = []
        rows = 5
        cols = 5
        
        # Create initial positions in a hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) * (1.0 / cols) 
                y = i * (1.0 / rows)
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                circles.append([x, y, 0.0])
        
        # Fill remaining slots if needed
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.0])
            
        return np.array(circles[:n])
    
    # Get initial configuration
    circles = initialize_hexagonal_layout()
    
    # Set initial radii to small values
    for i in range(n):
        circles[i, 2] = 0.02
    
    # Helper function to compute constraint violations
    def get_radius_sum(circles_flat):
        # Reshape flat array back to circles
        circles_arr = circles_flat.reshape(-1, 3)
        return np.sum(circles_arr[:, 2])
    
    # Constraint functions
    def containment_constraints(circles_flat):
        """Ensure all circles are within unit square"""
        circles_arr = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Each circle's center must be at least radius away from edges
        for i in range(len(circles_arr)):
            x, y, r = circles_arr[i]
            # Left constraint
            constraints.append(x - r)
            # Right constraint  
            constraints.append(1 - x - r)
            # Bottom constraint
            constraints.append(y - r)
            # Top constraint
            constraints.append(1 - y - r)
            
        return np.array(constraints)
    
    def overlap_constraints(circles_flat):
        """Ensure no overlaps between circles"""
        circles_arr = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Check pairwise distances
        for i in range(len(circles_arr)):
            for j in range(i+1, len(circles_arr)):
                x1, y1, r1 = circles_arr[i]
                x2, y2, r2 = circles_arr[j]
                
                # Distance between centers
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # Should be at least r1 + r2
                constraints.append(dist - (r1 + r2))
                
        return np.array(constraints)
    
    # Objective function (negative because we want to maximize)
    def objective(circles_flat):
        return -get_radius_sum(circles_flat)
    
    # Constraints for optimization
    def constraint_func(x):
        # Return positive values when constraints are satisfied
        cont = containment_constraints(x)
        overlap = overlap_constraints(x)
        # Combine constraints: negative values indicate constraint violation
        return np.concatenate([cont, overlap])
    
    # Create bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Flatten initial circles for optimization
    initial_flat = circles.flatten()
    
    # Use a simple gradient-based optimization approach
    try:
        # First, do a coarse optimization to improve initial configuration
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
        else:
            # If optimization fails, return initial configuration
            final_circles = circles
            
    except Exception:
        # Fallback to initial configuration if optimization fails
        final_circles = circles
    
    # Final refinement: enforce constraints manually
    # Make sure we have valid radii
    for i in range(n):
        x, y, r = final_circles[i]
        # Adjust radius to satisfy boundary constraints
        max_r = min(x, 1-x, y, 1-y)
        final_circles[i, 2] = min(r, max_r)
    
    # Ensure no overlaps by adjusting radii
    for _ in range(10):  # Multiple iterations to resolve conflicts
        changed = False
        for i in range(n):
            x1, y1, r1 = final_circles[i]
            # Find minimum distance to other circles
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    x2, y2, r2 = final_circles[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_dist = min(min_dist, dist)
            
            # Reduce radius if necessary
            new_r = min(r1, min_dist - 0.001)
            if new_r < r1:
                final_circles[i, 2] = new_r
                changed = True
                
        if not changed:
            break
    
    return final_circles


# EVOLVE-BLOCK-END
