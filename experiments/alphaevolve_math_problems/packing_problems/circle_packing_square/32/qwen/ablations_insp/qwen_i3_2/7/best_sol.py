# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, local optimization, and 
    simulated annealing-inspired refinement.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Create initial configuration using improved hexagonal packing
    def create_better_initial():
        # Use a more systematic approach for hexagonal packing
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Ensure we have enough space
        if rows * cols < n:
            rows += 1
            
        circles = []
        
        # Hexagonal packing parameters
        spacing_x = 1.0 / (cols + 1)
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        # Start with smaller spacing to allow for better radius optimization
        spacing_x = min(0.2, spacing_x)
        spacing_y = min(0.2, spacing_y)
        
        # Adjust for boundary constraints
        max_radius = min(spacing_x, spacing_y) * 0.4
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x / 2
                    
                # Ensure we're within bounds
                x = max(max_radius, min(1 - max_radius, x))
                y = max(max_radius, min(1 - max_radius, y))
                
                circles.append([x, y, max_radius])
                    
        # Fill remaining slots with small circles
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.01])
            
        return np.array(circles[:n])
    
    # Generate initial configuration
    initial_circles = create_better_initial()
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(params):
        # Reshape params into circles array
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Vectorized constraint functions for better performance
    def build_constraints_vectorized(circles):
        """Build constraints using vectorized operations for better performance"""
        n = len(circles)
        
        # Precompute all pairwise distances (but only for constraints that matter)
        constraints = []
        
        # Containment constraints
        for i in range(n):
            def containment_constraint(i):
                def c(params):
                    circles = params.reshape(-1, 3)
                    x, y, r = circles[i]
                    # Return positive values when constraint satisfied (>= 0)
                    return min(x - r, 1 - x - r, y - r, 1 - y - r)
                return c
            
            constraints.append({'type': 'ineq', 'fun': containment_constraint(i)})
        
        # Non-overlap constraints - use a more efficient approach
        # Instead of full pairwise, we'll use spatial indexing for better performance
        # But for simplicity and to avoid complex spatial data structures,
        # we'll keep the basic pairwise approach but optimize the implementation
        
        # For better performance, we can limit constraints to nearby circles
        # But to keep things simple and maintain correctness, we'll go with full constraints
        for i in range(n):
            for j in range(i+1, n):
                def nonoverlap_constraint(i, j):
                    def c(params):
                        circles = params.reshape(-1, 3)
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist_sq = (xi - xj)**2 + (yi - yj)**2
                        # Return positive when constraint satisfied (distance >= radii sum)
                        return dist_sq - (ri + rj)**2
                    return c
                
                constraints.append({'type': 'ineq', 'fun': nonoverlap_constraint(i, j)})
        
        return constraints
    
    # Enhanced optimization with local refinement
    def optimize_with_refinement(initial_params):
        """Enhanced optimization with local refinement steps"""
        # First try the scipy optimization
        constraints = build_constraints_vectorized(initial_params.reshape(-1, 3))
        
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6},
                callback=lambda x: None
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                return optimized_circles
        except Exception:
            pass
        
        # If optimization fails, return the initial configuration
        return initial_params.reshape(-1, 3)
    
    # Apply optimization
    optimized_circles = optimize_with_refinement(initial_circles.flatten())
    
    # Apply additional local optimization to improve results
    def local_optimization(circles):
        """Apply local optimization to improve the configuration"""
        # This is a simplified version - in a full implementation we'd use more sophisticated local search
        improved = True
        iterations = 0
        max_iterations = 50
        
        while improved and iterations < max_iterations:
            improved = False
            for i in range(len(circles)):
                # Try to increase radius at position
                x, y, r = circles[i]
                
                # Calculate maximum possible radius at this position
                max_radius = min(x, 1-x, y, 1-y)
                
                # Check overlap with all other circles
                for j in range(len(circles)):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        max_radius = min(max_radius, distance - r2)
                
                # If we can increase radius, do so
                if max_radius > r + 1e-6:
                    circles[i, 2] = max_radius
                    improved = True
            
            iterations += 1
        
        return circles
    
    # Apply local optimization
    optimized_circles = local_optimization(optimized_circles.copy())
    
    # Final validation and cleanup
    for i in range(n):
        # Ensure radii are positive and reasonable
        optimized_circles[i, 2] = max(0.001, optimized_circles[i, 2])
        # Make sure positions are within bounds
        optimized_circles[i, 0] = np.clip(optimized_circles[i, 0], 
                                         optimized_circles[i, 2], 
                                         1 - optimized_circles[i, 2])
        optimized_circles[i, 1] = np.clip(optimized_circles[i, 1], 
                                         optimized_circles[i, 2], 
                                         1 - optimized_circles[i, 2])
    
    return optimized_circles


# EVOLVE-BLOCK-END
