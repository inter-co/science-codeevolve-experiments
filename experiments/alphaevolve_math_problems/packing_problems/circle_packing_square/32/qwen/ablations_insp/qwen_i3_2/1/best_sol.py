# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

# Global constants for the problem
N_CIRCLES = 32
UNIT_SQUARE_SIZE = 1.0
BENCHMARK = 2.937944526205518

def create_better_initial():
    """Create a better initial configuration using a more strategic approach."""
    # Start with a hexagonal-like arrangement but with better spacing
    circles = []
    
    # Try to create a more uniform distribution
    # We'll use a 6x6 grid pattern with adjusted spacing
    rows = 6
    cols = 6
    
    # Calculate spacing to accommodate all circles reasonably
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Create a more strategic initial placement
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= N_CIRCLES:
                break
            # Create hexagonal pattern with offset rows
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Offset odd rows for better hexagonal packing
            if i % 2 == 1:
                x += spacing_x / 2
                
            # Adjust radius based on proximity to boundaries
            max_radius = min(x, 1-x, y, 1-y)
            # Use a slightly smaller radius to allow for optimization
            radius = max(0.01, min(0.15, max_radius * 0.9))
            
            circles.append([x, y, radius])
    
    # If we don't have enough circles, add more strategically
    while len(circles) < N_CIRCLES:
        # Place remaining circles near edges or corners
        circles.append([0.5, 0.5, 0.05])
        
    return np.array(circles[:N_CIRCLES])

def create_constraint_functions():
    """Create constraint functions for scipy optimization."""
    def containment_constraint(i):
        def constraint(params):
            # Reshape params into circles array
            circles = params.reshape(-1, 3)
            x, y, r = circles[i]
            # Return positive values when constraint satisfied (>= 0)
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        return constraint
    
    def nonoverlap_constraint(i, j):
        def constraint(params):
            circles = params.reshape(-1, 3)
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            # Return positive when constraint satisfied (distance >= radii sum)
            return dist_sq - (ri + rj)**2
        return constraint
    
    return containment_constraint, nonoverlap_constraint

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and scipy optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize with a better starting configuration
    initial_circles = create_better_initial()
    
    # Prepare constraints
    containment_constraint, nonoverlap_constraint = create_constraint_functions()
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Build constraints list
    constraints = []
    
    # Add containment constraints
    for i in range(N_CIRCLES):
        constraints.append({'type': 'ineq', 'fun': containment_constraint(i)})
    
    # Add non-overlap constraints
    for i in range(N_CIRCLES):
        for j in range(i+1, N_CIRCLES):
            constraints.append({'type': 'ineq', 'fun': nonoverlap_constraint(i, j)})
    
    # Flatten initial circles for optimization
    initial_params = initial_circles.flatten()
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6},
            callback=lambda x: None
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Post-process to ensure validity
            for i in range(N_CIRCLES):
                # Ensure reasonable radii
                optimized_circles[i, 2] = max(0.001, optimized_circles[i, 2])
                # Ensure positions are within bounds
                optimized_circles[i, 0] = np.clip(optimized_circles[i, 0], 
                                                 optimized_circles[i, 2], 
                                                 1 - optimized_circles[i, 2])
                optimized_circles[i, 1] = np.clip(optimized_circles[i, 1], 
                                                 optimized_circles[i, 2], 
                                                 1 - optimized_circles[i, 2])
            return optimized_circles
        else:
            # Return initial configuration if optimization fails
            return initial_circles
            
    except Exception as e:
        # Fallback to initial configuration
        return initial_circles


# EVOLVE-BLOCK-END
