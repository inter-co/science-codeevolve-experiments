# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and advanced optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles using a more sophisticated approach
    circles = initialize_better_circles(n)
    
    # Optimize using scipy minimize with constraints
    circles = optimize_circles(circles)
    
    return circles

def initialize_better_circles(n: int) -> np.ndarray:
    """Initialize circle positions using a more sophisticated approach than simple grid."""
    # Start with a hexagonal packing pattern approximation for better initial distribution
    circles = np.zeros((n, 3))
    
    # Calculate how many rows/columns we need approximately
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Create a more even distribution
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    # Adjust spacing to account for circular geometry better
    adjusted_spacing = min(spacing_x, spacing_y) * 0.8
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 1) * spacing_x
        y = (row + 1) * spacing_y
        
        # Apply slight offset for odd rows to create hexagonal pattern
        if row % 2 == 1:
            x += spacing_x * 0.5
            
        # Initial radius - start with a reasonable estimate
        r = adjusted_spacing * 0.3
        
        circles[i] = [x, y, r]
    
    return circles

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate the sum of all circle radii."""
    return np.sum(circles[:, 2])

def create_constraints(n: int) -> list:
    """Create constraint dictionaries for scipy optimization."""
    constraints = []
    
    # Add containment constraints
    for i in range(n):
        def contain_constraint(x, i=i):
            # Extract circle parameters
            ci = x[i*3:i*3+3]
            # Distance to boundaries - return positive when constraint is satisfied
            min_dist = min(ci[0], ci[1], 1-ci[0], 1-ci[1])
            return min_dist - ci[2]
        
        constraints.append({'type': 'ineq', 'fun': contain_constraint})
    
    # Add non-overlap constraints efficiently using vectorized operations
    # We'll create a more efficient constraint evaluation by precomputing indices
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(x, i=i, j=j):
                # Extract circle parameters
                ci = x[i*3:i*3+3]
                cj = x[j*3:j*3+3]
                # Distance between centers minus sum of radii
                dist = np.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
                return dist - (ci[2] + cj[2])
            
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    return constraints

def objective_function(circles_flat: np.ndarray) -> float:
    """Objective function to maximize (negative because scipy minimizes)."""
    # Reshape flat array back to circles
    n = len(circles_flat) // 3
    circles = circles_flat.reshape(n, 3)
    # Negative because we want to maximize sum of radii
    return -np.sum(circles[:, 2])

def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using scipy with better settings."""
    n = initial_circles.shape[0]
    
    # Flatten initial circles for scipy optimization
    initial_flat = initial_circles.flatten()
    
    # Create constraints
    constraints = create_constraints(n)
    
    # Set bounds for each variable (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x coordinate bounds - keep away from edges for stability
        bounds.append((0.001, 0.999))  
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds - tighter upper bound for better convergence
        bounds.append((0.001, 0.4))  
    
    # Try different optimization methods
    methods_to_try = ['SLSQP', 'trust-constr']
    
    for method in methods_to_try:
        try:
            result = minimize(
                objective_function,
                initial_flat,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(n, 3)
                # Ensure valid ranges
                optimized_circles[:, 0] = np.clip(optimized_circles[:, 0], 0.001, 0.999)
                optimized_circles[:, 1] = np.clip(optimized_circles[:, 1], 0.001, 0.999)
                optimized_circles[:, 2] = np.clip(optimized_circles[:, 2], 0.001, 0.4)
                return optimized_circles
        except Exception as e:
            continue
    
    # If all optimization attempts fail, return initial circles
    return initial_circles


# EVOLVE-BLOCK-END
