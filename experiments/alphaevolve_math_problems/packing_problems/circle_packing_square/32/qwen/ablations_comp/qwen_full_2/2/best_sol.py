# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def _initialize_hexagonal_layout(n):
    """
    Initialize circles using a hexagonal lattice pattern for better starting configuration.
    This approach is inspired by the first inspiration program.
    """
    circles = np.zeros((n, 3))
    
    # Create a hexagonal packing pattern - 6 rows and 6 columns for 36 positions, then trim to 32
    rows = 6
    cols = 6
    
    # Calculate spacing based on desired number of circles
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Start with larger initial radius
    max_radius = min(spacing_x, spacing_y) / 2.0
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = 0.0 if i % 2 == 0 else spacing_x / 2.0
            x = (j * spacing_x) + x_offset + max_radius
            y = (i * spacing_y) + max_radius
            
            # Ensure circles fit within bounds
            if x - max_radius >= 0 and x + max_radius <= 1 and y - max_radius >= 0 and y + max_radius <= 1:
                circles[idx] = [x, y, max_radius]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with smaller circles
    while idx < n:
        # Place remaining circles randomly within valid bounds
        x = np.random.uniform(max_radius, 1 - max_radius)
        y = np.random.uniform(max_radius, 1 - max_radius)
        # Small initial radius
        r = max_radius * 0.3
        circles[idx] = [x, y, r]
        idx += 1
        
    return circles

def _evaluate_objective(circles_flat):
    """Evaluate objective function (negative sum of radii)."""
    # Reshape flat array back to circles format
    circles = circles_flat.reshape(-1, 3)
    return -np.sum(circles[:, 2])  # Negative because we minimize

def _get_constraints(n):
    """
    Create constraint functions for optimization.
    This approach is inspired by the first inspiration program but made more efficient.
    """
    def boundary_constraint(i):
        def constraint(x):
            # x[3*i:3*i+2] = (x,y), x[3*i+2] = r
            x_c, y_c, r = x[3*i], x[3*i+1], x[3*i+2]
            # Circle must fit entirely within the square
            return min(r, 1 - r - x_c, 1 - r - y_c, x_c - r, y_c - r)
        return constraint
    
    def overlap_constraint(i, j):
        def constraint(x):
            x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
            x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
            # Distance between centers minus sum of radii
            dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            return dist - (r_i + r_j)
        return constraint
    
    cons = []
    
    # Add boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
    # Add non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
            
    return cons

def _optimize_with_sqp(initial_circles, n):
    """Refine circle positions using Sequential Quadratic Programming."""
    # Flatten initial circles for optimization
    initial_flat = initial_circles.flatten()
    
    # Set bounds for each variable (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y in [0,1], r in [0, 0.5]
    
    # Get constraints
    constraints = _get_constraints(n)
    
    try:
        result = minimize(
            _evaluate_objective,
            initial_flat,
            args=(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        pass
    
    # Return initial if optimization fails
    return initial_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal lattice initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Use hexagonal lattice initialization (from inspiration)
    circles = _initialize_hexagonal_layout(n)
    
    # Refine using optimization
    refined_circles = _optimize_with_sqp(circles, n)
    
    # Final validation and cleanup
    if refined_circles is not None:
        # Ensure final constraints are met
        for i in range(n):
            x, y, r = refined_circles[i]
            # Make sure radius is valid
            if r <= 0:
                refined_circles[i][2] = 0.01
            
            # Make sure circle is contained
            refined_circles[i][0] = np.clip(x, r, 1-r)
            refined_circles[i][1] = np.clip(y, r, 1-r)
    
    return refined_circles


# EVOLVE-BLOCK-END
