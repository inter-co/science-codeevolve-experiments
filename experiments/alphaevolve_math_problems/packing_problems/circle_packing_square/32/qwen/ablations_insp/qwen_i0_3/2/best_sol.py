# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach combining spatial initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles using a grid-based approach for good starting configuration
    circles = np.zeros((n, 3))
    
    # Create a grid layout as initial guess
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Distribute points on a grid with some padding
    x_positions = np.linspace(0.1, 0.9, cols)
    y_positions = np.linspace(0.1, 0.9, rows)
    
    # Fill circles array with grid positions and initial radii
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < n:
                circles[idx, 0] = x_positions[j % len(x_positions)]
                circles[idx, 1] = y_positions[i % len(y_positions)]
                # Initial radius: small value to start optimization
                circles[idx, 2] = 0.02
                idx += 1
    
    # Optimization objective function
    def objective(params):
        # Reshape params back to circles array
        circles_flat = params.reshape(-1, 3)
        radii_sum = np.sum(circles_flat[:, 2])
        return -radii_sum  # Negative because we want to maximize
    
    # Constraint functions
    def containment_constraints(params):
        """Ensure all circles are contained within unit square"""
        circles_flat = params.reshape(-1, 3)
        constraints = []
        
        # Each circle's center must be within bounds considering its radius
        for i in range(n):
            x, y, r = circles_flat[i]
            # Left constraint
            constraints.append(x - r)  # x - r >= 0
            # Right constraint  
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            # Bottom constraint
            constraints.append(y - r)  # y - r >= 0
            # Top constraint
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        return np.array(constraints)
    
    def overlap_constraints(params):
        """Ensure no overlapping circles"""
        circles_flat = params.reshape(-1, 3)
        constraints = []
        
        # Check pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_flat[i]
                x2, y2, r2 = circles_flat[j]
                
                # Distance between centers
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # Constraint: distance >= r1 + r2 (non-overlapping)
                constraints.append(dist - r1 - r2)
                
        return np.array(constraints)
    
    # Create initial parameter vector
    initial_params = circles.flatten()
    
    # Set up bounds for optimization (radius must be positive)
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0.001, 0.999))
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds
        bounds.append((0.001, 0.499))
    
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
        {'type': 'ineq', 'fun': lambda p: overlap_constraints(p)}
    ]
    
    try:
        # Run optimization
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return the initial grid configuration if optimization fails
            return circles
            
    except Exception as e:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
