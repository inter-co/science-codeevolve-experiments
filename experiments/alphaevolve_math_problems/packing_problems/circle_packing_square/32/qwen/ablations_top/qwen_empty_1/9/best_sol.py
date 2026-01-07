# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Improved initialization with better grid-based approach
    def initialize_better():
        circles = []
        # Use a more careful grid approach
        rows = 6
        cols = 6
        
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Create grid points with better spacing and initial radii
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Offset odd rows for better packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                # Add slight randomness to positions
                x += np.random.uniform(-spacing_x*0.03, spacing_x*0.03)
                y += np.random.uniform(-spacing_y*0.03, spacing_y*0.03)
                
                # Reasonable initial radius based on spacing
                radius = min(spacing_x, spacing_y) * 0.35
                
                # Add variation to radius with more controlled randomness
                radius *= (0.8 + np.random.random() * 0.4)
                
                circles.append([x, y, radius])
        
        # Fill remaining positions with strategic random placement
        while len(circles) < n:
            # Place in a way that avoids clustering
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = 0.02 + np.random.random() * 0.025  # More controlled initial size
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Initialize with better configuration
    circles = initialize_better()
    
    # Define objective function: negative sum of radii (we want to maximize)
    def objective(params):
        total_radius = 0
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            total_radius += r
        return -total_radius  # Negative because we're minimizing
    
    # Define constraints
    def constraint_containment(i):
        def func(params):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            # Circle must be fully inside unit square
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        return func
    
    def constraint_overlap(i, j):
        def func(params):
            x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
            x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
            # Distance between centers minus sum of radii
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            return dist - r1 - r2
        return func
    
    # Build constraints list - keep all constraints for better accuracy
    constraints = []
    
    # Add containment constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': constraint_containment(i)})
    
    # Add overlap constraints (all pairs for precision)
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_overlap(i, j)})
    
    # Bounds for parameters: [x, y, r] for each circle
    bounds = []
    for i in range(n):
        # x, y in [0,1], r in [0,0.5] (reasonable upper bound)
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
    
    # Optimization with tuned parameters - reduced iterations for stability
    try:
        # Flatten initial guess
        x0 = circles.flatten()
        
        # Run optimization with tuned parameters
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-6, 'iprint': 0}
        )
        
        # Extract optimized solution if successful
        if result.success:
            optimized_params = result.x
            circles = np.array([
                [optimized_params[3*i], optimized_params[3*i+1], optimized_params[3*i+2]] 
                for i in range(n)
            ])
            
    except Exception as e:
        # If optimization fails, return the initial solution
        pass
    
    return circles


# EVOLVE-BLOCK-END
