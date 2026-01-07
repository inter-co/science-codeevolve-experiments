# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach: initial hexagonal packing + constrained optimization
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    n = 26
    
    # Initialize with a more precise hexagonal packing pattern as starting point
    def initialize_hexagonal_packing():
        # Create a hexagonal grid pattern that fits in the unit square
        circles = []
        
        # Use a precise hexagonal lattice approach
        # For 26 circles, we want to fill as much area as possible with minimal gaps
        
        # Try 5x5 grid with hexagonal offset (most promising)
        rows = 5
        cols = 5
        radius_guess = 0.08
        
        # Place circles in a hexagonal pattern with proper spacing
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for better packing
                x = j * 2 * radius_guess + (i % 2) * radius_guess
                y = i * math.sqrt(3)/2 * 2 * radius_guess
                
                # Check if this fits within the unit square with safety margin
                if x + radius_guess <= 1 and y + radius_guess <= 1:
                    circles.append([x + radius_guess, y + radius_guess, radius_guess])
        
        # If we didn't get enough circles, fill with random valid ones
        while len(circles) < n:
            # Try to place circles with more careful consideration of existing ones
            x = np.random.uniform(radius_guess, 1 - radius_guess)
            y = np.random.uniform(radius_guess, 1 - radius_guess)
            
            # Find maximum radius at this location
            max_radius = min(x, 1-x, y, 1-y)
            
            # Check overlap with existing circles
            valid = True
            for cx, cy, cr in circles:
                dist = math.sqrt((cx-x)**2 + (cy-y)**2)
                if dist < max_radius + cr:
                    valid = False
                    break
            
            if valid and max_radius > 0.001:
                circles.append([x, y, max_radius])
        
        return np.array(circles[:n])
    
    # Initialize
    circles = initialize_hexagonal_packing()
    
    # Optimization function
    def objective(params):
        # params: [x0,y0,r0,x1,y1,r1,...,x25,y25,r25]
        # Extract positions and radii
        positions = params.reshape(-1, 3)[:, :2]
        radii = params.reshape(-1, 3)[:, 2]
        
        # Maximize sum of radii (minimize negative)
        return -np.sum(radii)
    
    def constraint_containment(params):
        # Ensure all circles are within the unit square
        positions = params.reshape(-1, 3)[:, :2]
        radii = params.reshape(-1, 3)[:, 2]
        
        # Each circle's center must be at least radius away from edges
        constraints = []
        
        # x constraints - center must be at least radius away from edges
        constraints.extend(positions[:, 0] - radii)  # x - r >= 0
        constraints.extend(1 - radii - positions[:, 0])  # 1 - r - x >= 0
        
        # y constraints  
        constraints.extend(positions[:, 1] - radii)  # y - r >= 0
        constraints.extend(1 - radii - positions[:, 1])  # 1 - r - y >= 0
        
        return np.array(constraints)
    
    def constraint_nonoverlap(params):
        # Ensure no two circles overlap
        positions = params.reshape(-1, 3)[:, :2]
        radii = params.reshape(-1, 3)[:, 2]
        
        constraints = []
        
        # Compute pairwise distances efficiently
        distances = cdist(positions, positions)
        
        # For each pair of circles, enforce that distance >= sum of radii
        for i in range(n):
            for j in range(i+1, n):
                # Distance between centers minus sum of radii should be >= 0
                dist = distances[i, j]
                sum_radii = radii[i] + radii[j]
                constraints.append(dist - sum_radii)
                
        return np.array(constraints)
    
    # Create parameter vector
    initial_params = circles.flatten()
    
    # Define bounds for parameters - tighter bounds for better optimization
    bounds = []
    for i in range(n):
        # x coordinates: [radius + 0.001, 1 - radius - 0.001] 
        bounds.append((circles[i, 2] + 0.001, 1 - circles[i, 2] - 0.001))
        # y coordinates: [radius + 0.001, 1 - radius - 0.001]
        bounds.append((circles[i, 2] + 0.001, 1 - circles[i, 2] - 0.001))
        # radii: [0.001, 0.5] (reasonable range)
        bounds.append((0.001, 0.5))
    
    # Create constraints
    cons = []
    
    # Add containment constraints
    cons.append({'type': 'ineq', 'fun': lambda p: constraint_containment(p)})
    
    # Add non-overlap constraints
    cons.append({'type': 'ineq', 'fun': lambda p: constraint_nonoverlap(p)})
    
    # Optimize using trust-constr method which is more robust than SLSQP
    try:
        result = minimize(
            objective,
            initial_params,
            method='trust-constr',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'xtol': 1e-8, 'gtol': 1e-8, 'verbose': 0}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return initial configuration if optimization fails
            return circles
            
    except Exception as e:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
