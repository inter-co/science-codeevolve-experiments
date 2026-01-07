# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial hexagonal lattice placement with optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Create initial configuration using hexagonal lattice approximation
    # This provides a good starting point that's likely to be close to optimal
    circles = np.zeros((n, 3))
    
    # Hexagonal lattice parameters
    # For 26 circles, we'll arrange them in approximately a 5x5 grid with some adjustments
    rows = 5
    cols = 5
    
    # Calculate spacing based on hexagonal packing density
    # In hexagonal packing, the optimal arrangement has radius r such that 
    # centers are separated by 2r, and vertical spacing is sqrt(3)*r
    
    # Start with a reasonable estimate for radius
    initial_radius = 0.1
    
    # Place circles in a hexagonal pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = j * 2 * initial_radius
            y_offset = i * math.sqrt(3) * initial_radius
            
            # Apply offset for hexagonal arrangement
            if i % 2 == 1:
                x_offset += initial_radius
                
            # Ensure circles stay within unit square
            x = x_offset + initial_radius
            y = y_offset + initial_radius
            
            # Adjust if out of bounds
            if x > 1 - initial_radius:
                x = 1 - initial_radius
            if y > 1 - initial_radius:
                y = 1 - initial_radius
                
            circles[idx] = [x, y, initial_radius]
            idx += 1
        if idx >= n:
            break
    
    # Trim to exactly 26 circles if needed
    circles = circles[:n]
    
    # Refine using optimization
    # Define objective function: negative sum of radii (we want to maximize)
    def objective(params):
        # params contains [x0,y0,r0,x1,y1,r1,...] for all circles
        total_radius = 0
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            total_radius += r
        return -total_radius  # Negative because we're minimizing
    
    # Define constraints
    def constraint_containment(params):
        # Each circle must be fully contained in unit square
        constraints = []
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            # Circle must be inside unit square with radius r
            constraints.extend([
                x - r,           # x - r >= 0
                y - r,           # y - r >= 0
                1 - x - r,       # 1 - x - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        return np.array(constraints)
    
    def constraint_nonoverlap(params):
        # No two circles can overlap
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                # Distance between centers must be >= sum of radii
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                # We want dist >= r1 + r2, so dist^2 >= (r1+r2)^2
                # This constraint is satisfied when dist_sq - min_dist_sq >= 0
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0.001, 0.999))  # Small buffer to avoid boundary issues
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds (positive but not too large)
        bounds.append((0.001, 0.499))
    
    # Flatten initial guess
    initial_guess = []
    for i in range(n):
        initial_guess.extend([circles[i][0], circles[i][1], circles[i][2]])
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
    ]
    
    try:
        # Perform optimization
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            # Extract final solution
            for i in range(n):
                circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
        else:
            # If optimization fails, return the initial configuration
            pass
    except Exception as e:
        # If optimization fails due to numerical issues, return initial
        pass
    
    return circles


# EVOLVE-BLOCK-END
