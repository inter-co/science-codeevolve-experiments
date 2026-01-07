# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach: grid initialization + constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Grid-based initialization to get a good starting configuration
    # Create a grid pattern that roughly fits 32 circles
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Initialize positions in a grid pattern
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            positions.append([x, y])
    
    # Adjust positions to be within bounds and distribute better
    adjusted_positions = []
    for i, (x, y) in enumerate(positions):
        # Keep positions within the unit square with some margin
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        adjusted_positions.append([x, y])
    
    # Initial radii estimation based on spacing
    initial_radii = []
    for i in range(n):
        # Estimate initial radius based on minimum distance to neighbors
        min_dist = float('inf')
        x, y = adjusted_positions[i]
        
        # Check distance to all other positions
        for j in range(n):
            if i != j:
                x2, y2 = adjusted_positions[j]
                dist = math.sqrt((x - x2)**2 + (y - y2)**2)
                min_dist = min(min_dist, dist)
        
        # Set radius to half of minimum distance to neighbors, capped at 0.2
        radius = min(0.2, min_dist / 2.0)
        initial_radii.append(max(0.01, radius))  # Ensure minimum radius
    
    # Flatten initial parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = []
    for i in range(n):
        initial_params.extend([adjusted_positions[i][0], adjusted_positions[i][1], initial_radii[i]])
    
    # Define constraint functions
    def containment_constraints(params):
        """Ensure all circles are within the unit square"""
        constraints = []
        for i in range(n):
            x_idx = 3*i
            y_idx = 3*i + 1
            r_idx = 3*i + 2
            
            x = params[x_idx]
            y = params[y_idx]
            r = params[r_idx]
            
            # Circle must be fully contained
            constraints.extend([
                r - x,           # x - r >= 0
                r - y,           # y - r >= 0
                1 - r - x,       # 1 - x - r >= 0
                1 - r - y        # 1 - y - r >= 0
            ])
        return np.array(constraints)
    
    def overlap_constraints(params):
        """Ensure no overlapping circles"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1_idx = 3*i
                y1_idx = 3*i + 1
                r1_idx = 3*i + 2
                x2_idx = 3*j
                y2_idx = 3*j + 1
                r2_idx = 3*j + 2
                
                x1, y1, r1 = params[x1_idx], params[y1_idx], params[r1_idx]
                x2, y2, r2 = params[x2_idx], params[y2_idx], params[r2_idx]
                
                # Distance between centers
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # Minimum distance to avoid overlap
                min_dist_sq = (r1 + r2)**2
                
                # Constraint: distance^2 >= (r1 + r2)^2 (non-overlapping)
                constraints.append(min_dist_sq - dist_sq)
        return np.array(constraints)
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        total_radius = 0
        for i in range(n):
            r_idx = 3*i + 2
            total_radius += params[r_idx]
        return -total_radius
    
    # Combine all constraints
    def combined_constraints(params):
        return np.concatenate([containment_constraints(params), overlap_constraints(params)])
    
    # Set up bounds: x, y in [0,1], r in [0,0.5] (conservative upper bound)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Use scipy's minimize with SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': combined_constraints},
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        # Extract final solution
        final_params = result.x
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
        
        # Ensure all circles are valid
        for i in range(n):
            x, y, r = circles[i]
            if x < r or y < r or x > 1-r or y > 1-r:
                # Revert to initial guess if invalid
                circles[i] = [adjusted_positions[i][0], adjusted_positions[i][1], initial_radii[i]]
                
        return circles
    
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [adjusted_positions[i][0], adjusted_positions[i][1], initial_radii[i]]
        return circles


# EVOLVE-BLOCK-END
