# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal packing initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using hexagonal packing pattern for good starting configuration
    def initialize_hexagonal():
        circles = np.zeros((n, 3))
        
        # Hexagonal packing parameters
        sqrt3 = math.sqrt(3)
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        
        # Calculate spacing based on hexagonal arrangement
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        min_spacing = min(spacing_x, spacing_y)
        
        # Place circles in hexagonal pattern
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                    
                # Offset every other row for hexagonal packing
                x_offset = (i % 2) * spacing_x * 0.5
                x = (j + 0.5) * spacing_x + x_offset
                y = (i + 0.5) * spacing_y
                
                # Ensure we're within bounds
                if x < 0.5 * min_spacing or x > 1 - 0.5 * min_spacing:
                    continue
                if y < 0.5 * min_spacing or y > 1 - 0.5 * min_spacing:
                    continue
                    
                # Initial radius as small as possible while maintaining containment
                r = min_spacing * 0.4
                
                circles[idx] = [x, y, r]
                idx += 1
                
                if idx >= n:
                    break
                    
        return circles
    
    # Create initial configuration
    circles = initialize_hexagonal()
    
    # Extract positions and radii for optimization
    initial_positions = circles[:, :2].flatten()
    initial_radii = circles[:, 2]
    
    # Combine into single vector for optimization
    initial_params = np.concatenate([initial_positions, initial_radii])
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must stay within the unit square
        def boundary_constraint(params):
            positions = params[:2*n].reshape(-1, 2)
            radii = params[2*n:]
            
            # Each circle center must be within bounds considering radius
            # x - r >= 0 and x + r <= 1
            # y - r >= 0 and y + r <= 1
            constraints = []
            
            for i in range(n):
                x, y = positions[i]
                r = radii[i]
                
                # x - r >= 0
                constraints.append(x - r)
                # 1 - x - r >= 0
                constraints.append(1 - x - r)
                # y - r >= 0
                constraints.append(y - r)
                # 1 - y - r >= 0
                constraints.append(1 - y - r)
                
            return np.array(constraints)
        
        cons.append({'type': 'ineq', 'fun': boundary_constraint})
        
        # Non-overlap constraints
        def overlap_constraint(params):
            positions = params[:2*n].reshape(-1, 2)
            radii = params[2*n:]
            
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    r1 = radii[i]
                    r2 = radii[j]
                    
                    # Distance between centers minus sum of radii should be >= 0
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist_sq = (r1 + r2)**2
                    
                    # We want dist_sq >= min_dist_sq (non-overlapping)
                    # So we return dist_sq - min_dist_sq for inequality constraint
                    constraints.append(dist_sq - min_dist_sq)
                    
            return np.array(constraints)
        
        cons.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return cons
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(params):
        return -np.sum(params[2*n:])  # Negative because minimize
    
    # Get constraints
    constraints = get_constraints()
    
    # Optimization with bounds
    bounds = []
    
    # Add bounds for positions (0 to 1)
    for i in range(2*n):
        bounds.append((0, 1))
    
    # Add bounds for radii (small positive values to avoid numerical issues)
    for i in range(n):
        bounds.append((0.001, 0.5))  # Radii bounded between 0.001 and 0.5
    
    # Perform optimization
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            final_positions = result.x[:2*n].reshape(-1, 2)
            final_radii = result.x[2*n:]
            
            # Update circles with optimized values
            circles = np.column_stack([final_positions, final_radii])
        else:
            # If optimization fails, return the initial configuration
            pass
            
    except Exception as e:
        # If optimization fails due to any reason, return initial configuration
        pass
    
    return circles


# EVOLVE-BLOCK-END
