# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: geometric initialization + constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a better hexagonal pattern
    def initialize_hexagonal_layout():
        # Create a hexagonal grid pattern that fits within the unit square
        # More balanced approach than previous attempts
        rows = 6
        cols = 6
        
        # Hexagonal spacing (using sqrt(3)/2 for proper hexagon packing)
        spacing_x = 1.0 / (cols + 1)
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                # Offset every other row for hexagonal packing
                x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                # Only keep points that fit within the unit square
                if 0 <= x <= 1 and 0 <= y <= 1:
                    positions.append([x, y])
            if len(positions) >= n:
                break
        
        # Trim to exact number needed
        positions = positions[:n]
        
        # Start with reasonable initial radii
        radii = [0.05] * n
        
        return np.array(positions), radii
    
    # Initialize
    positions, radii = initialize_hexagonal_layout()
    
    # Create flattened parameter vector [x0, y0, r0, x1, y1, r1, ...]
    def pack_params(positions, radii):
        params = []
        for i in range(n):
            params.extend([positions[i][0], positions[i][1], radii[i]])
        return np.array(params)
    
    def unpack_params(params):
        positions = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            positions[i][0] = params[3*i]
            positions[i][1] = params[3*i+1]
            radii[i] = params[3*i+2]
        return positions, radii
    
    # Objective function: minimize negative sum of radii (equivalent to maximizing sum)
    def objective(params):
        _, radii = unpack_params(params)
        return -np.sum(radii)
    
    # Constraint functions
    def create_constraints():
        cons = []
        
        # Boundary constraints: each circle must be within unit square
        def boundary_constraint(i):
            def constraint(params):
                x, y, r = params[3*i], params[3*i+1], params[3*i+2]
                # r <= x <= 1-r and r <= y <= 1-r
                return np.array([
                    x - r,           # x >= r
                    1 - x - r,       # x <= 1-r
                    y - r,           # y >= r
                    1 - y - r        # y <= 1-r
                ])
            return constraint
        
        # Non-overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(i, j):
            def constraint(params):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                # Distance squared between centers
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # Non-overlap condition: dist^2 >= (r1 + r2)^2
                return dist_sq - (r1 + r2)**2
            return constraint
        
        # Add boundary constraints for all circles
        for i in range(n):
            # Add four boundary constraints per circle (x >= r, x <= 1-r, y >= r, y <= 1-r)
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
            
        # Add non-overlap constraints - use a more manageable subset to reduce computation
        # Instead of all pairs, use a smart sampling approach
        for i in range(n):
            for j in range(i+1, min(i+8, n)):  # Limit to nearby circles to reduce constraint count
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
                
        return cons
    
    # Initial guess
    initial_params = pack_params(positions, radii)
    
    # Create constraints
    constraints = create_constraints()
    
    # Bounds for variables (x, y, r)
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n  # x, y in [0,1], r in [0, 0.5]
    
    # Optimize using SLSQP
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6},
            tol=1e-6
        )
        
        if result.success:
            final_positions, final_radii = unpack_params(result.x)
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_positions[i][0], final_positions[i][1], final_radii[i]]
            return circles
    except Exception as e:
        pass
    
    # Fallback to initial configuration
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [positions[i][0], positions[i][1], radii[i]]
    return circles


# EVOLVE-BLOCK-END
