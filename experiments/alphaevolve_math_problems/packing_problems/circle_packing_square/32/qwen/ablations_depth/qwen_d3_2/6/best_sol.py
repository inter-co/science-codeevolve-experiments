# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: geometric initialization + constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Geometric initialization using hexagonal packing pattern
    def initialize_hexagonal_layout():
        # Create a hexagonal grid pattern that fits in unit square
        # For 32 circles, we can arrange in roughly 6x6 grid with some empty spaces
        rows = 6
        cols = 6
        padding = 0.05  # Space around edges
        
        # Calculate spacing based on number of circles needed
        spacing_x = (1 - 2*padding) / (cols - 1) if cols > 1 else 0.5
        spacing_y = (1 - 2*padding) / (rows - 1) if rows > 1 else 0.5
        
        # Adjust spacing so that we don't exceed 32 circles
        actual_circles = min(rows * cols, n)
        
        centers = []
        radii = []
        
        # Generate centers in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(centers) >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = spacing_x * j
                y_offset = spacing_y * i
                
                # Apply hexagonal offset
                if i % 2 == 1:
                    x_offset += spacing_x / 2
                    
                x = padding + x_offset
                y = padding + y_offset
                
                # Ensure center is within bounds
                if x >= 0 and x <= 1 and y >= 0 and y <= 1:
                    centers.append([x, y])
                    
        # Set initial radii - start with small values that fit in the square
        initial_radii = []
        for i, (x, y) in enumerate(centers):
            # Maximum possible radius at this position
            max_radius = min(x, 1-x, y, 1-y)
            # Start with a fraction of maximum radius
            initial_radii.append(max_radius * 0.3)
            
        # Trim to exact count
        centers = centers[:n]
        initial_radii = initial_radii[:n]
        
        return np.array(centers), initial_radii
    
    # Phase 2: Optimization using scipy minimize
    def compute_objective_and_constraints():
        # Define constraint functions
        def get_constraints():
            cons = []
            
            # Boundary constraints: radius must be such that circle stays within square
            def boundary_constraint(i):
                def func(params):
                    x, y, r = params[3*i:3*i+3]
                    # Circle must stay within bounds
                    return min(x - r, 1 - x - r, y - r, 1 - y - r)
                return {'type': 'ineq', 'fun': func}
            
            # Non-overlap constraints
            def non_overlap_constraint(i, j):
                def func(params):
                    x1, y1, r1 = params[3*i:3*i+3]
                    x2, y2, r2 = params[3*j:3*j+3]
                    # Distance between centers minus sum of radii should be >= 0
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    return dist - (r1 + r2)
                return {'type': 'ineq', 'fun': func}
            
            # Add boundary constraints for all circles
            for i in range(n):
                cons.append(boundary_constraint(i))
                
            # Add non-overlap constraints for all pairs
            for i in range(n):
                for j in range(i+1, n):
                    cons.append(non_overlap_constraint(i, j))
                    
            return cons
        
        def objective(params):
            # Sum of all radii (we want to maximize)
            total_radius = 0
            for i in range(n):
                total_radius += params[3*i + 2]  # radius is third component
            return -total_radius  # negative because we're minimizing
            
        return objective, get_constraints
    
    # Initialize with hexagonal layout
    centers, initial_radii = initialize_hexagonal_layout()
    
    # Flatten initial guess for optimization
    initial_guess = []
    for i in range(n):
        initial_guess.extend([centers[i][0], centers[i][1], initial_radii[i]])
    
    # Get objective and constraints
    objective, get_constraints = compute_objective_and_constraints()
    constraints = get_constraints()
    
    # Perform optimization
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6},
            tol=1e-6
        )
        
        if result.success:
            # Extract final positions and radii
            final_params = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [
                    final_params[3*i],      # x coordinate
                    final_params[3*i+1],    # y coordinate
                    final_params[3*i+2]     # radius
                ]
            return circles
        else:
            # If optimization fails, return the initial hexagonal layout
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [centers[i][0], centers[i][1], initial_radii[i]]
            return circles
    except Exception as e:
        # Fallback to initial layout if optimization fails
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [centers[i][0], centers[i][1], initial_radii[i]]
        return circles


# EVOLVE-BLOCK-END
