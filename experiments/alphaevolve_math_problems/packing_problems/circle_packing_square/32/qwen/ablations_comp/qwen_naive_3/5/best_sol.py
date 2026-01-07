# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach combining initial hexagonal placement with optimization.
    """
    n = 32
    
    # Create initial configuration using hexagonal packing pattern
    def create_initial_placement():
        # Arrange circles in a hexagonal pattern
        circles = []
        rows = 6
        cols = 6
        
        # Hexagonal grid parameters
        spacing_x = 0.15
        spacing_y = 0.15 * math.sqrt(3) / 2
        
        # Generate points in hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, 0.05])  # Initial radius guess
        
        # Fill remaining slots if needed
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.05])
            
        return np.array(circles[:n])
    
    # Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: each circle must fit completely within unit square
        def boundary_constraint(vars, idx):
            x, y, r = vars[3*idx:3*idx+3]
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        
        # Non-overlap constraints
        def overlap_constraint(vars, i, j):
            x1, y1, r1 = vars[3*i:3*i+3]
            x2, y2, r2 = vars[3*j:3*j+3]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            return dist_sq - (r1 + r2)**2
            
        return boundary_constraint, overlap_constraint
    
    # Objective function to maximize (negative because minimize)
    def objective(vars):
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i + 2]  # radius is third component
        return -total_radius
    
    # Constraints
    def constraint_func(vars):
        # Return array of constraint values (positive means violated)
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            x, y, r = vars[3*i:3*i+3]
            # Each circle must fit within bounds
            constraints.extend([
                x - r,           # left boundary
                1 - x - r,       # right boundary
                y - r,           # bottom boundary
                1 - y - r        # top boundary
            ])
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = vars[3*i:3*i+3]
                x2, y2, r2 = vars[3*j:3*j+3]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # Distance squared should be >= (r1 + r2)^2
                constraints.append(dist_sq - (r1 + r2)**2)
                
        return np.array(constraints)
    
    # Create initial placement
    circles = create_initial_placement()
    
    # Flatten into variables [x1, y1, r1, x2, y2, r2, ...]
    initial_vars = circles.flatten()
    
    # Set bounds for variables (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # r < 0.5 to allow some margin
    
    # Optimize using SLSQP method which handles constraints well
    try:
        # Use scipy minimize with constraints
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_vars = result.x
            circles = optimized_vars.reshape(-1, 3)
        else:
            # If optimization fails, return initial placement
            pass
            
    except Exception as e:
        # Fallback to initial placement if optimization fails
        pass
    
    # Final validation and adjustment
    # Make sure all circles are valid
    for i in range(n):
        x, y, r = circles[i]
        # Ensure valid bounds
        circles[i][0] = max(r, min(1-r, x))
        circles[i][1] = max(r, min(1-r, y))
        circles[i][2] = max(0.001, min(0.499, r))
    
    return circles


# EVOLVE-BLOCK-END
