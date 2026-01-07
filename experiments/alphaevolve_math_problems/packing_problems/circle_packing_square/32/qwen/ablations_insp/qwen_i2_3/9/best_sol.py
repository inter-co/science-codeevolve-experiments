# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Initialize with a hexagonal grid pattern for good starting configuration
    def initialize_hexagonal_grid():
        # Create a hexagonal grid pattern
        rows = 6
        cols = 6
        circles = []
        
        # Hexagonal spacing
        spacing_x = 0.9 / cols  # Leave some margin
        spacing_y = 0.9 / rows  # Leave some margin
        hex_height = spacing_y * math.sqrt(3) / 2
        
        # Place circles in a hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                y = 0.05 + (i + 0.5) * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                circles.append([x, y, min(spacing_x, spacing_y) / 3])
        
        # Fill remaining circles with random positions near center
        while len(circles) < n:
            x = 0.5 + np.random.normal(0, 0.1)
            y = 0.5 + np.random.normal(0, 0.1)
            r = 0.05 + np.random.random() * 0.05
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Better initialization using a more systematic approach
    def initialize_better_grid():
        circles = []
        # Create a more systematic grid that fills the space better
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing_x = 0.9 / grid_size
        spacing_y = 0.9 / grid_size
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                y = 0.05 + (i + 0.5) * spacing_y
                # Initial radius - small enough to fit in grid cell
                r = min(spacing_x, spacing_y) * 0.3
                
                circles.append([x, y, r])
                idx += 1
            if idx >= n:
                break
                
        # Pad with random circles if needed
        while len(circles) < n:
            x = 0.5 + np.random.normal(0, 0.1)
            y = 0.5 + np.random.normal(0, 0.1)
            r = 0.05 + np.random.random() * 0.05
            circles.append([x, y, r])
                
        return np.array(circles)
    
    # Improved constraint functions based on INSPIRATION PROGRAM 2
    def contain_constraints(params):
        """Ensure all circles stay within the unit square"""
        constraints = []
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            # Circle must fit within bounds: x-r >= 0, y-r >= 0, x+r <= 1, y+r <= 1
            constraints.extend([
                x - r,      # x - r >= 0
                y - r,      # y - r >= 0  
                1 - x - r,  # 1 - x - r >= 0
                1 - y - r   # 1 - y - r >= 0
            ])
        return np.array(constraints)
    
    def overlap_constraints(params):
        """Ensure no two circles overlap"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                # Distance between centers should be >= sum of radii
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(distance - (r1 + r2))  # Should be >= 0
        return np.array(constraints)
    
    # Try multiple optimization attempts with different starting configurations
    best_solution = None
    best_sum = 0
    
    # Try 3 different optimization attempts with different seeds and strategies
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        
        # Try different initialization strategies
        for init_name, init_func in [("hex", initialize_hexagonal_grid), ("grid", initialize_better_grid)]:
            try:
                circles = init_func()
                
                # Flatten initial parameters [x1,y1,r1,x2,y2,r2,...]
                initial_params = circles.flatten()
                
                # Objective function to maximize sum of radii (minimize negative sum)
                def objective(params):
                    total_radius = 0
                    for i in range(n):
                        total_radius += params[3*i + 2]  # radii are at indices 2,5,8,...
                    return -total_radius  # negative because we want to maximize
                
                # Set up bounds: x,y in [0,1], r in [0.001, 0.5] to avoid numerical issues
                bounds = []
                for i in range(n):
                    bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])
                
                # Set up constraints for scipy.optimize - cleaner approach
                constraints = [
                    {'type': 'ineq', 'fun': lambda p: contain_constraints(p)},
                    {'type': 'ineq', 'fun': lambda p: overlap_constraints(p)}
                ]
                
                # Use scipy's minimize with SLSQP method for better constraint handling
                # Increase maxiter for better convergence
                result = minimize(
                    objective, 
                    initial_params, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
                )
                
                if result.success:
                    # Extract sum of radii
                    current_sum = -objective(result.x)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_solution = result.x
                        
            except Exception as e:
                continue
    
    # If optimization didn't work well, fallback to initial configuration
    if best_solution is None:
        circles = initialize_hexagonal_grid()
        best_solution = circles.flatten()
    
    # Extract final circle positions and radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [best_solution[3*i], best_solution[3*i+1], best_solution[3*i+2]]
    
    # Final validation and adjustment to ensure all constraints are satisfied
    for i in range(n):
        x, y, r = circles[i]
        # Ensure containment constraints
        max_r = min(x, y, 1-x, 1-y)
        circles[i][2] = min(r, max_r)
        
        # Ensure center stays within bounds
        circles[i][0] = max(r, min(1-r, x))
        circles[i][1] = max(r, min(1-r, y))
    
    return circles


# EVOLVE-BLOCK-END
