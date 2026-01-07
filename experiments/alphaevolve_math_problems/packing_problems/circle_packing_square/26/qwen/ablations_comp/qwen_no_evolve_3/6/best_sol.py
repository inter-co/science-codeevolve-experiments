# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining grid initialization and gradient-based optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize using a grid-based approach for good starting configuration
    def initialize_grid():
        circles = np.zeros((n, 3))
        # Arrange circles in a grid pattern with some randomness
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        
        # Grid spacing
        grid_size = min(1.0/cols, 1.0/rows)
        padding = grid_size * 0.1  # Small padding to prevent boundary issues
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 0.5) * grid_size
                y = (i + 0.5) * grid_size
                # Add small random perturbation
                x += np.random.uniform(-padding/2, padding/2)
                y += np.random.uniform(-padding/2, padding/2)
                # Ensure within bounds
                x = max(padding, min(1-padding, x))
                y = max(padding, min(1-padding, y))
                circles[idx] = [x, y, min(x, 1-x, y, 1-y) * 0.4]  # Initial radius
                idx += 1
        return circles
    
    # Constraint checking
    def check_constraints(circles):
        # Check containment
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
                return False
        
        # Check non-overlap
        positions = circles[:, :2]
        radii = circles[:, 2]
        distances = cdist(positions, positions)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                if dist < radii[i] + radii[j]:
                    return False
        return True
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        # Reshape flat array back to circles
        circles = circles_flat.reshape((n, 3))
        return -np.sum(circles[:, 2])
    
    # Constraints
    def containment_constraint(circles_flat):
        circles = circles_flat.reshape((n, 3))
        constraints = []
        for i in range(n):
            x, y, r = circles[i]
            # r <= x <= 1-r
            constraints.append(x - r)  # x - r >= 0
            constraints.append(1 - r - x)  # 1 - r - x >= 0
            # r <= y <= 1-r  
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - r - y)  # 1 - r - y >= 0
        return np.array(constraints)
    
    def overlap_constraint(circles_flat):
        circles = circles_flat.reshape((n, 3))
        constraints = []
        positions = circles[:, :2]
        radii = circles[:, 2]
        distances = cdist(positions, positions)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                # We want dist >= radii[i] + radii[j], so we add constraint: dist - radii[i] - radii[j] >= 0
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    # Try multiple initializations to find better solutions
    best_sum = 0
    best_circles = None
    
    for attempt in range(5):
        # Initialize with grid
        circles = initialize_grid()
        
        # Flatten for optimization
        circles_flat = circles.flatten()
        
        # Define bounds for optimization (x, y, r) for each circle
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
        
        # Run optimization
        try:
            # Use SLSQP method which handles constraints well
            result = minimize(
                objective,
                circles_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: containment_constraint(x)},
                    {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
                ],
                options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                optimized_circles = result.x.reshape((n, 3))
                
                # Verify final solution
                if check_constraints(optimized_circles):
                    current_sum = np.sum(optimized_circles[:, 2])
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_circles = optimized_circles.copy()
        except Exception:
            continue
    
    # If we didn't find a valid solution, return the grid initialization
    if best_circles is None:
        best_circles = initialize_grid()
    
    return best_circles


# EVOLVE-BLOCK-END
