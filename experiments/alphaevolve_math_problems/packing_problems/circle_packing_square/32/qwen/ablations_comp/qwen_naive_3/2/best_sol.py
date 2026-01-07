# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining Voronoi-based initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Step 1: Generate initial configuration using a grid-based approach with some randomness
    # Create a grid of points and perturb them slightly
    grid_size = int(np.ceil(np.sqrt(n)))
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i + 0.5) / grid_size
            y = (j + 0.5) / grid_size
            # Add small random perturbation
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            points.append([x, y])
    
    points = np.array(points[:n])
    
    # Step 2: Initialize radii to be small but feasible
    radii = np.full(n, 0.02)
    
    # Step 3: Define constraint functions
    def get_constraints():
        cons = []
        
        # Boundary constraints: radius <= x <= 1-radius and radius <= y <= 1-radius
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})  # x >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})  # y >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})  # 1-x >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})  # 1-y >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2]})  # r >= 0
            
        # Circle-to-circle distance constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) - (x[3*i+2] + x[3*j+2])
                })
                
        return cons
    
    # Step 4: Objective function to maximize sum of radii (minimize negative sum)
    def objective(x):
        return -np.sum(x[2::3])  # Negative because we're minimizing
    
    # Step 5: Initial guess: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = points[i][0]  # x coordinate
        x0[3*i+1] = points[i][1]  # y coordinate
        x0[3*i+2] = radii[i]      # radius
    
    # Step 6: Set up constraints
    constraints = get_constraints()
    
    # Step 7: Bounds for variables (x, y, r) - all must be positive and within square
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Step 8: Optimization with multiple restarts for better results
    best_result = None
    best_sum = -np.inf
    
    # Try multiple random initializations
    for _ in range(5):
        # Slightly randomize initial values
        x0_local = x0.copy()
        for i in range(n):
            x0_local[3*i] += np.random.normal(0, 0.01)
            x0_local[3*i+1] += np.random.normal(0, 0.01)
            x0_local[3*i+2] += np.random.normal(0, 0.005)
            
        # Ensure bounds are respected
        for i in range(n):
            x0_local[3*i] = np.clip(x0_local[3*i], 0.001, 0.999)
            x0_local[3*i+1] = np.clip(x0_local[3*i+1], 0.001, 0.999)
            x0_local[3*i+2] = np.clip(x0_local[3*i+2], 0.001, 0.499)
            
        try:
            # Use SLSQP method which handles constraints well
            result = minimize(objective, x0_local, method='SLSQP', bounds=bounds, 
                            constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-6})
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except:
            continue
    
    # If no successful optimization, return the simple initial configuration
    if best_result is None:
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [points[i][0], points[i][1], radii[i]]
        return circles
    
    # Extract final solution
    final_x = best_result.x
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [final_x[3*i], final_x[3*i+1], final_x[3*i+2]]
    
    return circles


# EVOLVE-BLOCK-END
