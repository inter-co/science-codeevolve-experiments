# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial grid-based placement + local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles array
    circles = np.zeros((n, 3))
    
    # Step 1: Grid-based initialization for good starting configuration
    # Create a coarse grid to distribute initial placements
    grid_size = int(math.ceil(math.sqrt(n)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    # Place initial points on grid
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) < n:
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                positions.append([x, y])
    
    # Assign initial positions and small radii
    for i in range(min(n, len(positions))):
        circles[i][0] = positions[i][0]  # x coordinate
        circles[i][1] = positions[i][1]  # y coordinate
        circles[i][2] = 0.02  # Initial small radius
    
    # Step 2: Optimization using scipy minimize
    # We'll optimize the radii and positions simultaneously
    
    def objective(params):
        # params contains [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i + 2]  # radius is third element in each group
        return -total_radius  # negative because we want to maximize
    
    def constraint_func(params):
        # Check containment and non-overlap constraints
        constraints = []
        
        # Containment constraints: radius <= x <= 1-radius, radius <= y <= 1-radius
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            # x >= r and x <= 1-r
            constraints.append(x - r)  # >= 0
            constraints.append(1 - x - r)  # >= 0
            # y >= r and y <= 1-r
            constraints.append(y - r)  # >= 0
            constraints.append(1 - y - r)  # >= 0
        
        # Non-overlap constraints: distance >= sum of radii
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                # distance >= r1 + r2 (so we want distance - r1 - r2 >= 0)
                constraints.append(distance - r1 - r2)
        
        return np.array(constraints)
    
    # Initial guess from our grid placement
    initial_guess = []
    for i in range(n):
        initial_guess.extend([circles[i][0], circles[i][1], circles[i][2]])
    
    # Define bounds for variables
    bounds = []
    for i in range(n):
        # x bounds: [r, 1-r]
        bounds.append((0.001, 0.999))  # x coordinate
        bounds.append((0.001, 0.999))  # y coordinate
        bounds.append((0.001, 0.499))  # radius (max possible is 0.5)
    
    # Use SLSQP method which handles constraints well
    try:
        result = minimize(objective, 
                         initial_guess,
                         method='SLSQP',
                         bounds=bounds,
                         constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            # Extract optimized values
            for i in range(n):
                circles[i][0] = result.x[3*i]
                circles[i][1] = result.x[3*i+1]
                circles[i][2] = result.x[3*i+2]
        else:
            # Fallback to simple greedy approach if optimization fails
            circles = greedy_placement()
    except Exception:
        # Fallback to simple greedy approach
        circles = greedy_placement()
    
    return circles

def greedy_placement():
    """Fallback method using greedy placement with spacing"""
    n = 32
    circles = np.zeros((n, 3))
    
    # Simple approach: place circles in a grid pattern with some randomness
    grid_size = int(math.ceil(math.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n:
                break
            x = (i + 1) * spacing
            y = (j + 1) * spacing
            
            # Add small random perturbation to avoid perfect grid issues
            x += np.random.uniform(-spacing/8, spacing/8)
            y += np.random.uniform(-spacing/8, spacing/8)
            
            # Ensure it stays within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Set radius to a reasonable value
            radius = min(0.1, 0.5 * min(x, 1-x, y, 1-y))
            circles[idx] = [x, y, radius]
            idx += 1
    
    # Adjust radii to satisfy constraints more carefully
    for i in range(n):
        # Find minimum distance to other circles
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = math.sqrt((circles[i][0] - circles[j][0])**2 + 
                               (circles[i][1] - circles[j][1])**2)
                min_dist = min(min_dist, dist)
        
        # Set radius so that circles don't overlap
        if min_dist > 0:
            max_radius = min_dist / 2.0
            # But also respect boundary constraints
            boundary_radius = min(circles[i][0], 1-circles[i][0], 
                                circles[i][1], 1-circles[i][1])
            circles[i][2] = min(max_radius, boundary_radius, 0.3)
    
    return circles


# EVOLVE-BLOCK-END
