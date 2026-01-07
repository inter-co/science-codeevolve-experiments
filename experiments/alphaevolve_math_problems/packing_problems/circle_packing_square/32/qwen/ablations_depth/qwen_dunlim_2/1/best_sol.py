# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import random
from itertools import combinations

# Global constants
N_CIRCLES = 32
MAX_RADIUS = 0.5

def initialize_grid_placement():
    """Initialize circle positions using grid placement for good initial distribution"""
    # Create a hexagonal grid pattern for better initial spacing
    rows = int(np.ceil(np.sqrt(N_CIRCLES)))
    cols = int(np.ceil(N_CIRCLES / rows))
    
    # Create grid points with some jitter
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= N_CIRCLES:
                break
            x = 0.1 + 0.8 * j / (cols - 1) if cols > 1 else 0.5
            y = 0.1 + 0.8 * i / (rows - 1) if rows > 1 else 0.5
            # Add slight jitter
            x += random.uniform(-0.02, 0.02)
            y += random.uniform(-0.02, 0.02)
            points.append([x, y])
    
    # Trim to exactly N_CIRCLES
    points = points[:N_CIRCLES]
    
    # Initialize with equal small radii
    circles = np.zeros((N_CIRCLES, 3))
    for i, point in enumerate(points):
        circles[i] = [point[0], point[1], 0.03]
    
    return circles

def calculate_feasible_radii(circles):
    """Calculate maximum feasible radius for each circle given current configuration"""
    n = len(circles)
    max_radii = np.zeros(n)
    
    for i in range(n):
        x_i, y_i, _ = circles[i]
        
        # Distance to boundaries
        boundary_dist = min(x_i, y_i, 1-x_i, 1-y_i)
        
        # Distance to other circles
        min_dist_to_others = float('inf')
        for j in range(n):
            if i != j:
                x_j, y_j, _ = circles[j]
                dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                min_dist_to_others = min(min_dist_to_others, dist)
        
        # Maximum radius is limited by both boundaries and other circles
        max_radii[i] = min(boundary_dist, min_dist_to_others/2.0)
    
    return max_radii

def is_valid_configuration(circles, tolerance=1e-8):
    """Check if the current configuration is valid (no overlaps, within bounds)"""
    n = len(circles)
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2 - tolerance:
                return False
    
    return True

def objective_function(params, circles):
    """Objective function to maximize sum of radii"""
    # Update circles with new parameters
    for i in range(N_CIRCLES):
        circles[i, 0] = params[3*i]     # x coordinate
        circles[i, 1] = params[3*i+1]   # y coordinate  
        circles[i, 2] = params[3*i+2]   # radius
    
    # Calculate sum of radii (negative because we minimize)
    return -np.sum(circles[:, 2])

def constraint_boundaries(params, circles):
    """Constraint function for boundary limits"""
    for i in range(N_CIRCLES):
        x = params[3*i]
        y = params[3*i+1]
        r = params[3*i+2]
        # x - r >= 0
        yield x - r
        # x + r <= 1
        yield 1 - x - r
        # y - r >= 0
        yield y - r
        # y + r <= 1
        yield 1 - y - r

def constraint_overlaps(params, circles):
    """Constraint function for overlap avoidance"""
    for i, j in combinations(range(N_CIRCLES), 2):
        x1 = params[3*i]
        y1 = params[3*i+1]
        r1 = params[3*i+2]
        x2 = params[3*j]
        y2 = params[3*j+1]
        r2 = params[3*j+2]
        
        # Distance between centers minus sum of radii should be >= 0
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        yield distance - r1 - r2

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a sequential quadratic programming approach with proper constraint handling.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Start with grid initialization
    circles = initialize_grid_placement()
    
    # Ensure we have a valid starting configuration
    if not is_valid_configuration(circles):
        # If invalid, start with simpler configuration
        circles = np.zeros((N_CIRCLES, 3))
        for i in range(N_CIRCLES):
            circles[i] = [0.1 + 0.8 * (i % 8) / 7, 0.1 + 0.8 * (i // 8) / 3, 0.03]
    
    # Prepare optimization variables
    initial_params = []
    bounds = []
    
    for i in range(N_CIRCLES):
        x, y, r = circles[i]
        initial_params.extend([x, y, r])
        
        # Bounds: x in [r, 1-r], y in [r, 1-r], r in [0, min(0.5, boundary_distance)]
        bounds.append((r, 1-r))      # x bound
        bounds.append((r, 1-r))      # y bound
        bounds.append((r, 0.5))      # r bound (cap at 0.5)
    
    # Define constraints
    constraints = []
    
    # Boundary constraints
    for i in range(N_CIRCLES):
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, idx=i: p[3*idx] - p[3*idx+2]  # x - r >= 0
        })
        constraints.append({
            'type': 'ineq', 
            'fun': lambda p, idx=i: 1 - p[3*idx] - p[3*idx+2]  # 1 - x - r >= 0
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, idx=i: p[3*idx+1] - p[3*idx+2]  # y - r >= 0
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, idx=i: 1 - p[3*idx+1] - p[3*idx+2]  # 1 - y - r >= 0
        })
    
    # Overlap constraints
    for i, j in combinations(range(N_CIRCLES), 2):
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i, j=j: np.sqrt((p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2) - p[3*i+2] - p[3*j+2]
        })
    
    # Optimization parameters
    options = {
        'maxiter': 1000,
        'ftol': 1e-6,
        'gtol': 1e-6
    }
    
    try:
        # Perform optimization
        result = minimize(
            objective_function,
            initial_params,
            args=(circles,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=1e-6
        )
        
        # Update circles with optimized values
        if result.success:
            for i in range(N_CIRCLES):
                circles[i, 0] = result.x[3*i]
                circles[i, 1] = result.x[3*i+1]
                circles[i, 2] = result.x[3*i+2]
        
        # Final validation
        if not is_valid_configuration(circles):
            # If optimization failed to produce valid solution, fall back to grid
            circles = initialize_grid_placement()
            
    except Exception as e:
        # Fallback to simple initialization if optimization fails
        circles = initialize_grid_placement()
    
    return circles


# EVOLVE-BLOCK-END
