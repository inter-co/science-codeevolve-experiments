# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a hexagonal packing pattern for good starting configuration
    circles = initialize_hexagonal_packing(n)
    
    # Refine using optimization to maximize sum of radii
    circles = optimize_circles(circles)
    
    return circles

def initialize_hexagonal_packing(n: int) -> np.ndarray:
    """Initialize circle positions using a hexagonal packing pattern"""
    # Estimate how many rows/columns we need
    # Hexagonal packing density is ~0.9069
    area_needed = n * (1/32)  # Rough estimate
    side_length = math.sqrt(area_needed)
    
    # Create hexagonal grid
    circles = []
    radius_guess = 0.05  # Initial guess
    
    # Place circles in a hexagonal pattern
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Adjust spacing based on number of circles needed
    spacing_x = 1.0 / max(1, cols)
    spacing_y = 1.0 / max(1, rows)
    
    # Create hexagonal pattern with offset rows
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            # Ensure circles fit within bounds
            if x >= radius_guess and x <= 1-radius_guess and y >= radius_guess and y <= 1-radius_guess:
                circles.append([x, y, radius_guess])
    
    # Fill remaining spots if needed
    while len(circles) < n:
        # Add some random positions with small radii
        x = np.random.uniform(radius_guess, 1-radius_guess)
        y = np.random.uniform(radius_guess, 1-radius_guess)
        circles.append([x, y, radius_guess])
    
    return np.array(circles[:n])

def compute_radius_constraints(circles: np.ndarray) -> tuple:
    """Compute constraints for circle packing"""
    n = len(circles)
    
    # Position constraints: radius must be such that circle fits in unit square
    pos_constraints = []
    for i in range(n):
        x, y, r = circles[i]
        # Circle must fit completely within unit square
        pos_constraints.extend([
            {'type': 'ineq', 'fun': lambda c, i=i: c[i*3+2] - (1 - c[i*3])},  # r <= 1-x
            {'type': 'ineq', 'fun': lambda c, i=i: c[i*3+2] - c[i*3]},       # r <= x
            {'type': 'ineq', 'fun': lambda c, i=i: c[i*3+2] - (1 - c[i*3+1])}, # r <= 1-y
            {'type': 'ineq', 'fun': lambda c, i=i: c[i*3+2] - c[i*3+1]}      # r <= y
        ])
    
    # Non-overlap constraints
    overlap_constraints = []
    for i in range(n):
        for j in range(i+1, n):
            overlap_constraints.append({
                'type': 'ineq',
                'fun': lambda c, i=i, j=j: (
                    np.sqrt((c[i*3] - c[j*3])**2 + (c[i*3+1] - c[j*3+1])**2) 
                    - (c[i*3+2] + c[j*3+2])
                )
            })
    
    return pos_constraints, overlap_constraints

def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using scipy minimize"""
    n = len(initial_circles)
    
    # Flatten initial circles array for optimization
    initial_flat = initial_circles.flatten()
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(x_flat):
        circles = x_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we're minimizing
    
    # Define constraint functions
    def position_constraint(x_flat):
        circles = x_flat.reshape(-1, 3)
        constraints = []
        for i in range(n):
            x, y, r = circles[i]
            # r <= x, r <= 1-x, r <= y, r <= 1-y
            constraints.extend([
                x - r,           # x >= r
                1 - x - r,       # 1-x >= r
                y - r,           # y >= r
                1 - y - r        # 1-y >= r
            ])
        return np.array(constraints)
    
    def overlap_constraint(x_flat):
        circles = x_flat.reshape(-1, 3)
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                # Distance between centers >= sum of radii
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                constraints.append(dist - (r1 + r2))  # Must be >= 0
        return np.array(constraints)
    
    # Set up constraints
    constraints = [
        {'type': 'ineq', 'fun': position_constraint},
        {'type': 'ineq', 'fun': overlap_constraint}
    ]
    
    # Bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([
            (0.001, 0.999),   # x coordinate
            (0.001, 0.999),   # y coordinate
            (0.001, 0.5)      # radius (max radius limited to prevent extreme values)
        ])
    
    try:
        # Run optimization
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure all circles are valid
            for i in range(len(optimized_circles)):
                x, y, r = optimized_circles[i]
                # Clamp to valid range
                optimized_circles[i] = [
                    np.clip(x, r, 1-r),
                    np.clip(y, r, 1-r),
                    np.clip(r, 0.001, 0.5)
                ]
            return optimized_circles
        else:
            # If optimization fails, return the initial configuration
            return initial_circles
            
    except Exception as e:
        # If optimization fails, return the initial configuration
        return initial_circles


# EVOLVE-BLOCK-END
