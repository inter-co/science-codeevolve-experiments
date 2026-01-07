# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal lattice initialization with gradient-based optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using hexagonal lattice packing pattern
    circles = initialize_hexagonal_lattice(n)
    
    # Refine using optimization
    circles = optimize_circles(circles)
    
    return circles

def initialize_hexagonal_lattice(n: int) -> np.ndarray:
    """Initialize circle positions using hexagonal lattice packing"""
    # For 32 circles, we'll use approximately 6 rows and 6 columns in hexagonal arrangement
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Hexagonal lattice parameters
    sqrt3 = math.sqrt(3)
    spacing_x = 1.0 / (cols + 0.5)
    spacing_y = sqrt3 * spacing_x / 2
    
    # Create initial positions
    circles = []
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
                
            # Offset every other row
            x_offset = (i % 2) * spacing_x / 2
            x = (j + 0.5) * spacing_x + x_offset
            y = (i + 0.5) * spacing_y
            
            # Ensure positions are within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Initial radius guess based on spacing
                radius = min(spacing_x, spacing_y) / 2
                circles.append([x, y, radius])
                count += 1
                
        if count >= n:
            break
    
    # Fill remaining circles with uniform distribution if needed
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = 0.05
        circles.append([x, y, radius])
    
    return np.array(circles)

def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using constrained optimization"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_flat = []
    for i in range(n):
        initial_flat.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    def objective(x_flat):
        # Extract positions and radii
        total_radius = 0
        for i in range(n):
            total_radius += x_flat[3*i + 2]
        return -total_radius  # Negative because we want to maximize
    
    def constraint_func(x_flat):
        # Check containment constraints
        constraints = []
        
        # Circle containment in unit square
        for i in range(n):
            x, y, r = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
            
            # Radius must be positive
            constraints.append(r)
            
            # Circle must be within square boundaries
            constraints.append(1 - r - x)  # Right boundary
            constraints.append(1 - r - y)  # Top boundary
            constraints.append(x - r)      # Left boundary
            constraints.append(y - r)      # Bottom boundary
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
                x2, y2, r2 = x_flat[3*j], x_flat[3*j+1], x_flat[3*j+2]
                
                # Distance constraint: d >= r1 + r2
                dx = x1 - x2
                dy = y1 - y2
                distance = math.sqrt(dx*dx + dy*dy)
                constraints.append(distance - (r1 + r2))
        
        return np.array(constraints)
    
    # Set up bounds: x, y in [r, 1-r], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])  # x, y, r bounds
    
    # Use SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [optimized[3*i], optimized[3*i+1], optimized[3*i+2]]
            return circles
    except Exception:
        pass
    
    # If optimization fails, return initial configuration
    return initial_circles


# EVOLVE-BLOCK-END
