# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from scipy.optimize import differential_evolution
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-strategy approach combining geometric initialization with advanced optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Strategy 1: Try multiple initialization strategies
    best_result = None
    best_sum = 0
    
    # Strategy 1: Hexagonal packing with better parameters
    circles1 = initialize_better_hexagonal(n)
    
    # Strategy 2: Grid-based initialization
    circles2 = initialize_grid(n)
    
    # Strategy 3: Random with constraint satisfaction
    circles3 = initialize_random_with_constraints(n)
    
    # Try optimization on all initializations
    strategies = [
        ("hexagonal", circles1),
        ("grid", circles2),
        ("random", circles3)
    ]
    
    for name, init_circles in strategies:
        try:
            optimized = optimize_circles_advanced(init_circles)
            total_radius = np.sum(optimized[:, 2])
            if total_radius > best_sum:
                best_sum = total_radius
                best_result = optimized.copy()
        except Exception as e:
            continue
    
    # If no good result from strategies, fallback to basic approach
    if best_result is None:
        circles = initialize_better_hexagonal(n)
        best_result = optimize_circles_advanced(circles)
    
    return best_result

def initialize_better_hexagonal(n: int) -> np.ndarray:
    """Initialize circle positions using improved hexagonal packing"""
    # For 32 circles, we can arrange in a 6x6 grid with some adjustments
    rows = 6
    cols = 6
    
    # Adjust spacing to fit 32 circles optimally
    spacing_x = 0.9 / cols  # Leave small margin
    spacing_y = spacing_x * math.sqrt(3) / 2
    
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
                # Initial radius - smaller than spacing to allow for optimization
                radius = min(spacing_x, spacing_y) / 3
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

def initialize_grid(n: int) -> np.ndarray:
    """Initialize using regular grid pattern"""
    # Arrange in a grid-like fashion
    side = int(math.ceil(math.sqrt(n)))
    spacing = 1.0 / (side + 1)
    
    circles = []
    count = 0
    
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            x = (j + 1) * spacing
            y = (i + 1) * spacing
            radius = spacing / 3
            circles.append([x, y, radius])
            count += 1
            
        if count >= n:
            break
    
    # Fill remaining circles
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = 0.05
        circles.append([x, y, radius])
    
    return np.array(circles)

def initialize_random_with_constraints(n: int) -> np.ndarray:
    """Initialize with random positions but respect basic constraints"""
    circles = []
    attempts = 0
    max_attempts = 1000
    
    while len(circles) < n and attempts < max_attempts:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = 0.05
        
        # Check if this circle conflicts with existing ones
        valid = True
        for existing_circle in circles:
            existing_x, existing_y, existing_r = existing_circle
            dist = math.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            if dist < (radius + existing_r):
                valid = False
                break
                
        if valid:
            circles.append([x, y, radius])
        attempts += 1
    
    # Fill remaining with random positions
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = 0.05
        circles.append([x, y, radius])
    
    return np.array(circles)

def optimize_circles_advanced(initial_circles: np.ndarray) -> np.ndarray:
    """Advanced optimization with multiple strategies"""
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
    
    # Try multiple optimization approaches
    results = []
    
    # Approach 1: Differential evolution (global search)
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            constraints=[{'type': 'ineq', 'fun': constraint_func}],
            maxiter=500,
            popsize=15,
            seed=42
        )
        if de_result.success:
            results.append(de_result)
    except Exception:
        pass
    
    # Approach 2: Local optimization with restarts
    try:
        # Multiple restarts with different initial points
        for _ in range(3):
            # Perturb initial point slightly
            perturbed = initial_flat.copy()
            for i in range(len(perturbed)):
                if i % 3 == 2:  # radius
                    perturbed[i] = max(0.001, min(0.5, perturbed[i] + np.random.normal(0, 0.01)))
                else:  # x or y
                    perturbed[i] = max(0.001, min(0.999, perturbed[i] + np.random.normal(0, 0.02)))
            
            local_result = minimize(
                objective,
                perturbed,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            if local_result.success:
                results.append(local_result)
    except Exception:
        pass
    
    # Select best result
    best_result = None
    best_value = float('inf')
    
    for result in results:
        if result.success and result.fun < best_value:
            best_value = result.fun
            best_result = result
    
    # If we have a good result, convert back to circles format
    if best_result is not None:
        optimized = best_result.x
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [optimized[3*i], optimized[3*i+1], optimized[3*i+2]]
        return circles
    
    # Fallback to initial configuration if optimization fails
    return initial_circles


# EVOLVE-BLOCK-END
