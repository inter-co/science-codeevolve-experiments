# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import warnings
warnings.filterwarnings('ignore')

def validate_circles(circles: np.ndarray) -> bool:
    """Validate that all circles are within bounds and non-overlapping."""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints using optimized distance calculation
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Use scipy cdist for efficient pairwise distance computation
    distances = cdist(positions, positions)
    
    # Check for overlaps (avoid double counting by only checking upper triangle)
    for i in range(n):
        for j in range(i+1, n):
            if distances[i, j] < radii[i] + radii[j]:
                return False
    
    return True

def objective_function(circles_flat: np.ndarray) -> float:
    """Objective function to maximize sum of radii (negative for minimization)."""
    return -np.sum(circles_flat[2::3])  # Sum of all radii (indices 2, 5, 8, ...)

def create_hexagonal_initialization(n_circles: int) -> np.ndarray:
    """Create a good initial configuration using hexagonal packing approach."""
    circles = np.zeros((n_circles, 3))
    
    # Use a cleaner hexagonal packing approach similar to successful implementations
    rows = 5
    cols = 5
    
    # Use more precise spacing
    spacing = 0.2
    hex_radius = spacing * 0.4
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
            # Offset every other row for hexagonal packing
            x_offset = spacing * 0.5 if i % 2 == 1 else 0
            x = spacing * j + x_offset + hex_radius
            y = spacing * i + hex_radius
            
            # Ensure within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                max_radius = min(x, 1-x, y, 1-y)
                if max_radius > 0.01:
                    radius = min(hex_radius, max_radius * 0.8)
                    circles[idx] = [x, y, radius]
                    idx += 1
        if idx >= n_circles:
            break
    
    # Fill remaining circles with better random positions
    for i in range(idx, n_circles):
        # Try to place with better constraint checking
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            max_radius = min(x, 1-x, y, 1-y)
            if max_radius > 0.01:
                # Check overlap with existing circles
                radius = min(max_radius * 0.3, 0.15)
                overlap = False
                
                for k in range(i):
                    cx, cy, cr = circles[k]
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < radius + cr:
                        overlap = True
                        break
                
                if not overlap:
                    circles[i] = [x, y, radius]
                    placed = True
            attempts += 1
        
        if not placed:
            # Fallback to simple positioning
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            max_radius = min(x, 1-x, y, 1-y)
            radius = max_radius * 0.1
            circles[i] = [x, y, radius]
    
    return circles

def create_constraints(n_circles: int):
    """Create constraint functions for optimization."""
    constraints = []
    
    # Containment constraints
    for i in range(n_circles):
        # x >= r and x <= 1-r
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
        # y >= r and y <= 1-r
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
    
    # Overlap constraints - using more stable formulation
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, i=i, j=j: np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) - x[3*i+2] - x[3*j+2] - 1e-10
            })
    
    return constraints

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial hexagonal packing + constrained optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = 26
    
    # Generate initial configuration using hexagonal approach
    initial_circles = create_hexagonal_initialization(n)
    initial_flat = initial_circles.flatten()
    
    # Set bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Create constraints
    constraints = create_constraints(n)
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective_function,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-8},
            tol=1e-8
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure all circles are within bounds
            for i in range(n):
                x, y, r = optimized_circles[i]
                # Clip to valid ranges
                optimized_circles[i] = [
                    np.clip(x, r, 1-r),
                    np.clip(y, r, 1-r),
                    np.clip(r, 0.001, 0.499)
                ]
            
            # Final validation
            if validate_circles(optimized_circles):
                return optimized_circles
            else:
                # If optimization failed validation, return initial configuration
                return initial_circles
        else:
            # If optimization fails, return the initial configuration
            return initial_circles
            
    except Exception as e:
        # If optimization fails, return the initial configuration
        return initial_circles


# EVOLVE-BLOCK-END
