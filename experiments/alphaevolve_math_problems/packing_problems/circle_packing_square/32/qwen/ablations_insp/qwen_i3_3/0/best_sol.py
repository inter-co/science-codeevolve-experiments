# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import warnings
warnings.filterwarnings('ignore')
import random

def initialize_hexagonal_layout(n: int) -> np.ndarray:
    """Initialize circles using a hexagonal packing layout"""
    circles = []
    sqrt3 = np.sqrt(3)
    
    # Determine grid dimensions for approximately n circles
    rows = int(np.ceil(np.sqrt(n * 2 / sqrt3)))
    cols = int(np.ceil(n / rows))
    
    # Spacing based on hexagonal packing
    radius = 0.08  # Starting radius estimate
    horizontal_spacing = 2 * radius
    vertical_spacing = sqrt3 * radius
    
    # Create hexagonal grid
    for i in range(rows):
        y = radius + i * vertical_spacing
        if y > 1 - radius:
            break
        for j in range(cols):
            x = radius + j * horizontal_spacing
            if x > 1 - radius:
                break
            # Offset every other row
            if i % 2 == 1:
                x += horizontal_spacing / 2
            if x <= 1 - radius and y <= 1 - radius:
                circles.append([x, y, radius])
    
    # Fill remaining circles with random placement near grid points
    while len(circles) < n:
        # Add random placements near existing grid points
        if circles:
            base_idx = np.random.randint(len(circles))
            base_x, base_y, base_r = circles[base_idx]
            x = np.clip(base_x + np.random.normal(0, 0.03), base_r, 1-base_r)
            y = np.clip(base_y + np.random.normal(0, 0.03), base_r, 1-base_r)
            circles.append([x, y, base_r])
        else:
            # If no circles yet, place randomly
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
    return np.array(circles[:n])

def initialize_voronoi(n: int) -> np.ndarray:
    """Initialize circles using Voronoi diagram approach"""
    try:
        np.random.seed(42)  # For reproducibility
        points = np.random.rand(50, 2)  # More points than needed for Voronoi
        
        vor = Voronoi(points)
        # Use Voronoi cell centroids as initial positions (but keep within bounds)
        positions = []
        for i in range(min(n, len(vor.points))):
            point = vor.points[i]
            x = np.clip(point[0], 0.05, 0.95)
            y = np.clip(point[1], 0.05, 0.95)
            positions.append([x, y])
        
        # Fill remaining positions randomly
        while len(positions) < n:
            positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
            
        return np.array(positions[:n])
        
    except Exception:
        # Fallback to grid initialization
        positions = []
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                positions.append([x, y])
                
        # Fill remaining positions randomly
        while len(positions) < n:
            positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
            
        return np.array(positions[:n])

def initialize_grid_placement(n: int) -> np.ndarray:
    """Initialize circles using a grid-based approach with random perturbations"""
    # Create a grid layout
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Calculate spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    circles = np.zeros((n, 3))
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base position with some padding
        x_base = (col + 0.5) * spacing_x
        y_base = (row + 0.5) * spacing_y
        
        # Add small random perturbation
        perturbation = 0.05 * spacing_x
        x = max(perturbation, min(1.0 - perturbation, x_base + random.uniform(-perturbation, perturbation)))
        y = max(perturbation, min(1.0 - perturbation, y_base + random.uniform(-perturbation, perturbation)))
        
        # Initial radius - start with small values
        r = min(spacing_x, spacing_y) * 0.2
        
        circles[i] = [x, y, r]
    
    return circles

def objective(params):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(params[2::3])  # Sum of all radii (every 3rd element starting from index 2)

def constraint_containment(params):
    """Ensure all circles fit inside the unit square"""
    n = 32
    constraints = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        r = params[3*i+2]
        # Circle must stay inside square: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
        constraints.extend([
            x - r,           # x >= r
            1 - x - r,       # 1 - x >= r
            y - r,           # y >= r
            1 - y - r        # 1 - y >= r
        ])
    return np.array(constraints)

def constraint_nonoverlap(params):
    """Ensure no overlaps between circles - optimized version"""
    n = 32
    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
            x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
            
            # Distance between centers
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            dist = np.sqrt(dist_sq)
            
            # Non-overlap constraint: distance >= r1 + r2
            constraints.append(dist - (r1 + r2))
    return np.array(constraints)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies with robust optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Constraints list
    cons = [
        {'type': 'ineq', 'fun': constraint_containment},
        {'type': 'ineq', 'fun': constraint_nonoverlap}
    ]
    
    # Bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])  # x, y, r
    
    # Try multiple initialization strategies and optimization attempts
    best_circles = None
    best_sum = 0
    
    # Try different initialization methods with multiple optimization attempts
    init_methods = [initialize_hexagonal_layout, initialize_voronoi, initialize_grid_placement]
    
    for init_method in init_methods:
        # Try several optimization attempts with same initialization
        for attempt in range(3):  # Reduced attempts for speed
            try:
                # Get initial configuration
                if attempt == 0:
                    circles = init_method(n)
                else:
                    # Perturb previous result
                    circles = best_circles.copy() if best_circles is not None else init_method(n)
                    for i in range(n):
                        circles[i, 0] += np.random.normal(0, 0.02)
                        circles[i, 1] += np.random.normal(0, 0.02)
                        circles[i, 0] = np.clip(circles[i, 0], 0.01, 0.99)
                        circles[i, 1] = np.clip(circles[i, 1], 0.01, 0.99)
                
                # Flatten for optimization
                initial_params = circles.flatten()
                
                # Try multiple optimization methods for better results
                methods = ['SLSQP']  # Simplified to just SLSQP for speed
                best_result = None
                best_value = float('inf')
                
                for method in methods:
                    try:
                        result = minimize(
                            objective, 
                            initial_params, 
                            method=method, 
                            bounds=bounds, 
                            constraints=cons,
                            options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}  # Reduced iterations for speed
                        )
                        
                        if result.success:
                            # Check if this is better than previous attempts
                            if result.fun < best_value:
                                best_value = result.fun
                                best_result = result
                                
                    except Exception:
                        continue
                
                # Return best result or fallback to initial
                if best_result is not None and best_result.success:
                    optimized_circles = best_result.x.reshape((n, 3))
                else:
                    # If optimization fails, return initial configuration with corrected radii
                    optimized_circles = circles.copy()
                    for i in range(n):
                        x, y, r = optimized_circles[i]
                        optimized_circles[i, 2] = min(r, x, 1-x, y, 1-y)
                
                # Validate and correct
                final_circles = optimized_circles.copy()
                for i in range(n):
                    x, y, r = final_circles[i]
                    # Ensure circle fits in unit square
                    r = min(r, x, 1-x, y, 1-y)
                    # Ensure positive radius
                    r = max(1e-6, r)
                    final_circles[i] = [x, y, r]
                
                # Calculate sum of radii
                radii_sum = np.sum(final_circles[:, 2])
                if radii_sum > best_sum:
                    best_sum = radii_sum
                    best_circles = final_circles
                    
            except Exception:
                continue
    
    # If we still don't have a good solution, return fallback
    if best_circles is None:
        # Final fallback: grid-based solution with better spacing
        best_circles = initialize_grid_placement(n)
    
    return best_circles


# EVOLVE-BLOCK-END
