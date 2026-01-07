# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from scipy.spatial import Voronoi
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Try multiple initialization strategies and pick the best
    best_result = None
    best_sum = 0
    
    # Strategy 1: Hexagonal grid with refinement
    circles1 = initialize_hexagonal_pack(n)
    result1 = optimize_circles(circles1)
    sum1 = np.sum(result1[:, 2])
    
    # Strategy 2: Square grid with refinement  
    circles2 = initialize_square_pack(n)
    result2 = optimize_circles(circles2)
    sum2 = np.sum(result2[:, 2])
    
    # Strategy 3: Random initialization with some structure
    circles3 = initialize_random_structured(n)
    result3 = optimize_circles(circles3)
    sum3 = np.sum(result3[:, 2])
    
    # Strategy 4: Voronoi-based initialization
    circles4 = initialize_voronoi_pack(n)
    result4 = optimize_circles(circles4)
    sum4 = np.sum(result4[:, 2])
    
    # Pick the best initialization
    results = [(result1, sum1), (result2, sum2), (result3, sum3), (result4, sum4)]
    best_result = max(results, key=lambda x: x[1])[0]
    
    # Final optimization with multiple restarts
    final_result = multi_restart_optimization(best_result)
    
    # Validate and adjust if needed
    final_result = validate_and_adjust(final_result)
    
    return final_result

def initialize_hexagonal_pack(n):
    """Initialize circles using hexagonal packing pattern"""
    # Create a hexagonal grid pattern
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Adjust for better packing
    if rows * cols < n:
        rows += 1
    
    # Calculate spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Initialize positions
    circles = np.zeros((n, 3))
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = 0.5 * (i % 2)
            x = (j + x_offset) * spacing_x
            y = i * spacing_y
            
            # Ensure we're within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius based on spacing - slightly larger for better packing
            radius = min(spacing_x, spacing_y) * 0.45
            
            circles[idx] = [x, y, radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Set remaining circles with random positions but reasonable radii
    for i in range(idx, n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = np.random.uniform(0.01, 0.1)
        circles[i] = [x, y, radius]
    
    return circles

def initialize_square_pack(n):
    """Initialize circles using square packing pattern"""
    side_length = int(np.ceil(np.sqrt(n)))
    
    # Calculate spacing
    spacing = 1.0 / side_length
    
    circles = np.zeros((n, 3))
    
    idx = 0
    for i in range(side_length):
        for j in range(side_length):
            if idx >= n:
                break
            x = (j + 0.5) * spacing
            y = (i + 0.5) * spacing
            
            # Ensure we're within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius based on spacing
            radius = spacing * 0.4
            
            circles[idx] = [x, y, radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Set remaining circles with random positions but reasonable radii
    for i in range(idx, n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = np.random.uniform(0.01, 0.1)
        circles[i] = [x, y, radius]
    
    return circles

def initialize_random_structured(n):
    """Initialize with some structured randomness"""
    circles = np.zeros((n, 3))
    
    # Place some circles in a structured way first
    structured_count = min(16, n // 2)
    
    # Grid placement for structured part
    grid_size = int(np.ceil(np.sqrt(structured_count)))
    spacing = 1.0 / grid_size
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= structured_count:
                break
            x = (j + 0.5) * spacing
            y = (i + 0.5) * spacing
            radius = spacing * 0.4
            circles[idx] = [x, y, radius]
            idx += 1
    
    # Fill remaining with random
    for i in range(idx, n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = np.random.uniform(0.01, 0.1)
        circles[i] = [x, y, radius]
    
    return circles

def initialize_voronoi_pack(n):
    """Initialize using Voronoi diagram approach"""
    # Generate random points and use Voronoi to create regions
    # Then place circles at centroids with appropriate radii
    points = np.random.rand(n, 2)
    
    # Add boundary points to ensure good coverage
    boundary_points = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5]
    ])
    points = np.vstack([points, boundary_points[:min(8, n-len(points))]])
    
    # Use Voronoi to get regions
    try:
        vor = Voronoi(points)
        # For simplicity, just use the original points as centers
        circles = np.zeros((n, 3))
        for i in range(min(len(points), n)):
            x, y = points[i]
            # Make sure we're within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            # Radius based on proximity to neighbors
            if i < len(vor.points):
                # Estimate radius based on Voronoi cell size
                radius = 0.1
            else:
                radius = np.random.uniform(0.01, 0.1)
            circles[i] = [x, y, radius]
    except:
        # Fallback to random initialization
        circles = np.zeros((n, 3))
        for i in range(n):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            radius = np.random.uniform(0.01, 0.1)
            circles[i] = [x, y, radius]
    
    return circles

def optimize_circles(initial_circles):
    """Optimize circle positions using scipy with better constraints"""
    n = len(initial_circles)
    
    # Flatten parameters: [x0,y0,r0,x1,y1,r1,...]
    initial_params = initial_circles.flatten()
    
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we want to maximize
    
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        
        # Constraint: all circles must be within the unit square
        # x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
        constraints = []
        
        # Containment constraints
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([x - r, 1 - x - r, y - r, 1 - y - r])
        
        # Overlap constraints: distance >= r1 + r2
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                # We want dist >= r1 + r2, so we add -(r1 + r2 - dist) to constraints
                constraints.append(dist - r1 - r2)
        
        return np.array(constraints)
    
    # Define bounds: [x_min, x_max, y_min, y_max, r_min, r_max] for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])  # x, y, r bounds
    
    # Try different optimization approaches
    try:
        # Use trust-constr method which is often better for constrained problems
        cons = {'type': 'ineq', 'fun': constraint_func}
        result = minimize(objective, initial_params, method='trust-constr', 
                         bounds=bounds, constraints=cons, 
                         options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        # Fallback to SLSQP with fewer iterations
        try:
            cons = {'type': 'ineq', 'fun': constraint_func}
            result = minimize(objective, initial_params, method='SLSQP', 
                             bounds=bounds, constraints=cons, 
                             options={'maxiter': 300, 'ftol': 1e-5})
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                return optimized_circles
        except:
            pass
    
    # Fallback: simple gradient descent approach
    return initial_circles

def multi_restart_optimization(initial_circles):
    """Run optimization multiple times with different starting points"""
    best_circles = initial_circles.copy()
    best_sum = np.sum(initial_circles[:, 2])
    
    # Try 3 different restarts
    for restart in range(3):
        # Perturb the initial configuration slightly
        perturbed = initial_circles.copy()
        for i in range(len(perturbed)):
            # Small random perturbations
            perturbed[i, 0] += np.random.normal(0, 0.01)
            perturbed[i, 1] += np.random.normal(0, 0.01)
            perturbed[i, 2] += np.random.normal(0, 0.005)
        
        # Ensure bounds
        for i in range(len(perturbed)):
            perturbed[i, 0] = np.clip(perturbed[i, 0], 0.001, 0.999)
            perturbed[i, 1] = np.clip(perturbed[i, 1], 0.001, 0.999)
            perturbed[i, 2] = np.clip(perturbed[i, 2], 0.001, 0.5)
        
        # Optimize this version
        optimized = optimize_circles(perturbed)
        current_sum = np.sum(optimized[:, 2])
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized
    
    return best_circles

def validate_and_adjust(circles):
    """Ensure final solution is valid"""
    # Make sure all circles are within bounds and don't overlap
    n = len(circles)
    
    # Simple validation and adjustment
    for i in range(n):
        x, y, r = circles[i]
        # Keep radius positive and reasonable
        r = max(0.001, min(0.5, r))
        # Keep center within bounds
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    # Refine to avoid overlaps with more sophisticated approach
    # Use a simple iterative improvement with better convergence criteria
    for iteration in range(50):  # More iterations than before
        improved = False
        # Check all pairs for overlaps
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                if dist < r1 + r2:
                    # Move circles apart
                    dx = x2 - x1
                    dy = y2 - y1
                    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                        # Same position, move randomly
                        angle = np.random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                    else:
                        norm = np.sqrt(dx*dx + dy*dy)
                        dx /= norm
                        dy /= norm
                    
                    # Move them apart
                    move_dist = (r1 + r2 - dist) * 0.5
                    circles[i, 0] -= dx * move_dist
                    circles[i, 1] -= dy * move_dist
                    circles[j, 0] += dx * move_dist
                    circles[j, 1] += dy * move_dist
                    
                    # Keep within bounds
                    circles[i, 0] = np.clip(circles[i, 0], r1, 1-r1)
                    circles[i, 1] = np.clip(circles[i, 1], r1, 1-r1)
                    circles[j, 0] = np.clip(circles[j, 0], r2, 1-r2)
                    circles[j, 1] = np.clip(circles[j, 1], r2, 1-r2)
                    
                    improved = True
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
