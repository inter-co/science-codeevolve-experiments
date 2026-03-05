# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from scipy.optimize import differential_evolution
import random
from sklearn.cluster import KMeans
import time
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining smart initialization with local optimization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set up optimization parameters
    n = 21
    max_time = 55  # Leave some buffer for final processing
    start_time = time.time()
    
    # Best known aspect ratio from previous optimizations
    best_aspect_ratio = (1.5, 0.5)
    best_sum = 0
    best_circles = None
    
    # Focus on the most promising aspect ratios
    test_ratios = [
        (1.5, 0.5), (2.0, 0.5), (1.2, 0.8), (0.8, 1.2), (1.0, 1.0),
        (1.8, 0.7), (0.7, 1.8), (1.3, 0.9), (0.9, 1.3), (1.6, 0.6)
    ]
    
    # Use a much more focused and efficient approach
    for width_ratio, height_ratio in test_ratios:
        if time.time() - start_time > max_time:
            break
            
        width = 2 * width_ratio / (width_ratio + height_ratio)
        height = 2 * height_ratio / (width_ratio + height_ratio)
        
        # Create a more intelligent initial configuration
        circles = np.zeros((n, 3))
        
        # Use a hexagonal grid pattern optimized for 21 circles
        # 5 rows × 4 columns pattern with alternating offsets
        rows = 5
        cols = 4
        
        grid_width = width / cols
        grid_height = height / rows
        
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n:
                    break
                x = (col + 0.5) * grid_width
                if row % 2 == 1:
                    x += grid_width * 0.5  # Full offset for hexagonal packing
                y = (row + 0.5) * grid_height
                
                # Apply boundary checks with margin
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                
                # Conservative initial radius calculation
                r = min(x, width - x, y, height - y) * 0.35
                r = max(0.01, min(r, min(width, height) * 0.35))
                circles[count] = [x, y, r]
                count += 1
        
        # Fill remaining circles with more intelligent placement
        for i in range(count, n):
            # Try to place in areas with more available space
            max_attempts = 100
            for attempt in range(max_attempts):
                x = random.uniform(0.01, width - 0.01)
                y = random.uniform(0.01, height - 0.01)
                
                # Find minimum distance to existing circles
                min_dist = float('inf')
                for j in range(i):
                    existing_x, existing_y = circles[j, 0], circles[j, 1]
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    min_dist = min(min_dist, dist)
                
                # Calculate appropriate radius based on available space
                max_radius = min(x, width - x, y, height - y)
                if min_dist > 0:
                    # Allow for some overlap tolerance for better optimization
                    max_radius = min(max_radius, min_dist * 0.45)
                
                r = max(0.01, min(max_radius, min(width, height) * 0.3))
                if r > 0:
                    circles[i] = [x, y, r]
                    break
        
        # Define constraint function for optimization
        def constraint_func(params):
            # params: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
            positions = params.reshape(-1, 3)[:, :2]
            radii = params.reshape(-1, 3)[:, 2]
            
            # Distance constraint: no overlap
            distances = cdist(positions, positions)
            constraints = []
            
            # Non-overlap constraints (distance >= sum of radii)
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    min_dist = radii[i] + radii[j]
                    # We want dist >= min_dist, so constraint is (dist - min_dist) >= 0
                    constraints.append(dist - min_dist)
            
            # Boundary constraints - ensure circles are within rectangle
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                constraints.append(x - r)  # left boundary
                constraints.append(width - x - r)  # right boundary
                constraints.append(y - r)  # bottom boundary
                constraints.append(height - y - r)  # top boundary
                
            return np.array(constraints)
        
        # Objective function to maximize (negative because minimize)
        def objective(params):
            radii = params.reshape(-1, 3)[:, 2]
            return -np.sum(radii)
        
        # More efficient constraint check
        def check_constraints(params):
            positions = params.reshape(-1, 3)[:, :2]
            radii = params.reshape(-1, 3)[:, 2]
            
            # Check boundary constraints
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                    return False
            
            # Check overlap constraints efficiently
            distances = cdist(positions, positions)
            # Only check pairs where i < j to avoid double counting
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        return False
                        
            return True
        
        # Flatten initial parameters
        initial_params = circles.flatten()
        
        # Try multiple local optimization restarts with better parameters
        try:
            best_local_sum = 0
            best_local_circles = None
            
            # Try with more aggressive local optimization
            for restart in range(8):  # More restarts for better chance
                # Start with slightly perturbed version
                if restart == 0:
                    start_params = initial_params.copy()
                else:
                    # Perturb with larger variance for better exploration
                    noise = np.random.normal(0, 0.03, len(initial_params))
                    start_params = initial_params + noise
                    
                local_result = minimize(
                    objective,
                    start_params,
                    method='SLSQP',
                    bounds=[(0, width), (0, height), (0.001, min(width, height)/2)] * n,
                    constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                    options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-6}
                )
                
                if local_result.success:
                    optimized_circles = local_result.x.reshape(-1, 3)
                    current_sum = np.sum(optimized_circles[:, 2])
                    if current_sum > best_local_sum:
                        best_local_sum = current_sum
                        best_local_circles = optimized_circles.copy()
            
            if best_local_circles is not None and best_local_sum > best_sum:
                best_sum = best_local_sum
                best_circles = best_local_circles.copy()
                
        except Exception as e:
            continue
    
    # If we still don't have a good solution, try a simpler but more focused approach
    if best_circles is None or best_sum < 2.0:
        # Use a clean, well-tested pattern with high-quality local optimization
        width, height = 1.5, 0.5  # Proven good aspect ratio
        
        # Create a clean hexagonal grid pattern
        circles = np.zeros((n, 3))
        
        rows = 5
        cols = 4
        grid_width = width / cols
        grid_height = height / rows
        
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n:
                    break
                x = (col + 0.5) * grid_width
                if row % 2 == 1:
                    x += grid_width * 0.5  # Hexagonal offset
                y = (row + 0.5) * grid_height
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                r = min(grid_width, grid_height) * 0.35
                r = max(0.01, min(r, min(width, height) * 0.35))
                circles[count] = [x, y, r]
                count += 1
        
        # Fill remaining with intelligent random placement
        for i in range(count, n):
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            r = random.uniform(0.01, min(width, height) * 0.25)
            circles[i] = [x, y, r]
        
        # Apply aggressive local optimization
        bounds = []
        for i in range(n):
            bounds.extend([(0, width), (0, height), (0.001, min(width, height)/2)])
        
        def objective(params):
            radii = params.reshape(-1, 3)[:, 2]
            return -np.sum(radii)
        
        def constraint_func(params):
            positions = params.reshape(-1, 3)[:, :2]
            radii = params.reshape(-1, 3)[:, 2]
            
            distances = cdist(positions, positions)
            constraints = []
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    min_dist = radii[i] + radii[j]
                    constraints.append(dist - min_dist)
            
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                constraints.append(x - r)
                constraints.append(width - x - r)
                constraints.append(y - r)
                constraints.append(height - y - r)
                
            return np.array(constraints)
        
        # Multiple restarts with better convergence control
        best_final_sum = 0
        best_final_circles = None
        
        for restart in range(10):
            if restart == 0:
                start_params = circles.flatten()
            else:
                # Perturb more aggressively for better exploration
                noise = np.random.normal(0, 0.02, len(circles.flatten()))
                start_params = circles.flatten() + noise
            
            try:
                local_result = minimize(
                    objective,
                    start_params,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                    options={'maxiter': 800, 'ftol': 1e-9, 'eps': 1e-7}
                )
                
                if local_result.success:
                    optimized_circles = local_result.x.reshape(-1, 3)
                    current_sum = np.sum(optimized_circles[:, 2])
                    if current_sum > best_final_sum:
                        best_final_sum = current_sum
                        best_final_circles = optimized_circles.copy()
            except:
                continue
        
        if best_final_circles is not None:
            best_circles = best_final_circles.copy()
    
    # Return the best solution found
    if best_circles is not None:
        return best_circles
    else:
        # Fallback to a simple structured solution
        circles = np.zeros((21, 3))
        width, height = 1.5, 0.5
        rows, cols = 5, 4
        grid_width = width / cols
        grid_height = height / rows
        
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= 21:
                    break
                x = (col + 0.5) * grid_width
                if row % 2 == 1:
                    x += grid_width * 0.25
                y = (row + 0.5) * grid_height
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                r = min(grid_width, grid_height) * 0.35
                circles[count] = [x, y, r]
                count += 1
        
        for i in range(count, 21):
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            r = random.uniform(0.01, min(width, height) * 0.2)
            circles[i] = [x, y, r]
        
        return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
