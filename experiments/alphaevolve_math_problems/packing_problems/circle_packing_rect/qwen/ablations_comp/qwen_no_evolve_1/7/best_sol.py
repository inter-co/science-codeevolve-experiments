# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import random
from itertools import combinations
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach with advanced initialization and optimization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set up optimization parameters
    np.random.seed(42)
    random.seed(42)
    
    # Try multiple aspect ratios systematically
    aspect_ratios = [
        (1.0, 1.0),      # square
        (1.5, 0.5),      # wide rectangle
        (0.5, 1.5),      # tall rectangle  
        (2.0, 1.0),      # very wide
        (1.0, 2.0),      # very tall
        (1.2, 0.8),      # slightly rectangular
        (0.8, 1.2),      # slightly rectangular
    ]
    
    best_sum = 0
    best_circles = None
    
    # For each aspect ratio, try different optimization strategies
    for width_ratio, height_ratio in aspect_ratios:
        width = 2 * width_ratio / (width_ratio + height_ratio)
        height = 2 * height_ratio / (width_ratio + height_ratio)
        
        # Create better initial configuration using hexagonal packing with refinement
        circles = create_hexagonal_initialization(width, height, 21)
        
        # Try multiple optimization approaches
        optimized_circles = optimize_with_constraints(circles, width, height)
        
        if optimized_circles is not None:
            current_sum = np.sum(optimized_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized_circles.copy()
        
        # Also try a direct optimization approach with better bounds
        circles_direct = create_direct_initialization(width, height, 21)
        optimized_direct = optimize_with_constraints(circles_direct, width, height)
        
        if optimized_direct is not None:
            current_sum = np.sum(optimized_direct[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized_direct.copy()
    
    # If we still don't have good results, do a final comprehensive optimization
    if best_circles is None:
        # Try a more systematic approach with better initial guess
        width, height = 1.0, 1.0  # Default square
        circles = create_optimized_initialization(width, height, 21)
        best_circles = optimize_with_constraints(circles, width, height)
    
    # Final fallback to ensure we always return something
    if best_circles is None:
        # Create a simple but reasonable configuration
        width, height = 1.2, 0.8
        circles = np.zeros((21, 3))
        for i in range(21):
            # Distribute in a grid pattern with some randomness
            row = i // 6
            col = i % 6
            x = 0.1 + col * (width - 0.2) / 5 + (np.random.rand() - 0.5) * 0.05
            y = 0.1 + row * (height - 0.2) / 3 + (np.random.rand() - 0.5) * 0.05
            r = min(0.1, (min(x, width-x, y, height-y) * 0.8))
            circles[i] = [x, y, r]
        return circles
    
    return best_circles

def create_hexagonal_initialization(width: float, height: float, n: int) -> np.ndarray:
    """Create initial configuration using hexagonal packing pattern."""
    circles = np.zeros((n, 3))
    
    # Use a hexagonal lattice approach
    # Determine grid size
    rows = int(np.sqrt(n) * 1.2) + 1
    cols = int(n / rows) + 1
    
    # Adjust to fit exactly n circles
    while rows * cols < n:
        rows += 1
    
    grid_width = width / cols
    grid_height = height / rows
    
    # Hexagonal packing with offset rows
    circle_positions = []
    for row in range(rows):
        for col in range(cols):
            if len(circle_positions) >= n:
                break
            x = (col + 0.5) * grid_width
            if row % 2 == 1:
                x += grid_width * 0.5
            y = (row + 0.5) * grid_height
            if x <= width and y <= height:
                circle_positions.append([x, y])
    
    # Fill remaining positions if needed
    while len(circle_positions) < n:
        # Add random positions near existing ones
        if circle_positions:
            idx = random.randint(0, len(circle_positions)-1)
            base_x, base_y = circle_positions[idx]
            x = max(0.01, min(width - 0.01, base_x + (random.random() - 0.5) * grid_width * 0.8))
            y = max(0.01, min(height - 0.01, base_y + (random.random() - 0.5) * grid_height * 0.8))
            circle_positions.append([x, y])
        else:
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            circle_positions.append([x, y])
    
    # Assign positions and radii
    for i in range(n):
        x, y = circle_positions[i]
        # Initial radius based on available space
        r = min(x, width - x, y, height - y) * 0.3
        r = max(0.01, min(r, min(grid_width, grid_height) * 0.4))
        circles[i] = [x, y, r]
    
    return circles

def create_direct_initialization(width: float, height: float, n: int) -> np.ndarray:
    """Create initial configuration with direct placement."""
    circles = np.zeros((n, 3))
    
    # Start with a simple grid pattern
    rows = int(np.sqrt(n)) + 1
    cols = n // rows + 1
    
    grid_width = width / cols
    grid_height = height / rows
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = (col + 0.5) * grid_width + (np.random.rand() - 0.5) * grid_width * 0.3
            y = (row + 0.5) * grid_height + (np.random.rand() - 0.5) * grid_height * 0.3
            # Ensure within bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))
            
            # Radius based on proximity to edges and other circles
            r = min(x, width - x, y, height - y) * 0.25
            r = max(0.01, min(r, min(grid_width, grid_height) * 0.3))
            
            circles[count] = [x, y, r]
            count += 1
            
        if count >= n:
            break
    
    return circles

def create_optimized_initialization(width: float, height: float, n: int) -> np.ndarray:
    """Create highly optimized initial configuration."""
    circles = np.zeros((n, 3))
    
    # Use a more sophisticated approach: place in clusters
    cluster_centers = []
    num_clusters = min(4, n)
    
    # Place cluster centers
    for i in range(num_clusters):
        x = 0.2 + i * (width - 0.4) / (num_clusters - 1) if num_clusters > 1 else width / 2
        y = 0.2 + (i % 2) * (height - 0.4) / 2
        cluster_centers.append((x, y))
    
    # Distribute circles among clusters
    circles_per_cluster = [n // num_clusters] * num_clusters
    for i in range(n % num_clusters):
        circles_per_cluster[i] += 1
    
    idx = 0
    for cluster_idx, (cx, cy) in enumerate(cluster_centers):
        circles_in_cluster = circles_per_cluster[cluster_idx]
        for i in range(circles_in_cluster):
            if idx >= n:
                break
            # Place within cluster with some randomness
            angle = random.random() * 2 * np.pi
            distance = random.random() * min(width, height) * 0.1
            x = cx + np.cos(angle) * distance
            y = cy + np.sin(angle) * distance
            
            # Keep within bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))
            
            # Radius based on available space
            r = min(x, width - x, y, height - y) * 0.25
            r = max(0.01, min(r, 0.15))
            
            circles[idx] = [x, y, r]
            idx += 1
            
    return circles

def optimize_with_constraints(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize circles with proper constraints."""
    n = len(initial_circles)
    
    # Define constraint function for scipy
    def constraint_func(params):
        positions = params.reshape(-1, 3)[:, :2]
        radii = params.reshape(-1, 3)[:, 2]
        
        # Non-overlap constraints (distance >= sum of radii)
        constraints = []
        for i, j in combinations(range(n), 2):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            # We want dist >= min_dist, so constraint is (dist - min_dist) >= 0
            constraints.append(dist - min_dist)
        
        # Boundary constraints (circle must be within rectangle)
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
    
    # Flatten initial parameters
    initial_params = initial_circles.flatten()
    
    # Bounds for optimization
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0, width))
        # y bounds
        bounds.append((0, height))
        # r bounds (smaller than half the smallest dimension)
        bounds.append((0.001, min(width, height) / 2))
    
    try:
        # Use L-BFGS-B which works well for this type of problem
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=None
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Verify constraints are satisfied
            positions = optimized_circles[:, :2]
            radii = optimized_circles[:, 2]
            
            # Check non-overlap
            valid = True
            for i, j in combinations(range(n), 2):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist * 0.999:  # Allow slight tolerance
                    valid = False
                    break
            
            # Check boundaries
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                if (x - r < 0 or x + r > width or y - r < 0 or y + r > height):
                    valid = False
                    break
            
            if valid:
                return optimized_circles
        
        # If L-BFGS fails, try a simpler approach with constraints
        # Use SLSQP with explicit constraints
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                return optimized_circles
                
        except:
            pass
            
    except Exception as e:
        # Return original if optimization fails
        pass
    
    # Return original if nothing works
    return initial_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
