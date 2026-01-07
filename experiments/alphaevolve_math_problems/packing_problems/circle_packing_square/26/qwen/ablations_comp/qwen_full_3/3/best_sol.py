# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
import math
from scipy.spatial import KDTree

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies with optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Strategy 1: Hexagonal grid (from inspiration 1)
    def initialize_hexagonal_grid():
        rows = 5
        cols = 6
        spacing_x = 0.9 / cols  # Leave 0.05 margin on each side
        spacing_y = 0.9 / rows
        
        circles = np.zeros((n, 3))
        idx = 0
        
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = 0.05 + j * spacing_x
                y = 0.05 + i * spacing_y
                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x / 2
                circles[idx] = [x, y, min(spacing_x, spacing_y) * 0.2]  # Start with reasonable radius
                idx += 1
                if idx >= n:
                    break
        return circles
    
    # Strategy 2: Better grid initialization (from inspiration 1)
    def initialize_grid():
        rows = 5
        cols = 6
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows
        
        circles = np.zeros((n, 3))
        idx = 0
        
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                y = 0.05 + (i + 0.5) * spacing_y
                circles[idx] = [x, y, min(spacing_x, spacing_y) * 0.25]
                idx += 1
                if idx >= n:
                    break
        return circles
    
    # Strategy 3: Improved random initialization (from inspiration 1)
    def initialize_random():
        circles = np.zeros((n, 3))
        for i in range(n):
            # Better distributed random points
            row = i // 5
            col = i % 5
            x = 0.1 + col * 0.18 + random.uniform(-0.02, 0.02)
            y = 0.1 + row * 0.18 + random.uniform(-0.02, 0.02)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            r = 0.07
            circles[i] = [x, y, r]
        return circles
    
    # Strategy 4: From inspiration 2 - more systematic approach
    def initialize_from_inspiration2():
        circles = np.zeros((n, 3))
        # Create a more uniform distribution
        for i in range(n):
            # Use a systematic approach to place points
            row = i // 5
            col = i % 5
            x = 0.1 + col * 0.18
            y = 0.1 + row * 0.18
            # Add some jitter to prevent perfect alignment
            x += random.uniform(-0.01, 0.01)
            y += random.uniform(-0.01, 0.01)
            # Clip to keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            r = 0.075
            circles[i] = [x, y, r]
        return circles
    
    # Strategy 5: Optimized grid with better spacing
    def initialize_optimized_grid():
        # Create a more optimized 5x6 grid with better initial radii
        circles = np.zeros((n, 3))
        rows, cols = 5, 6
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                y = 0.05 + (i + 0.5) * spacing_y
                # Set initial radius based on spacing
                r = min(spacing_x, spacing_y) * 0.3
                circles[idx] = [x, y, r]
        return circles
    
    # Try different initialization strategies
    strategies = [
        initialize_hexagonal_grid(),
        initialize_grid(),
        initialize_random(),
        initialize_from_inspiration2(),
        initialize_optimized_grid()
    ]
    
    best_circles = None
    best_sum = 0
    
    for strategy in strategies:
        sum_radii = np.sum(strategy[:, 2])
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_circles = strategy.copy()
    
    # Run several optimization attempts with better settings
    best_final = None
    best_final_sum = 0
    
    # Run multiple optimization attempts to get better results
    for attempt in range(20):  # Reduced from 25 to stay within time limits
        # Create a slightly randomized version of our best initial circles
        current_circles = best_circles.copy()
        
        # Add moderate random perturbations for better exploration
        for i in range(n):
            current_circles[i, 0] += random.uniform(-0.02, 0.02)
            current_circles[i, 1] += random.uniform(-0.02, 0.02)
            current_circles[i, 2] += random.uniform(-0.01, 0.01)
            # Keep within bounds
            current_circles[i, 0] = np.clip(current_circles[i, 0], 0.001, 0.999)
            current_circles[i, 1] = np.clip(current_circles[i, 1], 0.001, 0.999)
            current_circles[i, 2] = np.clip(current_circles[i, 2], 0.001, 0.499)
        
        # Optimize with moderate precision and iterations
        optimized = optimize_circles_moderate_precision(current_circles)
        
        # Apply enhanced refinement with early stopping
        refined = enhanced_refine_positions(optimized)
        
        # Check if this is better
        current_sum = np.sum(refined[:, 2])
        if current_sum > best_final_sum:
            best_final_sum = current_sum
            best_final = refined.copy()
    
    # If we got a better result from optimization, return it
    if best_final is not None:
        return best_final
    
    # Otherwise, return the best initialization
    return best_circles

def optimize_circles_moderate_precision(circles):
    """Use scipy SLSQP optimization with moderate precision for faster execution"""
    # Flatten circles array for optimization
    circles_flat = circles.flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(len(circles_flat)):
        if i % 3 == 0:  # x coordinate
            bounds.append((0.001, 0.999))
        elif i % 3 == 1:  # y coordinate
            bounds.append((0.001, 0.999))
        else:  # r coordinate
            bounds.append((0.001, 0.5))  # Radius bounded
    
    # Vectorized constraint functions for better performance
    def constraint_containment(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        n = len(circles)
        # Vectorized containment constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        constraints = np.empty(4 * n)
        constraints[0::4] = x - r          # x - r >= 0
        constraints[1::4] = 1 - x - r      # 1 - x - r >= 0
        constraints[2::4] = y - r          # y - r >= 0
        constraints[3::4] = 1 - y - r      # 1 - y - r >= 0
        return constraints

    def constraint_overlap(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        n = len(circles)
        # Vectorized overlap constraints using efficient distance calculation
        constraints = []
        
        # Use scipy's distance matrix for efficiency
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Fill in overlap constraints for all pairs (only upper triangle to avoid duplicates)
        for i in range(n):
            for j in range(i+1, n):
                distance = distances[i, j]
                r1 = circles[i, 2]
                r2 = circles[j, 2]
                constraints.append(distance - r1 - r2)
        
        return np.array(constraints)
    
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_overlap(x)}
    ]
    
    try:
        # Perform optimization with moderate precision for speed
        result = minimize(
            lambda x: -np.sum(x[2::3]),  # Maximize sum of radii
            circles_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-6}  # Moderate tolerances for speed
        )
        
        if result.success:
            circles = result.x.reshape(-1, 3)
        else:
            # If optimization fails, return original circles
            pass
    except Exception as e:
        # If optimization fails, return original circles
        pass
    
    return circles

def enhanced_refine_positions(circles):
    """Enhanced local refinement to maximize sum of radii with better convergence"""
    n = len(circles)
    
    # Use spatial data structure for faster neighbor queries
    tree = KDTree(circles[:, :2])
    
    # Controlled refinement process with early stopping and smarter updates
    for iteration in range(300):  # Reduced iterations to stay within time limits
        improved = False
        # Process circles in random order for better exploration
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            # Try to increase radius of circle i while maintaining constraints
            old_r = circles[i, 2]
            # Calculate maximum possible radius considering boundary constraints
            max_radius = min(
                circles[i, 0], 1 - circles[i, 0],
                circles[i, 1], 1 - circles[i, 1]
            )
            
            # Check overlap constraints with other circles using spatial index
            # Get neighbors within a reasonable distance to reduce computation
            neighbors = tree.query_ball_point(circles[i, :2], max_radius * 2)
            
            # Only check actual overlapping circles
            for j in neighbors:
                if i != j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    max_allowed = distance - r2
                    max_radius = min(max_radius, max_allowed)
            
            # Increase radius if possible - more conservative step
            if max_radius > old_r:
                # Use a smaller increment for better stability but still make progress
                increment = min(0.0015, (max_radius - old_r) * 0.5)  # Balanced approach
                new_r = old_r + increment
                circles[i, 2] = new_r
                improved = True
                
            # Adjust position if needed to accommodate larger radius
            if max_radius > old_r:
                # Keep center within bounds
                circles[i, 0] = np.clip(circles[i, 0], new_r, 1 - new_r)
                circles[i, 1] = np.clip(circles[i, 1], new_r, 1 - new_r)
        
        # Early stopping if no improvement for several iterations
        if not improved and iteration > 50:
            break
    
    return circles


# EVOLVE-BLOCK-END
