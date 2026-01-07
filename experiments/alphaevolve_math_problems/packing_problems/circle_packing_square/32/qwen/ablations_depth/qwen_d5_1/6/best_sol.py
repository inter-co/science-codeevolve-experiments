# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining advanced initialization, gradient-free optimization,
    and mathematical programming insights to achieve superior results.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Improved initialization with better spatial distribution
    def initialize_population():
        circles = np.zeros((n, 3))
        
        # Strategy: Create a more uniform initial distribution using a hexagonal lattice
        # This mimics good circle packing patterns
        rows = 6
        cols = 6
        padding = 0.05
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal packing pattern
                x = padding + j * (1 - 2*padding) / (cols - 1) if cols > 1 else 0.5
                y = padding + i * (1 - 2*padding) / (rows - 1) if rows > 1 else 0.5
                
                # Add slight offset for hexagonal pattern
                if i % 2 == 1 and cols > 1:
                    x += (1 - 2*padding) / (2 * (cols - 1))
                    
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Start with a more uniform initial radius
                    radius = 0.04 + 0.03 * np.random.random()
                    circles[idx] = [x, y, radius]
                    idx += 1
        
        # Fill remaining spots with more careful random placement
        while idx < n:
            # Try placing in a way that avoids clustering
            attempts = 0
            placed = False
            while not placed and attempts < 50:
                # Try to place in a region that's not too close to existing circles
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                radius = 0.03 + 0.04 * np.random.random()
                
                # Check proximity to existing circles for better spacing
                valid = True
                for k in range(idx):
                    cx, cy, cr = circles[k]
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    # Require minimum distance to be 1.5 times the sum of radii for better spacing
                    if dist < 1.5 * (radius + cr):
                        valid = False
                        break
                
                if valid:
                    circles[idx] = [x, y, radius]
                    placed = True
                attempts += 1
            
            if not placed:
                # Fallback to simple random placement
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                radius = 0.03 + 0.04 * np.random.random()
                circles[idx] = [x, y, radius]
            idx += 1
            
        return circles
    
    # Optimized collision checking using spatial data structure
    def check_collision_fast(circles, i, j, tree=None):
        """Fast collision checking with optional spatial indexing"""
        if tree is not None:
            # Use KDTree for fast neighborhood search
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_dist_sq = (r1 + r2)**2
            return dist_sq < min_dist_sq
        else:
            # Direct computation
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_dist_sq = (r1 + r2)**2
            return dist_sq < min_dist_sq
    
    # Better fitness function with proper penalty scaling
    def fitness(circles):
        # Sum of radii (higher is better)
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for violations - more sophisticated
        penalty = 0
        
        # Boundary penalties - severe penalty for boundary violations
        for i in range(n):
            x, y, r = circles[i]
            # Check if circle touches or crosses boundary
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 100000  # Very large penalty
        
        # Overlap penalties with proper scaling
        # Use spatial indexing for better performance on large numbers
        tree = cKDTree(circles[:, :2]) if n > 10 else None
        
        for i in range(n):
            for j in range(i+1, n):
                if check_collision_fast(circles, i, j, tree):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    overlap = (r1 + r2) - dist
                    # Quadratic penalty for overlap severity
                    penalty += 1000000 * overlap**2
        
        return total_radius - penalty
    
    # Enhanced local optimization with better strategies
    def local_optimize(circles):
        # Create a copy to work with
        optimized = circles.copy()
        
        # Apply iterative improvement with multiple strategies
        improved = True
        iterations = 0
        max_iterations = 50
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Strategy 1: Try to increase radii globally
            for i in range(n):
                old_x, old_y, old_r = optimized[i]
                
                # Compute maximum possible radius
                max_possible_radius = min(
                    old_x, 1 - old_x, 
                    old_y, 1 - old_y
                )
                
                if max_possible_radius <= old_r:
                    continue
                
                # Try to increase radius while respecting constraints
                test_radius = min(old_r + 0.005, max_possible_radius)
                
                # Check if increasing radius is valid
                valid = True
                for j in range(n):
                    if i != j:
                        x1, y1, r1 = optimized[i]
                        x2, y2, r2 = optimized[j]
                        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist < (test_radius + r2):
                            valid = False
                            break
                
                if valid and test_radius > old_r:
                    optimized[i][2] = test_radius
                    improved = True
            
            # Strategy 2: Small adjustments to positions to help with packing
            for i in range(n):
                # Only adjust if we're not near boundaries
                x, y, r = optimized[i]
                if r > 0.01 and x > 0.02 and x < 0.98 and y > 0.02 and y < 0.98:
                    # Try small position adjustments to improve packing
                    dx = np.random.normal(0, 0.002)
                    dy = np.random.normal(0, 0.002)
                    new_x = np.clip(x + dx, r, 1 - r)
                    new_y = np.clip(y + dy, r, 1 - r)
                    
                    # Check if this improves the configuration
                    valid = True
                    for j in range(n):
                        if i != j:
                            x1, y1, r1 = new_x, new_y, r
                            x2, y2, r2 = optimized[j]
                            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            if dist < (r1 + r2):
                                valid = False
                                break
                    
                    if valid:
                        optimized[i][0] = new_x
                        optimized[i][1] = new_y
                        improved = True
            
            # Apply boundary corrections after each iteration
            for i in range(n):
                x, y, r = optimized[i]
                max_radius_x = min(x, 1 - x)
                max_radius_y = min(y, 1 - y)
                max_radius = min(max_radius_x, max_radius_y)
                optimized[i][2] = max(0.001, min(r, max_radius))
        
        return optimized
    
    # A more effective optimization approach using a hybrid of strategies
    def advanced_optimization(initial_circles, max_time=55):
        current = initial_circles.copy()
        current_fitness = fitness(current)
        
        best = current.copy()
        best_fitness = current_fitness
        
        start_time = time.time()
        
        # Phase 1: Local optimization with many iterations
        for _ in range(100):
            if time.time() - start_time > max_time * 0.6:
                break
            current = local_optimize(current)
            new_fitness = fitness(current)
            if new_fitness > best_fitness:
                best = current.copy()
                best_fitness = new_fitness
        
        # Phase 2: Simulated Annealing with better parameters
        temp = 0.05
        cooling_rate = 0.9995
        min_temp = 0.00001
        
        step = 0
        while temp > min_temp and (time.time() - start_time) < max_time:
            step += 1
            
            # Create neighbor solution
            neighbor = current.copy()
            
            # Select multiple circles to modify for better exploration
            indices_to_modify = np.random.choice(n, size=min(5, n//2), replace=False)
            
            for idx in indices_to_modify:
                # Perturb position and radius
                neighbor[idx][0] = np.clip(neighbor[idx][0] + np.random.normal(0, 0.005), 0.01, 0.99)
                neighbor[idx][1] = np.clip(neighbor[idx][1] + np.random.normal(0, 0.005), 0.01, 0.99)
                neighbor[idx][2] = np.clip(neighbor[idx][2] + np.random.normal(0, 0.003), 0.001, 0.45)
            
            # Apply local optimization to the neighbor
            neighbor = local_optimize(neighbor)
            
            # Calculate fitness difference
            neighbor_fitness = fitness(neighbor)
            delta = neighbor_fitness - current_fitness
            
            # Accept or reject based on Metropolis criterion
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current = neighbor
                current_fitness = neighbor_fitness
                
                if current_fitness > best_fitness:
                    best = current.copy()
                    best_fitness = current_fitness
            
            # Cool down
            temp *= cooling_rate
            
            # Occasionally reset temperature to escape local optima
            if step % 500 == 0:
                temp = max(temp * 0.95, min_temp)
        
        return best
    
    # Main optimization process
    # Start with a good initialization
    initial_solution = initialize_population()
    
    # Apply aggressive local optimization first
    initial_solution = local_optimize(initial_solution)
    
    # Then run advanced optimization for global improvement
    final_solution = advanced_optimization(initial_solution)
    
    # Final validation and refinement
    def validate_and_refine(final_circles):
        # Ensure all circles fit properly
        valid_circles = final_circles.copy()
        
        # Enforce boundary constraints more strictly
        for i in range(n):
            x, y, r = valid_circles[i]
            max_radius_x = min(x, 1 - x)
            max_radius_y = min(y, 1 - y)
            max_radius = min(max_radius_x, max_radius_y)
            valid_circles[i][2] = max(0.001, min(r, max_radius))
        
        # Resolve any remaining overlaps more thoroughly with iterative refinement
        max_iterations = 100
        for iteration in range(max_iterations):
            changed = False
            # Check all pairs for overlaps
            for i, j in combinations(range(n), 2):
                x1, y1, r1 = valid_circles[i]
                x2, y2, r2 = valid_circles[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                min_dist = r1 + r2
                
                if dist < min_dist:
                    # Calculate overlap amount
                    overlap = min_dist - dist
                    # Reduce both radii proportionally but with more careful handling
                    reduction = overlap * 0.6  # Slightly more aggressive reduction
                    
                    # Try reducing radii
                    new_r1 = max(0.001, r1 - reduction)
                    new_r2 = max(0.001, r2 - reduction)
                    
                    # Only apply change if it's beneficial
                    if new_r1 < r1 or new_r2 < r2:
                        valid_circles[i][2] = new_r1
                        valid_circles[j][2] = new_r2
                        changed = True
            
            # Stop early if no changes
            if not changed:
                break
        
        return valid_circles
    
    final_circles = validate_and_refine(final_solution)
    
    return final_circles


# EVOLVE-BLOCK-END
