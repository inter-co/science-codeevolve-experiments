# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import time
from scipy.optimize import minimize
from sklearn.cluster import KMeans

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining simulated annealing, local optimization, and 
    strategic initialization to improve upon the baseline.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # More sophisticated initialization using clustering and systematic placement
    def initialize_population():
        # Strategy 1: Cluster-based initialization
        # First, create a good starting configuration using a greedy approach
        circles = np.zeros((n, 3))
        
        # Place some circles systematically in a grid-like pattern
        rows = 5
        cols = 7
        padding = 0.05
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = padding + j * (1 - 2*padding) / (cols - 1) if cols > 1 else 0.5
                y = padding + i * (1 - 2*padding) / (rows - 1) if rows > 1 else 0.5
                
                # Add slight offset for better packing
                if i % 2 == 1 and cols > 1:
                    x += (1 - 2*padding) / (2 * (cols - 1))
                    
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Start with a reasonable initial radius
                    radius = 0.05 + 0.03 * np.random.random()
                    circles[idx] = [x, y, radius]
                    idx += 1
        
        # Fill remaining spots with random placement but avoiding dense areas
        while idx < n:
            # Try to place in less crowded areas
            attempts = 0
            placed = False
            while not placed and attempts < 100:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                radius = 0.03 + 0.04 * np.random.random()
                
                # Check if this location is relatively empty
                valid = True
                for k in range(idx):
                    cx, cy, cr = circles[k]
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < (radius + cr):
                        valid = False
                        break
                
                if valid:
                    circles[idx] = [x, y, radius]
                    placed = True
                attempts += 1
            
            if not placed:
                # Fallback to random placement
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                radius = 0.03 + 0.04 * np.random.random()
                circles[idx] = [x, y, radius]
            idx += 1
            
        return circles
    
    # Efficient collision checking using spatial data structure
    def check_collision(circles, i, j):
        """Check if two circles collide efficiently"""
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
        min_dist_sq = (r1 + r2)**2
        return dist_sq < min_dist_sq
    
    def get_collisions(circles):
        """Get list of colliding pairs efficiently"""
        collisions = []
        n = len(circles)
        for i in range(n):
            for j in range(i+1, n):
                if check_collision(circles, i, j):
                    collisions.append((i, j))
        return collisions
    
    # Fitness function with better penalty system
    def fitness(circles):
        # Sum of radii (higher is better)
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for violations
        penalty = 0
        
        # Boundary penalties - more severe than before
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 10000  # Large penalty for boundary violations
        
        # Overlap penalties with quadratic scaling for severity
        for i in range(n):
            for j in range(i+1, n):
                if check_collision(circles, i, j):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    overlap = (r1 + r2) - dist
                    penalty += 100000 * overlap**2  # Quadratic penalty for overlap severity
        
        return total_radius - penalty
    
    # Advanced local optimization using gradient-based approach
    def local_optimize(circles):
        # Create a copy to work with
        optimized = circles.copy()
        
        # Apply local improvements to increase radii where possible
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try to increase each radius
            for i in range(n):
                old_x, old_y, old_r = optimized[i]
                
                # Try to increase radius while maintaining constraints
                max_possible_radius = min(
                    old_x, 1 - old_x, 
                    old_y, 1 - old_y
                )
                
                if max_possible_radius <= old_r:
                    continue
                    
                # Try to increase radius gradually
                test_radius = min(old_r + 0.01, max_possible_radius)
                
                # Check if we can actually increase it without violating constraints
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
            
            # Apply boundary corrections
            for i in range(n):
                x, y, r = optimized[i]
                max_radius_x = min(x, 1 - x)
                max_radius_y = min(y, 1 - y)
                max_radius = min(max_radius_x, max_radius_y)
                optimized[i][2] = max(0.001, min(r, max_radius))
        
        return optimized
    
    # Simulated Annealing approach for better global optimization
    def simulated_annealing(initial_circles, max_time=55):
        current = initial_circles.copy()
        current_fitness = fitness(current)
        
        best = current.copy()
        best_fitness = current_fitness
        
        # Initial temperature and cooling schedule
        temp = 0.1
        cooling_rate = 0.999
        min_temp = 0.0001
        
        start_time = time.time()
        
        # Track how many steps we've taken
        step = 0
        
        while temp > min_temp and (time.time() - start_time) < max_time:
            step += 1
            
            # Create neighbor solution
            neighbor = current.copy()
            
            # Randomly select a circle to modify
            idx = np.random.randint(0, n)
            
            # Perturb position and radius
            neighbor[idx][0] = np.clip(neighbor[idx][0] + np.random.normal(0, 0.01), 0.01, 0.99)
            neighbor[idx][1] = np.clip(neighbor[idx][1] + np.random.normal(0, 0.01), 0.01, 0.99)
            neighbor[idx][2] = np.clip(neighbor[idx][2] + np.random.normal(0, 0.005), 0.001, 0.45)
            
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
            if step % 1000 == 0:
                temp = max(temp * 0.95, min_temp)
        
        return best
    
    # Main optimization process
    # Start with a good initialization
    initial_solution = initialize_population()
    
    # Apply local optimization first
    initial_solution = local_optimize(initial_solution)
    
    # Then run simulated annealing for global optimization
    final_solution = simulated_annealing(initial_solution)
    
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
        
        # Resolve any remaining overlaps more thoroughly
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
                    # Reduce both radii proportionally but prioritize keeping larger circles
                    reduction = overlap / 2.0
                    
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
