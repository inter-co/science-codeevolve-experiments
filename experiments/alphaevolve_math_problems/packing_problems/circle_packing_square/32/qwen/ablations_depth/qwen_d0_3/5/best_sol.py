# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random
from typing import Tuple, List
import time
from collections import defaultdict
import math
from itertools import combinations

# Global constants
N_CIRCLES = 32
MAX_ITERATIONS = 1000
TIME_LIMIT = 60.0

def validate_circles(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and non-overlapping"""
    n = len(circles)
    
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
            return False
    
    # Check non-overlap using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)
    
    for i in range(n):
        x, y, r = circles[i]
        # Find neighbors within distance 2*r (potential overlap)
        neighbors = tree.query_ball_point([x, y], 2*r)
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x-x2)**2 + (y-y2)**2)
                if distance < r + r2:
                    return False
    
    return True

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate sum of all radii"""
    return np.sum(circles[:, 2])

def initialize_random_placement() -> np.ndarray:
    """Initialize circles with random positions and small radii"""
    circles = np.zeros((N_CIRCLES, 3))
    for i in range(N_CIRCLES):
        # Random position
        circles[i, 0] = random.uniform(0.05, 0.95)  # x coordinate
        circles[i, 1] = random.uniform(0.05, 0.95)  # y coordinate
        # Small initial radius
        circles[i, 2] = random.uniform(0.01, 0.1)
    return circles

def compute_collision_graph(circles: np.ndarray) -> dict:
    """Build a graph of circle collisions for constraint propagation"""
    n = len(circles)
    collision_graph = defaultdict(list)
    
    points = circles[:, :2]
    tree = cKDTree(points)
    
    for i in range(n):
        x, y, r = circles[i]
        neighbors = tree.query_ball_point([x, y], 2*r)
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x-x2)**2 + (y-y2)**2)
                if distance < r + r2:
                    collision_graph[i].append(j)
    return collision_graph

def geometric_decomposition_optimization(circles: np.ndarray) -> np.ndarray:
    """
    Enhanced geometric decomposition with better conflict resolution and adaptive grid
    """
    # Create spatial grid for efficient collision detection
    grid_size = 16  # Increased grid resolution for better precision
    cell_size = 1.0 / grid_size
    grid = defaultdict(list)  # maps grid cells to circle indices
    
    # Assign circles to grid cells
    for i, (x, y, r) in enumerate(circles):
        # Determine which grid cells this circle touches
        min_col = max(0, int((x - r) / cell_size))
        max_col = min(grid_size - 1, int((x + r) / cell_size))
        min_row = max(0, int((y - r) / cell_size))
        max_row = min(grid_size - 1, int((y + r) / cell_size))
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                grid[(row, col)].append(i)
    
    # Iteratively improve the configuration using constraint propagation
    improved_circles = circles.copy()
    
    # For each circle, find its conflicts and try to resolve them
    for iter_count in range(100):  # More iterations for better convergence
        # Rebuild grid for current state
        grid.clear()
        for i, (x, y, r) in enumerate(improved_circles):
            min_col = max(0, int((x - r) / cell_size))
            max_col = min(grid_size - 1, int((x + r) / cell_size))
            min_row = max(0, int((y - r) / cell_size))
            max_row = min(grid_size - 1, int((y + r) / cell_size))
            
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    grid[(row, col)].append(i)
        
        # Process each circle
        for i in range(len(improved_circles)):
            x, y, r = improved_circles[i]
            
            # Find potentially conflicting circles in adjacent cells
            conflicts = set()
            min_col = max(0, int((x - r) / cell_size))
            max_col = min(grid_size - 1, int((x + r) / cell_size))
            min_row = max(0, int((y - r) / cell_size))
            max_row = min(grid_size - 1, int((y + r) / cell_size))
            
            for row in range(max(0, min_row - 1), min(grid_size, max_row + 2)):
                for col in range(max(0, min_col - 1), min(grid_size, max_col + 2)):
                    for j in grid[(row, col)]:
                        if i != j:
                            x2, y2, r2 = improved_circles[j]
                            distance = np.sqrt((x-x2)**2 + (y-y2)**2)
                            if distance < r + r2:
                                conflicts.add(j)
            
            # If there are conflicts, try to resolve them more effectively
            if conflicts:
                # Calculate minimum safe radius based on conflicts
                min_safe_radius = r
                for j in conflicts:
                    x2, y2, r2 = improved_circles[j]
                    distance = np.sqrt((x-x2)**2 + (y-y2)**2)
                    # Reduce radius to maintain separation with safety margin
                    min_safe_radius = min(min_safe_radius, (distance - r2) * 0.95)
                
                if min_safe_radius > 0.0001:  # Only adjust if significant
                    # Adjust radius to avoid conflicts
                    improved_circles[i, 2] = min_safe_radius
                    
                    # More sophisticated position adjustment
                    if len(conflicts) > 0:
                        # Move away from conflicting circles more intelligently
                        total_dx, total_dy = 0, 0
                        for j in conflicts:
                            x2, y2, r2 = improved_circles[j]
                            distance = np.sqrt((x-x2)**2 + (y-y2)**2)
                            if distance > 0:
                                # Move away from the conflicting circle
                                dx = (x - x2) / distance
                                dy = (y - y2) / distance
                                # Weight by inverse distance (closer conflicts matter more)
                                weight = 1.0 / (distance * distance + 1e-8)
                                total_dx += dx * weight
                                total_dy += dy * weight
                        
                        # Normalize and apply adjustment
                        magnitude = np.sqrt(total_dx*total_dx + total_dy*total_dy)
                        if magnitude > 0:
                            total_dx = total_dx / magnitude * 0.005
                            total_dy = total_dy / magnitude * 0.005
                            
                            # Apply adjustment with boundary checking
                            new_x = x + total_dx
                            new_y = y + total_dy
                            improved_circles[i, 0] = max(r, min(1-r, new_x))
                            improved_circles[i, 1] = max(r, min(1-r, new_y))
    
    return improved_circles

def multi_scale_refinement(circles: np.ndarray) -> np.ndarray:
    """
    Enhanced multi-scale refinement with better local search strategies
    """
    # Coarse optimization - larger steps
    coarse_circles = circles.copy()
    
    # Apply geometric decomposition optimization with coarser grid for speed
    coarse_circles = geometric_decomposition_optimization(coarse_circles)
    
    # Fine optimization - smaller adjustments with smarter strategies
    fine_circles = coarse_circles.copy()
    
    # Use enhanced local search with multiple strategies
    for iteration in range(50):
        # Create candidate solutions by perturbing circles
        candidates = []
        
        # Strategy 1: Random perturbations
        for i in range(N_CIRCLES):
            test_circles = fine_circles.copy()
            x, y, r = test_circles[i]
            
            # Small random movement with bounded step size
            dx = random.uniform(-0.008, 0.008)
            dy = random.uniform(-0.008, 0.008)
            dr = random.uniform(-0.003, 0.003)
            
            test_circles[i, 0] = max(0.01, min(0.99, x + dx))
            test_circles[i, 1] = max(0.01, min(0.99, y + dy))
            test_circles[i, 2] = min(0.5, max(0.001, r + dr))
            
            if validate_circles(test_circles):
                candidates.append((test_circles, calculate_radius_sum(test_circles)))
        
        # Strategy 2: Coordinate-wise optimization for selected circles
        # Pick some circles to optimize more aggressively
        sample_indices = random.sample(range(N_CIRCLES), min(8, N_CIRCLES))
        for i in sample_indices:
            # Try to increase radius while maintaining validity
            test_circles = fine_circles.copy()
            x, y, r = test_circles[i]
            
            # Try increasing radius slightly
            new_r = min(0.5, r * 1.05)  # 5% increase
            if new_r > r:
                test_circles[i, 2] = new_r
                
                # If valid, keep trying to increase further
                if validate_circles(test_circles):
                    candidates.append((test_circles, calculate_radius_sum(test_circles)))
            
            # Try moving circle to better location
            test_circles = fine_circles.copy()
            # Move toward center of mass of neighbors
            neighbors = []
            for j in range(N_CIRCLES):
                if i != j:
                    dx = test_circles[i, 0] - test_circles[j, 0]
                    dy = test_circles[i, 1] - test_circles[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance < 0.2:  # Close enough to consider
                        neighbors.append(j)
            
            if neighbors:
                avg_x, avg_y = 0, 0
                for j in neighbors:
                    avg_x += test_circles[j, 0]
                    avg_y += test_circles[j, 1]
                avg_x /= len(neighbors)
                avg_y /= len(neighbors)
                
                # Move towards average neighbor position
                dx = avg_x - test_circles[i, 0]
                dy = avg_y - test_circles[i, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                if distance > 0:
                    dx = dx / distance * 0.005
                    dy = dy / distance * 0.005
                    test_circles[i, 0] = max(0.01, min(0.99, test_circles[i, 0] + dx))
                    test_circles[i, 1] = max(0.01, min(0.99, test_circles[i, 1] + dy))
                
                if validate_circles(test_circles):
                    candidates.append((test_circles, calculate_radius_sum(test_circles)))
        
        # Select best candidate
        if candidates:
            best_candidate = max(candidates, key=lambda x: x[1])
            fine_circles = best_candidate[0]
    
    return fine_circles

def improved_voronoi_initialization() -> np.ndarray:
    """Improved Voronoi-like initialization with better spacing"""
    circles = np.zeros((N_CIRCLES, 3))
    
    # Create a more systematic grid pattern with better distribution
    sqrt_n = int(np.ceil(np.sqrt(N_CIRCLES)))
    grid_size = max(1, sqrt_n)
    
    # Use a hexagonal-like arrangement for better packing
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= N_CIRCLES:
                break
            # Offset every other row for better packing
            x_offset = (j % 2) * spacing_x * 0.5
            x = (i + 1) * spacing_x + x_offset
            y = (j + 1) * spacing_y
            
            # Add randomness but keep within bounds
            x += random.uniform(-spacing_x/6, spacing_x/6)
            y += random.uniform(-spacing_y/6, spacing_y/6)
            
            # Clamp to valid range
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Start with a reasonable initial radius
            circles[count] = [x, y, 0.05]
            count += 1
        if count >= N_CIRCLES:
            break
    
    # Fill remaining positions with random circles
    for i in range(count, N_CIRCLES):
        circles[i] = [random.uniform(0.05, 0.95), random.uniform(0.05, 0.95), 0.05]
    
    return circles

def adaptive_local_search(circles: np.ndarray) -> np.ndarray:
    """
    Adaptive local search that dynamically adjusts strategy based on situation
    """
    current = circles.copy()
    best_solution = current.copy()
    best_radius_sum = calculate_radius_sum(best_solution)
    
    # Different phases of optimization
    for phase in range(3):
        # Phase 1: Coarse adjustment
        if phase == 0:
            iterations = 20
            step_size = 0.01
        # Phase 2: Medium adjustment  
        elif phase == 1:
            iterations = 30
            step_size = 0.005
        # Phase 3: Fine adjustment
        else:
            iterations = 50
            step_size = 0.002
        
        for _ in range(iterations):
            # Try to improve by adjusting each circle
            for i in range(N_CIRCLES):
                # Save current state
                old_state = current[i].copy()
                
                # Try small perturbations
                test_circles = current.copy()
                x, y, r = test_circles[i]
                
                # Try to increase radius
                new_r = min(0.5, r + random.uniform(0.001, 0.005))
                test_circles[i, 2] = new_r
                
                # Try to adjust position
                dx = random.uniform(-step_size, step_size)
                dy = random.uniform(-step_size, step_size)
                test_circles[i, 0] = max(0.01, min(0.99, x + dx))
                test_circles[i, 1] = max(0.01, min(0.99, y + dy))
                
                # Check if improvement
                if validate_circles(test_circles):
                    new_sum = calculate_radius_sum(test_circles)
                    if new_sum > best_radius_sum:
                        best_solution = test_circles.copy()
                        best_radius_sum = new_sum
                        current = test_circles.copy()
                    else:
                        # Accept with probability based on how much worse it is
                        delta = new_sum - calculate_radius_sum(current)
                        if random.random() < np.exp(delta * 10):  # Temperature parameter
                            current = test_circles.copy()
    
    return best_solution

def evolve_circles() -> np.ndarray:
    """Enhanced main evolution algorithm with multiple strategies"""
    start_time = time.time()
    
    # Method 1: Improved Voronoi-based initialization
    circles = improved_voronoi_initialization()
    
    # Method 2: Multiple rounds of optimization with different strategies
    for iteration in range(200):  # More iterations overall
        if time.time() - start_time > TIME_LIMIT * 0.8:
            break
            
        # Apply geometric decomposition optimization
        circles = geometric_decomposition_optimization(circles)
        
        # Apply multi-scale refinement
        circles = multi_scale_refinement(circles)
        
        # Occasionally run adaptive local search
        if iteration % 10 == 0:
            circles = adaptive_local_search(circles)
    
    # Final validation and optimization
    if not validate_circles(circles):
        # Re-initialize if invalid
        circles = improved_voronoi_initialization()
    
    # Final enhancement passes
    circles = multi_scale_refinement(circles)
    circles = adaptive_local_search(circles)
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        circles = evolve_circles()
        # Validate final result
        if not validate_circles(circles):
            # Fallback to simple initialization
            circles = initialize_random_placement()
            
        # Final optimization pass
        circles = multi_scale_refinement(circles)
        circles = adaptive_local_search(circles)
        
        return circles
    except Exception as e:
        # Fallback to basic approach
        print(f"Error in evolution: {e}")
        return initialize_random_placement()


# EVOLVE-BLOCK-END
