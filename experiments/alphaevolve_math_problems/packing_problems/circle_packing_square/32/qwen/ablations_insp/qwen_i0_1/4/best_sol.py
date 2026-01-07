# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import math
from sklearn.cluster import KMeans
import warnings
from numba import jit, prange
import random
from scipy.spatial import cKDTree
from itertools import combinations

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

@jit(nopython=True, parallel=True)
def compute_pairwise_distances_fast(circles):
    """Compute pairwise distances efficiently using Numba"""
    n = len(circles)
    distances = np.zeros((n, n))
    for i in prange(n):
        for j in range(i+1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, physics-inspired optimization, 
    and advanced constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Stage 1: Advanced initialization using multiple strategies
    best_circles = None
    best_sum = 0
    
    # Try multiple initialization strategies with different weights
    strategies = [
        ("hexagonal", 0.3),
        ("clustered", 0.3),
        ("random", 0.2),
        ("greedy", 0.2)
    ]
    
    for strategy_name, weight in strategies:
        if strategy_name == "hexagonal":
            circles = initialize_hexagonal_grid(n)
        elif strategy_name == "clustered":
            circles = initialize_clustered(n)
        elif strategy_name == "random":
            circles = initialize_random_better(n)
        elif strategy_name == "greedy":
            circles = initialize_greedy(n)
        
        # Stage 2: Advanced optimization with multiple restarts
        optimized = optimize_advanced(circles)
        
        # Keep the best solution
        current_sum = np.sum(optimized[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized.copy()
    
    return best_circles

def initialize_hexagonal_grid(n: int) -> np.ndarray:
    """Initialize circle positions using a high-quality hexagonal grid pattern."""
    # Create a hexagonal grid that's more tightly packed
    rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    cols = int(np.ceil(n / rows))
    
    # Adjust for better packing
    while rows * cols < n:
        rows += 1
        cols = int(np.ceil(n / rows))
    
    # Calculate spacing with better packing density
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Use a more precise hexagonal pattern
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = (i % 2) * spacing_x / 2
            x = (j + 0.5) * spacing_x + x_offset
            y = (i + 0.5) * spacing_y
            
            # Ensure we're within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Better radius estimation based on hexagonal packing
                # In hexagonal packing, the minimum distance is sqrt(3)/2 ≈ 0.866
                min_dist = min(x, 1-x, y, 1-y)
                # Estimate based on hexagonal packing efficiency - more aggressive
                max_radius = min_dist * 0.45  # Slightly more aggressive than before
                radius = max_radius
                
                circles.append([x, y, radius])
    
    # Fill remaining slots if needed
    while len(circles) < n:
        # Place randomly with better constraints
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        min_dist = min(x, 1-x, y, 1-y)
        # More aggressive radius estimate
        radius = min_dist * 0.3
        circles.append([x, y, radius])
    
    return np.array(circles[:n])

def initialize_clustered(n: int) -> np.ndarray:
    """Initialize circles using k-means clustering approach with better spread."""
    # Generate random points first
    points = np.random.rand(n, 2)
    
    # Use k-means to find cluster centers (we'll use fewer clusters for better distribution)
    k = min(8, n)  # Use fewer clusters to get well-separated centers
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(points)
    
    # Get cluster centers and assign radii
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Create circles with appropriate radii based on cluster density
    circles = []
    for i in range(min(len(centers), n)):
        x, y = centers[i]
        # Estimate radius based on proximity to other centers
        distances = [np.sqrt((x - centers[j][0])**2 + (y - centers[j][1])**2) 
                     for j in range(len(centers)) if j != i]
        if distances:
            min_dist = min(distances)
            # Radius should be about half the minimum distance to neighbors - more aggressive
            radius = min_dist / 2.5
        else:
            radius = 0.1
            
        # Ensure within bounds and make sure it's reasonable
        radius = min(radius, x, 1-x, y, 1-y)
        radius = max(radius, 0.01)
        
        circles.append([x, y, radius])
    
    # Fill any remaining slots with careful positioning
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        min_dist = min(x, 1-x, y, 1-y)
        radius = min_dist * 0.25
        circles.append([x, y, radius])
    
    return np.array(circles[:n])

def initialize_random_better(n: int) -> np.ndarray:
    """Better random initialization with constraint awareness."""
    circles = []
    
    # Start with some circles placed in strategic positions
    # Corners
    corner_positions = [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]
    # Edges (avoiding corners)
    edge_positions = [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]
    # Center
    center_positions = [(0.5, 0.5)]
    
    for x, y in corner_positions:
        if len(circles) < n:
            radius = min(x, 1-x, y, 1-y) * 0.35
            circles.append([x, y, radius])
    
    for x, y in edge_positions:
        if len(circles) < n:
            radius = min(x, 1-x, y, 1-y) * 0.3
            circles.append([x, y, radius])
    
    for x, y in center_positions:
        if len(circles) < n:
            radius = min(x, 1-x, y, 1-y) * 0.45
            circles.append([x, y, radius])
    
    # Fill remaining with random positions respecting constraints
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        # Estimate maximum possible radius considering nearby circles
        min_dist = min(x, 1-x, y, 1-y)
        radius = min_dist * 0.25
        circles.append([x, y, radius])
    
    return np.array(circles)

def initialize_greedy(n: int) -> np.ndarray:
    """Greedy initialization that places circles one by one maximizing available space."""
    circles = []
    
    # Place first circle in center
    circles.append([0.5, 0.5, 0.1])
    
    # Place remaining circles greedily
    for i in range(1, n):
        best_pos = None
        best_radius = 0
        best_score = -np.inf
        
        # Try many random positions to find good candidates
        for _ in range(2000):  # More attempts for better selection
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            
            # Find minimum distance to existing circles
            min_dist = min(x, 1-x, y, 1-y)  # Boundary distance
            if circles:
                for cx, cy, cr in circles:
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    min_dist = min(min_dist, dist - cr)  # Distance to closest circle
            
            # Score based on distance and boundary constraints
            # Higher score = better position
            score = min_dist  # Prefer more space
            
            if score > best_score:
                best_score = score
                best_radius = min_dist * 0.45  # More aggressive radius estimation
                best_pos = (x, y)
        
        if best_pos:
            circles.append([best_pos[0], best_pos[1], best_radius])
    
    return np.array(circles)

def create_optimization_variables(circles: np.ndarray) -> np.ndarray:
    """Convert circle data to optimization variables (x, y, r for each circle)."""
    return circles.flatten()

def unpack_optimization_variables(vars: np.ndarray) -> np.ndarray:
    """Convert optimization variables back to circle data."""
    return vars.reshape(-1, 3)

def objective_function(vars: np.ndarray) -> float:
    """Objective function to maximize sum of radii."""
    circles = unpack_optimization_variables(vars)
    return -np.sum(circles[:, 2])  # Negative because we minimize

def constraint_containment(vars: np.ndarray) -> np.ndarray:
    """Constraint function for containment (all circles within unit square)."""
    circles = unpack_optimization_variables(vars)
    n = len(circles)
    
    # Each circle must satisfy: r <= x <= 1-r and r <= y <= 1-r
    # This means: x - r >= 0, 1-x - r >= 0, y - r >= 0, 1-y - r >= 0
    constraints = []
    
    for i in range(n):
        x, y, r = circles[i]
        constraints.extend([
            x - r,           # x - r >= 0
            1 - x - r,       # 1 - x - r >= 0
            y - r,           # y - r >= 0
            1 - y - r        # 1 - y - r >= 0
        ])
    
    return np.array(constraints)

def constraint_overlaps(vars: np.ndarray) -> np.ndarray:
    """Constraint function for non-overlapping (distance >= sum of radii)."""
    circles = unpack_optimization_variables(vars)
    n = len(circles)
    
    # For each pair of circles, ensure distance >= sum of radii
    # This creates a constraint: distance - (r1 + r2) >= 0
    constraints = []
    
    # Use more efficient pairwise checking with early termination
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # We want: distance >= r1 + r2, which means: distance - (r1 + r2) >= 0
            constraints.append(distance - (r1 + r2))
    
    return np.array(constraints)

def optimize_advanced(initial_circles: np.ndarray) -> np.ndarray:
    """Advanced optimization using multiple techniques and restarts."""
    
    # Convert to optimization variables
    initial_vars = create_optimization_variables(initial_circles)
    
    # Define constraints
    # Containment constraints: all must be >= 0
    containment_cons = {
        'type': 'ineq',
        'fun': constraint_containment
    }
    
    # Overlap constraints: all must be >= 0
    overlap_cons = {
        'type': 'ineq', 
        'fun': constraint_overlaps
    }
    
    # Bounds for variables: [x, y, r] for each circle
    # x: [0, 1], y: [0, 1], r: [0, 0.5] (radius bounded by square size)
    bounds = []
    for i in range(len(initial_circles)):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Try multiple optimization methods with different settings
    methods_to_try = [
        ('trust-constr', {'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}),
        ('SLSQP', {'maxiter': 500, 'ftol': 1e-6})
    ]
    
    best_result = None
    best_sum = -np.inf
    
    # Try different optimization approaches
    for method, options in methods_to_try:
        try:
            result = minimize(
                objective_function,
                initial_vars,
                method=method,
                bounds=bounds,
                constraints=[containment_cons, overlap_cons],
                options=options,
                # callback=lambda x: print(f"Method {method} - Current objective: {-objective_function(x)}")
            )
            
            if result.success:
                current_sum = -objective_function(result.x)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            continue
    
    # If we found a good result, refine it with iterative improvement
    if best_result is not None:
        optimized_circles = unpack_optimization_variables(best_result.x)
        return iterative_improvement(optimized_circles)
    else:
        # Fallback to iterative improvement on initial configuration
        return iterative_improvement(initial_circles)

def iterative_improvement(circles: np.ndarray) -> np.ndarray:
    """Perform sophisticated iterative improvement."""
    # Ensure all circles are within bounds and have positive radii
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Keep circle within bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        r = max(r, 0.001)  # Ensure positive radius
        circles[i] = [x, y, r]
    
    # Iterative improvement with multiple phases
    improved = True
    iterations = 0
    max_iterations = 150
    
    # Precompute neighbor relationships for performance
    def get_neighbors(circle_idx, circles, radius_threshold=0.1):
        neighbors = []
        x, y, r = circles[circle_idx]
        for i in range(len(circles)):
            if i != circle_idx:
                nx, ny, nr = circles[i]
                dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                if dist < radius_threshold + r + nr:
                    neighbors.append(i)
        return neighbors
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Phase 1: Try to increase radii of individual circles
        for i in range(len(circles)):
            original = circles[i].copy()
            
            # Try to increase radius slightly
            test_radius = min(original[2] + 0.001, 0.5)
            
            # Check if we can increase radius without violating constraints
            valid = True
            for j in range(len(circles)):
                if i != j:
                    dist = np.sqrt((original[0] - circles[j][0])**2 + 
                                 (original[1] - circles[j][1])**2)
                    if dist < test_radius + circles[j][2]:
                        valid = False
                        break
            
            if valid:
                # Try to move the circle slightly to accommodate larger radius
                x_new = np.clip(original[0], test_radius, 1-test_radius)
                y_new = np.clip(original[1], test_radius, 1-test_radius)
                
                # Check if this movement still maintains constraints
                valid_move = True
                for j in range(len(circles)):
                    if i != j:
                        dist = np.sqrt((x_new - circles[j][0])**2 + 
                                     (y_new - circles[j][1])**2)
                        if dist < test_radius + circles[j][2]:
                            valid_move = False
                            break
                
                if valid_move:
                    circles[i] = [x_new, y_new, test_radius]
                    improved = True
                    continue
        
        # Phase 2: Try global improvements using a more intelligent approach
        if not improved and iterations < max_iterations - 50:  # Don't do this too late
            # Try to improve by moving circles closer to neighbors to allow bigger radii
            for i in range(len(circles)):
                original = circles[i].copy()
                
                # Check neighbors to see if we can move to make room for larger radius
                neighbors = get_neighbors(i, circles, 0.05)
                if len(neighbors) > 0:
                    # Try small moves that might free up space
                    best_move = None
                    best_radius = original[2]
                    best_improvement = 0
                    
                    # Try different small moves
                    moves = [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001)]
                    for dx, dy in moves:
                        x_test = np.clip(original[0] + dx, original[2], 1-original[2])
                        y_test = np.clip(original[1] + dy, original[2], 1-original[2])
                        
                        # Calculate how much we could potentially increase our radius
                        max_radius = min(x_test, 1-x_test, y_test, 1-y_test)
                        # Check what would happen with our neighbors
                        for j in neighbors:
                            nx, ny, nr = circles[j]
                            dist = np.sqrt((x_test - nx)**2 + (y_test - ny)**2)
                            if dist < max_radius + nr:
                                max_radius = dist - nr - 0.001  # Leave some margin
                        
                        if max_radius > original[2] and max_radius > best_radius:
                            best_radius = max_radius
                            best_move = (dx, dy)
                            best_improvement = max_radius - original[2]
                    
                    if best_move and best_improvement > 0.0001:
                        circles[i] = [original[0] + best_move[0], original[1] + best_move[1], best_radius]
                        improved = True
    
    # Final aggressive refinement with more thorough checks
    for _ in range(50):
        improved_local = False
        for i in range(len(circles)):
            original = circles[i].copy()
            # Try to increase radius slightly
            test_radius = min(original[2] + 0.0005, 0.5)
            
            # Check if we can increase radius without conflicts
            valid = True
            for j in range(len(circles)):
                if i != j:
                    dist = np.sqrt((original[0] - circles[j][0])**2 + 
                                 (original[1] - circles[j][1])**2)
                    if dist < test_radius + circles[j][2]:
                        valid = False
                        break
            
            if valid:
                circles[i] = [original[0], original[1], test_radius]
                improved_local = True
        
        if not improved_local:
            break
    
    return circles


# EVOLVE-BLOCK-END
