# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial.distance import cdist
from numba import jit
import time
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import warnings

@jit(nopython=True)
def check_overlap_and_boundaries_fast(positions, radii, n):
    """Fast check for overlap and boundary constraints"""
    for i in range(n):
        # Check boundary constraints
        if (positions[i, 0] - radii[i] < 0 or 
            positions[i, 0] + radii[i] > 1 or 
            positions[i, 1] - radii[i] < 0 or 
            positions[i, 1] + radii[i] > 1):
            return False
        
        # Check overlap constraints with early termination
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance_squared = dx*dx + dy*dy
            radii_sum = radii[i] + radii[j]
            if distance_squared < radii_sum * radii_sum:
                return False
    return True

@jit(nopython=True)
def compute_min_distances_fast(positions, radii, n):
    """Fast computation of minimum distances"""
    min_distances = np.full(n, np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance_squared = dx*dx + dy*dy
                distance = np.sqrt(distance_squared)
                min_distances[i] = min(min_distances[i], distance)
    return min_distances

@jit(nopython=True)
def get_max_safe_radii_fast(positions, radii, n):
    """Fast calculation of maximum safe radii for all circles"""
    max_radii = np.zeros(n)
    min_distances = compute_min_distances_fast(positions, radii, n)
    
    for i in range(n):
        # Maximum radius without overlapping others
        max_radius = min_distances[i] / 2.0
        
        # Maximum radius without going out of bounds
        max_radius = min(max_radius, 
                        positions[i, 0], 1-positions[i, 0],
                        positions[i, 1], 1-positions[i, 1])
        
        max_radii[i] = max_radius
    
    return max_radii

def initialize_positions_hexagonal(n):
    """Initialize positions using a hexagonal packing approach for better distribution"""
    # For 32 circles, we can arrange them in a roughly hexagonal pattern
    # We'll create a grid with slight offset for hexagonal arrangement
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Create hexagonal grid
    positions = []
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = 0.5 if i % 2 == 1 else 0.0
            x = spacing_x * (j + 1) + x_offset * spacing_x * 0.5
            y = spacing_y * (i + 1)
            
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            positions.append([x, y])
    
    # If we don't have enough positions, fill with random ones
    while len(positions) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        positions.append([x, y])
    
    return np.array(positions[:n])

def initialize_positions_clustered(n):
    """Initialize positions using k-means clustering for better distribution"""
    # Generate initial candidate points
    candidates = []
    grid_size = max(3, int(np.ceil(np.sqrt(n))))
    
    # Create a more uniform distribution
    for i in range(grid_size):
        for j in range(grid_size):
            x = 0.1 + 0.8 * i / (grid_size - 1) if grid_size > 1 else 0.5
            y = 0.1 + 0.8 * j / (grid_size - 1) if grid_size > 1 else 0.5
            candidates.append([x, y])
    
    # If we need more points, generate random ones
    if len(candidates) < n:
        for _ in range(n - len(candidates)):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            candidates.append([x, y])
    
    # Use k-means to cluster the points
    if len(candidates) >= n:
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(candidates[:n])
        positions = kmeans.cluster_centers_
    else:
        positions = np.array(candidates)
    
    # Ensure positions are within bounds
    positions[:, 0] = np.clip(positions[:, 0], 0.05, 0.95)
    positions[:, 1] = np.clip(positions[:, 1], 0.05, 0.95)
    
    return positions

def adaptive_optimization_step(positions, radii, n, step_size=0.01):
    """Perform adaptive optimization steps"""
    # Get current max safe radii
    max_radii = get_max_safe_radii_fast(positions, radii, n)
    
    improved = False
    
    # First try to increase radii where possible
    for i in range(n):
        if max_radii[i] > radii[i]:
            # Increase radius by a fraction of available space
            delta_radius = min(step_size, max_radii[i] - radii[i])
            if delta_radius > 1e-8:
                radii[i] = min(max_radii[i], radii[i] + delta_radius)
                improved = True
    
    # If no radii increased, try local moves
    if not improved:
        # Try small local moves for each circle
        for i in range(n):
            # Try a small random perturbation
            dx = np.random.uniform(-step_size*0.5, step_size*0.5)
            dy = np.random.uniform(-step_size*0.5, step_size*0.5)
            
            new_x = positions[i, 0] + dx
            new_y = positions[i, 1] + dy
            
            # Check if new position is valid
            if (0.05 <= new_x <= 0.95 and 0.05 <= new_y <= 0.95):
                # Temporarily move circle
                old_x, old_y = positions[i, 0], positions[i, 1]
                positions[i, 0] = new_x
                positions[i, 1] = new_y
                
                # Check if this improves the configuration
                if check_overlap_and_boundaries_fast(positions, radii, n):
                    # Accept the move
                    improved = True
                else:
                    # Reject the move
                    positions[i, 0] = old_x
                    positions[i, 1] = old_y
    
    return improved

def fast_local_search(positions, radii, n, max_iterations=1000):
    """Fast local search optimization"""
    for iteration in range(max_iterations):
        improved = adaptive_optimization_step(positions, radii, n, 0.01)
        if not improved:
            break
    return positions, radii

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining smart initialization and iterative refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    best_sum_radii = 0
    best_circles = None
    
    # Multi-start approach with different initialization strategies
    init_strategies = [
        lambda: initialize_positions_hexagonal(n),
        lambda: initialize_positions_clustered(n),
        lambda: np.random.rand(n, 2) * 0.8 + 0.1  # Random within bounds
    ]
    
    # Run multiple optimization attempts
    for start_iteration in range(15):
        # Select initialization strategy
        strategy_idx = start_iteration % len(init_strategies)
        np.random.seed(start_iteration * 100 + 42)
        
        # Initialize positions
        positions = init_strategies[strategy_idx]()
        
        # Initialize with reasonable starting radii
        # Start with smaller radii to allow for growth
        radii = np.full(n, 0.03) 
        
        # Apply optimization
        refined_positions, refined_radii = fast_local_search(positions, radii, n)
        
        # Validate final configuration
        if check_overlap_and_boundaries_fast(refined_positions, refined_radii, n):
            current_sum_radii = np.sum(refined_radii)
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles = np.column_stack([refined_positions, refined_radii])
        else:
            # If invalid, try a different approach
            pass
    
    # If we didn't find a good solution, use a more robust fallback
    if best_circles is None:
        # Use hexagonal initialization with more careful optimization
        positions = initialize_positions_hexagonal(n)
        radii = np.full(n, 0.02)
        
        # More aggressive optimization
        for _ in range(500):
            max_radii = get_max_safe_radii_fast(positions, radii, n)
            improved = False
            for i in range(n):
                if max_radii[i] > radii[i]:
                    new_radius = min(max_radii[i], radii[i] + 0.005)
                    if new_radius > radii[i]:
                        radii[i] = new_radius
                        improved = True
            if not improved:
                break
        
        best_circles = np.column_stack([positions, radii])
    
    # Final validation and cleanup
    circles = best_circles.copy()
    
    # Ensure all constraints are satisfied with some margin
    max_iter = 30
    for _ in range(max_iter):
        valid = True
        for i in range(n):
            # Check boundary constraints
            boundary_safe = (
                circles[i, 0] - circles[i, 2] >= 0.001 and
                circles[i, 0] + circles[i, 2] <= 0.999 and
                circles[i, 1] - circles[i, 2] >= 0.001 and
                circles[i, 1] + circles[i, 2] <= 0.999
            )
            
            if not boundary_safe:
                valid = False
                break
                
            # Check overlap constraints
            for j in range(n):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist_squared = dx*dx + dy*dy
                    radii_sum = circles[i, 2] + circles[j, 2]
                    if dist_squared < (radii_sum * 0.99)**2:  # Add small safety margin
                        valid = False
                        break
            
            if not valid:
                break
        
        if valid:
            break
            
        # If not valid, try to fix by reducing radii and adjusting positions
        for i in range(n):
            # Reduce radius slightly to ensure separation
            circles[i, 2] *= 0.98
            # Clamp to valid range
            circles[i, 2] = max(0.001, min(0.45, circles[i, 2]))
    
    return circles


# EVOLVE-BLOCK-END
