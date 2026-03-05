# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from numba import jit
import random
from sklearn.cluster import KMeans

@jit(nopython=True)
def compute_distances_jit(points):
    """Compute pairwise distances efficiently using Numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization with advanced optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    max_time = 55  # Leave 5 seconds for final processing
    start_time = time.time()
    
    # Strategy 1: Initialize using a better geometric approach - regular grid with perturbations
    initial_points = _initialize_better_grid_approach()
    
    # Strategy 2: Use advanced constrained optimization (L-BFGS-B) for global search
    optimized_points = _advanced_global_optimization(initial_points, max_time, start_time)
    
    # Strategy 3: Local refinement with improved simulated annealing
    refined_points = _improved_simulated_annealing(optimized_points, max_time, start_time)
    
    return refined_points


def _compute_min_max_ratio(points):
    """Compute the min/max distance ratio efficiently"""
    if len(points) < 2:
        return 0
    
    # Use scipy for distance computation
    distances = pdist(points)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist == 0:
        return 0
    
    return min_dist / max_dist


def _initialize_better_grid_approach():
    """
    Initialize points using a better geometric approach:
    1. Start with a regular grid pattern
    2. Add small random perturbations to avoid degenerate cases
    3. Ensure good coverage of the space
    """
    # Create a 4x4 grid pattern (16 points total)
    grid_size = 4
    points = []
    
    # Generate regular grid points
    for i in range(grid_size):
        for j in range(grid_size):
            x = i / (grid_size - 1) if grid_size > 1 else 0.5
            y = j / (grid_size - 1) if grid_size > 1 else 0.5
            points.append([x, y])
    
    points = np.array(points)
    
    # Add small random perturbations to break symmetry and improve distribution
    noise_scale = 0.02
    noise = np.random.normal(0, noise_scale, points.shape)
    points += noise
    
    # Ensure all points are within [0,1] x [0,1] bounds
    points = np.clip(points, 0, 1)
    
    return points


def _advanced_global_optimization(initial_points, max_time, start_time):
    """
    Use advanced global optimization with multiple strategies
    """
    # Flatten initial points for optimization
    initial_flat = initial_points.flatten()
    
    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates
    
    def objective(params):
        # Reshape parameters back to 2D array
        points = params.reshape(-1, 2)
        
        # Compute distances
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return -np.inf  # Penalize invalid configurations
        
        # Return negative because we want to maximize ratio
        return -min_dist / max_dist
    
    # Try multiple optimization methods with early stopping
    methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
    
    for method in methods_to_try:
        if time.time() - start_time >= max_time:
            break
            
        try:
            result = minimize(
                objective,
                initial_flat,
                method=method,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-6},
                callback=lambda x: None  # Suppress callbacks for performance
            )
            
            if result.success and time.time() - start_time < max_time:
                # Reshape back to points
                optimized_points = result.x.reshape(-1, 2)
                return np.clip(optimized_points, 0, 1)
        except Exception as e:
            continue
    
    # Fallback to differential evolution if other methods fail
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=150,
            popsize=20,
            seed=42,
            callback=lambda x, convergence: None  # Suppress callbacks
        )
        
        if time.time() - start_time < max_time:
            optimized_points = result.x.reshape(-1, 2)
            return np.clip(optimized_points, 0, 1)
    except Exception as e:
        pass
    
    # If all optimization fails, return initial points
    return initial_points.copy()


def _improved_simulated_annealing(initial_points, max_time, start_time):
    """
    Apply improved simulated annealing for fine-tuning the solution
    """
    points = initial_points.copy()
    current_ratio = _compute_min_max_ratio(points)
    
    # Improved simulated annealing parameters
    temperature = 0.1
    cooling_rate = 0.999
    min_temperature = 1e-6
    iterations_per_temp = 50
    
    best_points = points.copy()
    best_ratio = current_ratio
    
    iteration_count = 0
    max_iterations = 10000
    
    while temperature > min_temperature and (time.time() - start_time) < max_time and iteration_count < max_iterations:
        for _ in range(iterations_per_temp):
            # Make a small random perturbation to multiple points (not just one)
            num_changes = np.random.randint(1, 4)  # Change 1-3 points each iteration
            new_points = points.copy()
            
            for _ in range(num_changes):
                idx = np.random.randint(0, len(points))
                # Smaller moves for better convergence
                move = np.random.normal(0, 0.002, 2)
                new_points[idx] += move
                new_points[idx] = np.clip(new_points[idx], 0, 1)
            
            # Calculate new ratio
            new_ratio = _compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.random() < np.exp((new_ratio - current_ratio) / temperature):
                points = new_points
                current_ratio = new_ratio
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = points.copy()
        
        temperature *= cooling_rate
        iteration_count += 1
    
    return best_points


# EVOLVE-BLOCK-END
