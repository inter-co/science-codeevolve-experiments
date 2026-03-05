# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from numba import jit
import random
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import warnings

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
    Uses a hybrid approach combining geometric initialization with advanced optimization techniques.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    max_time = 55  # Leave 5 seconds for final processing
    start_time = time.time()
    
    # Strategy 1: Initialize using a better geometric approach - hexagonal packing with perturbations
    initial_points = _initialize_hexagonal_packing_approach()
    
    # Strategy 2: Use advanced constrained optimization (trust-constr) with better settings
    optimized_points = _advanced_constrained_optimization(initial_points, max_time, start_time)
    
    # Strategy 3: Refinement using local search with multiple strategies
    refined_points = _local_search_refinement(optimized_points, max_time, start_time)
    
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


def _initialize_hexagonal_packing_approach():
    """
    Initialize points using a hexagonal packing approach for better uniformity
    This creates a more evenly distributed starting configuration
    """
    # Create points in a hexagonal lattice pattern
    # We'll use a 4x4 grid of points and slightly perturb them
    points = []
    
    # Hexagonal lattice parameters
    spacing = 0.25
    offset = 0.5
    
    # Generate points in a hexagonal pattern
    rows = 4
    cols = 4
    
    for i in range(rows):
        for j in range(cols):
            # Alternate row offset for hexagonal packing
            x_offset = (i % 2) * spacing / 2
            x = x_offset + j * spacing
            y = i * spacing * np.sqrt(3) / 2
            
            # Center in [0,1] range
            x = offset + (x - 0.5) * 1.5
            y = offset + (y - 0.5) * 1.5
            
            # Add small random perturbation
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            
            points.append([x, y])
    
    # Convert to numpy array and clip to [0,1] range
    points = np.array(points[:16])  # Ensure exactly 16 points
    points = np.clip(points, 0, 1)
    
    # Additional strategy: if any points are too close, apply repulsion
    for _ in range(20):
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # If any points are too close, move them apart
        for i, min_dist in enumerate(min_distances):
            if min_dist < 0.03:  # If too close
                # Move away from nearest neighbor
                nearest_idx = np.argmin(distances[i])
                dx = points[i, 0] - points[nearest_idx, 0]
                dy = points[i, 1] - points[nearest_idx, 1]
                norm = np.sqrt(dx*dx + dy*dy)
                if norm > 0:
                    # Move points apart
                    points[i, 0] += dx/norm * 0.005
                    points[i, 1] += dy/norm * 0.005
                    
        points = np.clip(points, 0, 1)
    
    return points


def _advanced_constrained_optimization(initial_points, max_time, start_time):
    """
    Use advanced constrained optimization with multiple approaches
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
    
    # Try multiple optimization methods with improved settings
    methods_to_try = [
        ('trust-constr', {'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-10}),
        ('L-BFGS-B', {'maxiter': 1000}),
        ('TNC', {'maxiter': 1000})
    ]
    
    best_result = None
    best_ratio = -np.inf
    
    for method, options in methods_to_try:
        if time.time() - start_time >= max_time:
            break
            
        try:
            result = minimize(
                objective,
                initial_flat,
                method=method,
                bounds=bounds,
                options=options,
                callback=lambda x: None  # Suppress output for cleaner run
            )
            
            if result.success and time.time() - start_time < max_time:
                # Reshape back to points
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                
                # Evaluate the result
                ratio = _compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = optimized_points.copy()
                    
        except Exception as e:
            continue
    
    # If no optimization succeeded, fallback to differential evolution with better parameters
    if best_result is None:
        try:
            # Use more iterations and better population size
            result = differential_evolution(
                objective,
                bounds,
                maxiter=300,
                popsize=50,
                seed=42,
                callback=lambda x, convergence: None,
                tol=1e-10
            )
            
            if time.time() - start_time < max_time:
                optimized_points = result.x.reshape(-1, 2)
                return np.clip(optimized_points, 0, 1)
        except Exception as e:
            pass
    
    # Return best optimization result or initial points
    if best_result is not None:
        return best_result
    else:
        return initial_points.copy()


def _local_search_refinement(initial_points, max_time, start_time):
    """
    Apply local search refinement to improve the solution
    Uses multiple strategies: gradient-based and stochastic approaches
    """
    points = initial_points.copy()
    current_ratio = _compute_min_max_ratio(points)
    
    # First, try a more effective gradient-based approach with better step sizes
    try:
        # Use a more systematic gradient descent approach
        for iteration in range(2000):
            if time.time() - start_time >= max_time:
                break
                
            # For each point, compute gradients and update
            for i in range(len(points)):
                if np.random.random() < 0.7:  # Update 70% of points each iteration
                    old_point = points[i].copy()
                    
                    # Simple gradient approximation by finite differences
                    epsilon = 1e-5
                    best_move = None
                    best_ratio = current_ratio
                    
                    # Try several small moves in different directions
                    moves = [
                        np.array([epsilon, 0]),
                        np.array([-epsilon, 0]),
                        np.array([0, epsilon]),
                        np.array([0, -epsilon]),
                        np.array([epsilon/np.sqrt(2), epsilon/np.sqrt(2)]),
                        np.array([-epsilon/np.sqrt(2), epsilon/np.sqrt(2)]),
                        np.array([epsilon/np.sqrt(2), -epsilon/np.sqrt(2)]),
                        np.array([-epsilon/np.sqrt(2), -epsilon/np.sqrt(2)]),
                        # Additional diagonal moves
                        np.array([epsilon, epsilon]),
                        np.array([-epsilon, epsilon]),
                        np.array([epsilon, -epsilon]),
                        np.array([-epsilon, -epsilon])
                    ]
                    
                    for move in moves:
                        test_points = points.copy()
                        test_points[i] += move
                        test_points[i] = np.clip(test_points[i], 0, 1)
                        
                        new_ratio = _compute_min_max_ratio(test_points)
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_move = move
                    
                    # Apply the best move if found
                    if best_move is not None:
                        points[i] += best_move
                        points[i] = np.clip(points[i], 0, 1)
                        current_ratio = best_ratio
                        
    except Exception as e:
        pass
    
    # Then do a more sophisticated simulated annealing approach with better parameters
    # Simulated annealing parameters
    temperature = 0.1
    cooling_rate = 0.9999
    min_temperature = 1e-10
    iterations_per_temp = 50
    
    best_points = points.copy()
    best_ratio = current_ratio
    
    # Keep track of recent improvements to detect stagnation
    recent_improvements = []
    
    while temperature > min_temperature and (time.time() - start_time) < max_time:
        for _ in range(iterations_per_temp):
            # Make a small random perturbation to one point
            idx = np.random.randint(0, len(points))
            new_points = points.copy()
            
            # Larger random move for more exploration
            move = np.random.normal(0, 0.005, 2)
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
                    recent_improvements = []  # Reset recent improvements
                else:
                    recent_improvements.append(current_ratio)
                    
                    # If no improvement in last 50 steps, reduce temperature faster
                    if len(recent_improvements) > 50:
                        if all(r <= recent_improvements[-50] for r in recent_improvements[-50:]):
                            cooling_rate = min(cooling_rate, 0.998)
        
        temperature *= cooling_rate
    
    # Final local optimization with gradient-based method
    try:
        # Simple local search to fine-tune
        for _ in range(500):
            if time.time() - start_time >= max_time:
                break
                
            for i in range(len(best_points)):
                # Try small moves in all directions
                best_move = None
                best_ratio = current_ratio
                
                moves = [
                    np.array([1e-4, 0]),
                    np.array([-1e-4, 0]),
                    np.array([0, 1e-4]),
                    np.array([0, -1e-4])
                ]
                
                for move in moves:
                    test_points = best_points.copy()
                    test_points[i] += move
                    test_points[i] = np.clip(test_points[i], 0, 1)
                    
                    new_ratio = _compute_min_max_ratio(test_points)
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_move = move
                
                if best_move is not None:
                    best_points[i] += best_move
                    best_points[i] = np.clip(best_points[i], 0, 1)
                    current_ratio = best_ratio
    except Exception as e:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
