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
    Uses a hybrid approach combining spherical optimization with constrained optimization techniques.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    max_time = 55  # Leave 5 seconds for final processing
    start_time = time.time()
    
    # Strategy 1: Initialize using a more sophisticated geometric approach
    initial_points = _initialize_spherical_approach()
    
    # Strategy 2: Use advanced constrained optimization (trust-constr)
    optimized_points = _advanced_constrained_optimization(initial_points, max_time, start_time)
    
    # Strategy 3: Refinement using simulated annealing for better local search
    refined_points = _simulated_annealing_refinement(optimized_points, max_time, start_time)
    
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


def _initialize_spherical_approach():
    """
    Initialize points using a spherical approach:
    1. Distribute points on a sphere using Fibonacci lattice
    2. Project to 2D plane with appropriate scaling
    3. Apply perturbations to improve distribution
    """
    # Generate Fibonacci lattice points on sphere (approximation)
    n = 16
    points_3d = []
    
    # Fibonacci spiral approach for even distribution
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    for i in range(n):
        z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
        radius = np.sqrt(1 - z*z)
        theta = np.arccos(z)
        phi_angle = (i * 2 * np.pi) / phi
        
        # Convert to Cartesian coordinates
        x = radius * np.cos(phi_angle)
        y = radius * np.sin(phi_angle)
        z = z
        
        points_3d.append([x, y, z])
    
    points_3d = np.array(points_3d)
    
    # Project to 2D using stereographic projection or simple 3D to 2D
    # We'll use the first two coordinates and scale appropriately
    points_2d = points_3d[:, :2]
    
    # Normalize to unit circle
    norms = np.linalg.norm(points_2d, axis=1)
    points_2d = points_2d / np.max(norms) * 0.8  # Scale down slightly
    
    # Add some random perturbation to break symmetry
    noise = np.random.normal(0, 0.02, points_2d.shape)
    points_2d += noise
    
    # Transform to [0,1] x [0,1] space
    points_2d = (points_2d + 1) * 0.5  # Map from [-1,1] to [0,1]
    
    # Ensure all points are within bounds
    points_2d = np.clip(points_2d, 0, 1)
    
    return points_2d


def _advanced_constrained_optimization(initial_points, max_time, start_time):
    """
    Use advanced constrained optimization with trust-constr method
    This approach directly optimizes the objective function with proper constraints
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
    
    # Constraints to keep points within bounds are already handled by bounds
    
    # Use trust-constr method which is good for this type of problem
    try:
        result = minimize(
            objective,
            initial_flat,
            method='trust-constr',
            bounds=bounds,
            options={'maxiter': 500, 'gtol': 1e-6, 'ftol': 1e-6},
            callback=lambda x: print(f"Optimization progress: {time.time() - start_time:.1f}s")
        )
        
        if result.success and time.time() - start_time < max_time:
            # Reshape back to points
            optimized_points = result.x.reshape(-1, 2)
            return np.clip(optimized_points, 0, 1)
    except Exception as e:
        print(f"Trust-constr optimization failed: {e}")
    
    # Fallback to differential evolution if trust-constr fails
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            callback=lambda x, convergence: print(f"DE progress: {time.time() - start_time:.1f}s")
        )
        
        if time.time() - start_time < max_time:
            optimized_points = result.x.reshape(-1, 2)
            return np.clip(optimized_points, 0, 1)
    except Exception as e:
        print(f"Differential evolution also failed: {e}")
    
    # If all optimization fails, return initial points
    return initial_points.copy()


def _simulated_annealing_refinement(initial_points, max_time, start_time):
    """
    Apply simulated annealing for fine-tuning the solution
    This provides better local search than simple gradient methods
    """
    points = initial_points.copy()
    current_ratio = _compute_min_max_ratio(points)
    
    # Simulated annealing parameters
    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 1e-8
    iterations_per_temp = 100
    
    best_points = points.copy()
    best_ratio = current_ratio
    
    while temperature > min_temperature and (time.time() - start_time) < max_time:
        for _ in range(iterations_per_temp):
            # Make a small random perturbation to one point
            idx = np.random.randint(0, len(points))
            new_points = points.copy()
            
            # Small random move
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
        
        temperature *= cooling_rate
    
    return best_points


# EVOLVE-BLOCK-END
