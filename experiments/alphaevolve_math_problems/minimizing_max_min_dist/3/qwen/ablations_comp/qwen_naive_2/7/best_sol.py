# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import ConvexHull
import random
from numba import jit, prange
import time
from copy import deepcopy
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_min_max_ratio_jit(points):
    """JIT compiled version for faster computation"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(dist_sq)
            
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    if max_dist > 0:
        return min_dist / max_dist
    else:
        return 0.0


@jit(nopython=True, parallel=True)
def compute_distances_parallel(points):
    """Parallel computation of all pairwise distances"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    
    for i in prange(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(dist_sq)
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def compute_min_max_ratio_vectorized(points):
    """Vectorized computation of min/max ratio - faster for repeated evaluations"""
    distances = pdist(points)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    return min_dist / max_dist if max_dist > 0 else 0.0


def compute_min_max_ratio_with_distances(points, distances=None):
    """Compute ratio using precomputed distances for efficiency"""
    if distances is None:
        distances = pdist(points)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    return min_dist / max_dist if max_dist > 0 else 0.0


def generate_icosahedral_initialization(n_points: int) -> np.ndarray:
    """
    Generate initial points using icosahedral symmetry for better geometric distribution
    """
    # Vertices of regular icosahedron (normalized)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    ico_vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    
    ico_vertices = np.array(ico_vertices)
    norms = np.linalg.norm(ico_vertices, axis=1)
    ico_vertices = ico_vertices / norms[:, np.newaxis]
    
    # Select some vertices and add additional points via spherical sampling
    selected_indices = list(range(min(12, n_points)))
    points = ico_vertices[selected_indices].copy()
    
    # Fill remaining points with Fibonacci-like distribution on sphere
    if len(points) < n_points:
        # Use Fibonacci spiral with better spacing
        for i in range(len(points), n_points):
            # Improved Fibonacci distribution
            y = 1 - (i / (n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Use golden angle for better distribution
            golden_angle = 2.399963229728653  # 4*pi/(1+sqrt(5))
            phi_angle = i * golden_angle  # angle around z-axis
            
            x = radius * np.cos(phi_angle)
            z = radius * np.sin(phi_angle)
            
            points = np.vstack([points, [x, y, z]])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points


def generate_spherical_cap_initialization(n_points: int) -> np.ndarray:
    """
    Generate points using spherical cap arrangement for better distribution
    """
    # Use Fibonacci spiral on sphere with better parameterization
    points = []
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    for i in range(n_points):
        # Distribute points along a spiral on the sphere
        y = 1 - (i / (n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        # Better angle calculation for uniform distribution
        golden_angle = 2.399963229728653  # 4*pi/(1+sqrt(5))
        phi_angle = i * golden_angle  # angle around z-axis
        
        x = radius * np.cos(phi_angle)
        z = radius * np.sin(phi_angle)
        
        points.append([x, y, z])
    
    return np.array(points)


def generate_random_sphere_points(n_points: int) -> np.ndarray:
    """
    Generate random points on unit sphere
    """
    points = np.random.randn(n_points, 3)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    return points


def project_to_sphere(points):
    """Project points to unit sphere"""
    norms = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return points / norms[:, np.newaxis]


def advanced_local_search_improved(points: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Improved local search with better perturbation strategy and convergence detection
    """
    n = points.shape[0]
    current_points = points.copy()
    current_ratio = compute_min_max_ratio_jit(current_points)
    
    # Track best solution found
    best_points = current_points.copy()
    best_ratio = current_ratio
    
    # Use adaptive step sizes with better convergence criteria
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for iteration in range(max_iter):
        # Early stopping if already good enough
        if current_ratio > 0.495:
            break
            
        improvement_found = False
        
        # Try different step sizes
        for step_size in step_sizes:
            # Try multiple perturbation attempts per point
            for attempt in range(5):  # More attempts
                # Try perturbing each point
                for i in range(n):
                    # Try several random directions
                    for _ in range(5):  # More directions per point
                        direction = np.random.randn(3)
                        direction = direction / np.linalg.norm(direction)
                        
                        # Apply perturbation
                        temp_points = current_points.copy()
                        temp_points[i] += direction * step_size
                        
                        # Project back to sphere
                        norm = np.linalg.norm(temp_points[i])
                        if norm > 0:
                            temp_points[i] = temp_points[i] / norm
                        
                        # Check improvement
                        temp_ratio = compute_min_max_ratio_jit(temp_points)
                        
                        if temp_ratio > current_ratio:
                            current_points = temp_points
                            current_ratio = temp_ratio
                            improvement_found = True
                            
                            if temp_ratio > best_ratio:
                                best_ratio = temp_ratio
                                best_points = current_points.copy()
                            break
                
                if improvement_found:
                    break
            
            # If we made progress, continue with same step size
            if improvement_found:
                break
    
    return best_points


def improved_differential_evolution(points: np.ndarray, max_time: float = 10.0) -> np.ndarray:
    """
    Improved differential evolution with proper bounds and constraints
    """
    start_time = time.time()
    
    # Flatten the points array for optimization
    n_points = points.shape[0]
    bounds = [(-1.0, 1.0)] * (n_points * 3)
    
    def objective(x_flat):
        # Reshape back to 3D points
        points = x_flat.reshape(n_points, 3)
        # Project to sphere
        points = project_to_sphere(points)
        ratio = compute_min_max_ratio_jit(points)
        # We want to maximize ratio, so minimize negative ratio
        return -ratio
    
    # Use differential evolution with proper constraints and parameters
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,  # More iterations
            popsize=20,   # Larger population
            mutation=(0.8, 1.0),  # Different mutation
            recombination=0.9,    # Higher recombination
            seed=42,
            tol=1e-8,  # Tighter tolerance
            callback=lambda x, convergence: time.time() - start_time > max_time
        )
        
        # Reshape and project result
        optimized_points = result.x.reshape(n_points, 3)
        optimized_points = project_to_sphere(optimized_points)
        return optimized_points
    except Exception as e:
        # Fallback to simple local search if DE fails
        return advanced_local_search_improved(points, max_iter=100)


def hybrid_optimization_approach(n_points: int = 14, max_time: float = 55.0) -> np.ndarray:
    """
    Enhanced hybrid approach combining multiple optimization techniques
    """
    np.random.seed(42)
    random.seed(42)
    
    start_time = time.time()
    
    # Multi-start approach with different initialization strategies
    best_solution = None
    best_ratio = 0.0
    
    # Strategy 1: Icosahedral initialization with noise (most promising)
    for i in range(5):
        points = generate_icosahedral_initialization(n_points)
        noise = np.random.normal(0, 0.02, points.shape)  # Less noise
        points += noise
        points = project_to_sphere(points)
        
        # Local search refinement
        refined_points = advanced_local_search_improved(points, max_iter=100)
        ratio = compute_min_max_ratio_jit(refined_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = refined_points.copy()
    
    # Strategy 2: Spherical cap initialization
    for i in range(3):
        points = generate_spherical_cap_initialization(n_points)
        
        # Local search refinement
        refined_points = advanced_local_search_improved(points, max_iter=100)
        ratio = compute_min_max_ratio_jit(refined_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = refined_points.copy()
    
    # Strategy 3: Random points on sphere with DE optimization
    for i in range(3):
        points = generate_random_sphere_points(n_points)
        
        # Differential evolution refinement
        if time.time() - start_time < max_time - 15.0:
            de_points = improved_differential_evolution(points, max_time=8.0)
            ratio = compute_min_max_ratio_jit(de_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_solution = de_points.copy()
    
    # Strategy 4: Direct DE from icosahedral points (most promising starting point)
    if time.time() - start_time < max_time - 15.0:
        points = generate_icosahedral_initialization(n_points)
        de_points = improved_differential_evolution(points, max_time=15.0)
        ratio = compute_min_max_ratio_jit(de_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = de_points.copy()
    
    # Strategy 5: Additional local search rounds with different approaches
    if best_solution is not None and time.time() - start_time < max_time - 10.0:
        # Try a few more intense local searches
        for _ in range(5):
            if time.time() - start_time > max_time - 5.0:
                break
            refined = advanced_local_search_improved(best_solution, max_iter=150)
            ratio = compute_min_max_ratio_jit(refined)
            if ratio > best_ratio:
                best_ratio = ratio
                best_solution = refined.copy()
    
    return best_solution


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses hybrid optimization approach combining multiple strategies.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Use enhanced hybrid approach which combines multiple optimization strategies
    best_points = hybrid_optimization_approach(n_points=14, max_time=55.0)
    
    # Final refinement with enhanced local search
    final_points = advanced_local_search_improved(best_points, max_iter=150)
    
    return final_points


# EVOLVE-BLOCK-END
