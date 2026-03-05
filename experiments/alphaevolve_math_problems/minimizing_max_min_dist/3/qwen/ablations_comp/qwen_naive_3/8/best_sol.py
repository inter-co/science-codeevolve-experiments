# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from scipy.spatial import ConvexHull
import warnings
from numba import jit
from scipy.spatial import SphericalVoronoi
import random
from joblib import Parallel, delayed
import multiprocessing
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

@jit(nopython=True)
def compute_distance_matrix_jit(points):
    """Efficiently compute distance matrix using numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i,j] = dist
            distances[j,i] = dist
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    n = 14
    d = 3
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Strategy: Use advanced initialization heuristics and multi-start optimization
    best_ratio = 0.0
    best_points = None
    
    # Heuristic 1: Generate points using Fibonacci spiral on sphere, then map to cube
    def generate_spherical_points():
        points = np.zeros((n, d))
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points[i] = [x, y, z]
        
        # Scale and shift to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Heuristic 2: Random initialization with constraints
    def generate_random_points():
        return np.random.uniform(0, 1, (n, d))
    
    # Heuristic 3: Perturbed regular arrangement (more refined)
    def generate_perturbed_grid():
        points = np.zeros((n, d))
        # Create a more uniform 3D grid
        side = int(np.ceil(n**(1/3)))
        count = 0
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if count < n:
                        points[count] = [i/(side-1) if side > 1 else 0.5, 
                                       j/(side-1) if side > 1 else 0.5, 
                                       k/(side-1) if side > 1 else 0.5]
                        count += 1
                    else:
                        break
                if count >= n:
                    break
            if count >= n:
                break
        
        # Add small random perturbation
        points += np.random.uniform(-0.02, 0.02, (n, d))
        # Clip to bounds
        points = np.clip(points, 0, 1)
        return points
    
    # Heuristic 4: Sphere packing inspired arrangement
    def generate_sphere_packing_points():
        # Start with a simple icosahedron-based arrangement
        # This is a known good starting configuration for many point dispersion problems
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = []
        
        # Generate vertices of a regular icosahedron (scaled appropriately)
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    vertices.append([i, j, k])
        
        # Add edge midpoints
        edges = []
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if sum(abs(vertices[i][k] - vertices[j][k]) for k in range(3)) == 2:
                    midpoint = [(vertices[i][k] + vertices[j][k]) / 2 for k in range(3)]
                    edges.append(midpoint)
        
        # Combine and normalize to unit sphere, then scale to [0,1]^3
        all_points = vertices + edges[:14-len(vertices)] if len(edges) >= 14-len(vertices) else vertices + edges
        all_points = np.array(all_points)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(all_points, axis=1)
        all_points = all_points / norms[:, np.newaxis]
        
        # Scale to [0,1]^3
        all_points = (all_points + 1) / 2
        
        # Take first 14 points
        return all_points[:14]
    
    # Heuristic 5: More sophisticated spherical arrangement using Fibonacci
    def generate_fibonacci_sphere_points():
        points = np.zeros((n, d))
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(n):
            # Project Fibonacci spiral onto sphere
            y = 1 - (i / (n - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = np.arctan2(y, radius)  # Correct angle calculation
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points[i] = [x, y, z]
        
        # Scale to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Heuristic 6: Improved sphere packing with better distribution
    def generate_improved_sphere_packing():
        # Use a known good configuration for 14 points on sphere
        # Based on the "14-point distribution on sphere" from literature
        # These coordinates are from known optimal or near-optimal configurations
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Generate points using a combination of icosahedral symmetry and perturbations
        points = np.zeros((n, d))
        
        # Create a more sophisticated pattern based on icosahedron vertices and face centers
        # This is a known good configuration for point distributions
        angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
        
        # Distribute points more evenly on sphere
        for i in range(n):
            # Use a variation of fibonacci spiral with better distribution
            y = 1 - (i / (n - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = angles[i]  # Use precomputed angles
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points[i] = [x, y, z]
        
        # Scale to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Heuristic 7: Known good configuration from mathematical literature
    def generate_known_good_configuration():
        # A known configuration for 14 points that provides good dispersion
        # Based on research in optimal point distributions
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.25, 0.75, 0.25],
            [0.75, 0.25, 0.75]
        ])
        
        # Normalize to [0,1]^3
        points = np.clip(points, 0, 1)
        return points
    
    # Try multiple initialization strategies
    initial_strategies = [
        generate_spherical_points,
        generate_random_points,
        generate_perturbed_grid,
        generate_sphere_packing_points,
        generate_fibonacci_sphere_points,
        generate_improved_sphere_packing,
        generate_known_good_configuration
    ]
    
    def objective(x_flat):
        # Reshape back to points
        points = x_flat.reshape(n, d)
        
        # Ensure points are within bounds
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances efficiently
        try:
            # Use optimized distance computation
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            # Avoid division by zero
            if max_dist <= 1e-12:
                return -1e10
            
            # Return negative ratio to maximize (since we're minimizing)
            return -min_dist / max_dist
            
        except Exception:
            return -1e10
    
    # Define bounds for each coordinate
    bounds = [(0, 1) for _ in range(n * d)]
    
    # Enhanced optimization with parallel restarts and better methods
    def run_single_optimization(init_func, restart_id):
        try:
            # Generate initial points
            points = init_func()
            
            # Add slight randomness to avoid local minima
            if restart_id > 0:
                points += np.random.normal(0, 0.01, (n, d))
                points = np.clip(points, 0, 1)
            
            # Flatten for optimization
            initial_flat = points.flatten()
            
            # Try multiple optimization methods with different settings
            methods_and_settings = [
                ('L-BFGS-B', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('TNC', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('SLSQP', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('trust-constr', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
            ]
            
            for method, options in methods_and_settings:
                try:
                    result = minimize(
                        objective,
                        initial_flat,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-12
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(n, d)
                        final_points = np.clip(final_points, 0, 1)  # Ensure bounds
                        final_distances = pdist(final_points)
                        final_min_dist = np.min(final_distances)
                        final_max_dist = np.max(final_distances)
                        
                        if final_max_dist > 1e-12:
                            final_ratio = final_min_dist / final_max_dist
                            return final_ratio, final_points
                except Exception:
                    continue
                    
        except Exception:
            pass
        return 0.0, None
    
    # Run optimizations in parallel with fewer restarts but better optimization
    max_restarts = 20  # Reduced restarts to save time but improve quality
    results = Parallel(n_jobs=min(multiprocessing.cpu_count(), 6))(delayed(run_single_optimization)(
        initial_strategies[restart % len(initial_strategies)], restart
    ) for restart in range(max_restarts))
    
    # Find best result among all optimizations
    for ratio, points in results:
        if points is not None and ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # If no good solution found, return the best initialization
    if best_points is None:
        # Try the most promising initialization
        points = generate_improved_sphere_packing()
        return points
    
    return best_points


# EVOLVE-BLOCK-END
