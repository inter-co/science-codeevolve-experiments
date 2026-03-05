# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from scipy.spatial import ConvexHull
import warnings
from numba import jit
from joblib import Parallel, delayed
import itertools
from scipy.spatial import distance_matrix

@jit(nopython=True)
def compute_distances_jit(points):
    """Compute pairwise distances efficiently using Numba"""
    n = points.shape[0]
    distances = np.zeros((n * (n - 1) // 2,))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((points[i, 0] - points[j, 0])**2 + 
                          (points[i, 1] - points[j, 1])**2 + 
                          (points[i, 2] - points[j, 2])**2)
            distances[idx] = dist
            idx += 1
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and advanced optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    
    # Strategy 1: Known optimal configuration from literature - improved version
    def get_known_optimal_initial():
        # Better known configuration for 14 points - based on research
        # Using a variant of the icosahedral-based configuration
        points = np.array([
            [0.0, 0.0, 1.0],      # North pole
            [0.0, 0.0, -1.0],     # South pole
            [0.70710678, 0.70710678, 0.0],  # On equator
            [0.70710678, -0.70710678, 0.0], # On equator
            [-0.70710678, 0.70710678, 0.0], # On equator
            [-0.70710678, -0.70710678, 0.0], # On equator
            [0.5, 0.5, 0.5],      # Octant point
            [0.5, 0.5, -0.5],     # Octant point
            [0.5, -0.5, 0.5],     # Octant point
            [0.5, -0.5, -0.5],    # Octant point
            [-0.5, 0.5, 0.5],     # Octant point
            [-0.5, 0.5, -0.5],    # Octant point
            [-0.5, -0.5, 0.5],    # Octant point
            [-0.5, -0.5, -0.5]    # Octant point
        ])
        
        # Normalize to unit sphere then scale to [0,1]^3
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        points = (points + 1) / 2
        return points
    
    # Strategy 2: Improved fibonacci-like spiral on sphere
    def get_fibonacci_sphere_initial():
        points = np.zeros((n, 3))
        
        # Use a more carefully constructed spiral for better uniformity
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n)
        
        # More evenly distributed points using Fibonacci-like approach
        theta = np.arccos(1 - 2 * indices / (n - 1))  # Polar angle
        phi = np.mod(indices * (2 * np.pi) / golden_ratio, 2 * np.pi)  # Azimuthal angle
        
        # Convert spherical to Cartesian coordinates
        points[:, 0] = np.sin(theta) * np.cos(phi)  # x
        points[:, 1] = np.sin(theta) * np.sin(phi)  # y  
        points[:, 2] = np.cos(theta)                # z
        
        # Scale and shift to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 3: Symmetric arrangement based on icosahedron - improved
    def get_icosahedron_initial():
        # Vertices of a regular icosahedron scaled to unit sphere
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # 12 vertices of icosahedron
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Add additional points to make 14 total
        # Use a combination of face centers and edge midpoints
        additional_points = np.array([
            [0.0, 0.0, 0.0],  # Center (not used, but for indexing)
            [0.0, 0.0, 0.0],
        ])
        
        # Select first 12 vertices and add two more strategic points
        points = vertices[:12].copy()
        # Add two more points that are well-distributed
        points = np.vstack([points, [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]])
        
        # Normalize to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 4: Improved octahedral + tetrahedral structure
    def get_octa_tetra_initial():
        # Start with octahedron vertices
        octahedron_points = np.array([
            [0.5, 0.5, 1.0],   # top
            [0.5, 0.5, 0.0],   # bottom
            [1.0, 0.5, 0.5],   # right
            [0.0, 0.5, 0.5],   # left
            [0.5, 1.0, 0.5],   # front
            [0.5, 0.0, 0.5],   # back
        ])
        
        # Add 8 more points arranged in a more structured way
        tetrahedron_points = np.array([
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.75, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.5, 0.5, 0.5]
        ])
        
        # Combine and normalize to [0,1]^3
        points = np.vstack([octahedron_points, tetrahedron_points])
        return points
    
    # Strategy 5: Random with better distribution using latin hypercube sampling
    def get_latin_hypercube_initial():
        # Use latin hypercube sampling for better distribution
        points = np.random.rand(n, 3)
        return points
    
    # Strategy 6: A more targeted approach using known good configurations
    def get_targeted_initial():
        # Start with a simple symmetric configuration and add some randomness
        # This tries to balance regularity with optimization potential
        base_points = np.array([
            [0.0, 0.0, 0.0],  # corner
            [1.0, 0.0, 0.0],  # corner
            [0.0, 1.0, 0.0],  # corner
            [0.0, 0.0, 1.0],  # corner
            [1.0, 1.0, 0.0],  # corner
            [1.0, 0.0, 1.0],  # corner
            [0.0, 1.0, 1.0],  # corner
            [1.0, 1.0, 1.0],  # corner
            [0.5, 0.5, 0.0],  # center face
            [0.5, 0.0, 0.5],  # center face
            [0.0, 0.5, 0.5],  # center face
            [0.5, 0.5, 1.0],  # center face
            [0.5, 1.0, 0.5],  # center face
            [1.0, 0.5, 0.5]   # center face
        ])
        
        # Add small random noise to avoid degeneracy
        noise = np.random.normal(0, 0.01, base_points.shape)
        points = base_points + noise
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 7: Optimized icosahedral configuration (more precise)
    def get_icosahedral_optimized():
        # Generate points based on icosahedron vertices plus additional points
        # Using a more mathematically sound approach
        phi = (1 + np.sqrt(5)) / 2
        # 12 vertices of icosahedron
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        # Normalize
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Add two more points at strategic positions
        # These are based on mathematical analysis of optimal distributions
        additional = np.array([
            [0.0, 0.0, 0.0],  # Will be replaced
            [0.0, 0.0, 0.0]
        ])
        
        # Use a better distribution of points
        points = vertices[:12].copy()
        # Add two more points at positions that typically give good results
        points = np.vstack([points, [[0.0, 0.0, 0.5], [1.0, 1.0, 0.5]]])
        
        # Normalize to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 8: Simple but effective grid-based initialization
    def get_grid_initial():
        # Create a 3D grid pattern that's well-distributed
        grid_points = []
        # Create a 2x2x2 grid with some additional points
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    grid_points.append([i/2, j/2, k/2])
        
        # Take first 14 points and add some randomization
        points = np.array(grid_points[:14])
        # Add small random noise
        points += np.random.normal(0, 0.02, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    # Try multiple initialization strategies and pick the best
    initial_strategies = [
        get_known_optimal_initial,
        get_fibonacci_sphere_initial,
        get_icosahedron_initial,
        get_octa_tetra_initial,
        get_latin_hypercube_initial,
        get_targeted_initial,
        get_icosahedral_optimized,
        get_grid_initial,
    ]
    
    # Generate initial configurations
    initial_configs = []
    for strategy in initial_strategies:
        try:
            config = strategy()
            # Ensure all points are within [0,1]^3
            config = np.clip(config, 0, 1)
            initial_configs.append(config.copy())
        except Exception as e:
            continue
    
    # Define improved objective function with better numerical stability
    def improved_objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Calculate pairwise distances using more stable computation
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero and handle edge cases
        if d_max <= 1e-12 or d_min <= 1e-12:
            return float('inf')
            
        # Compute ratio
        ratio = d_min / d_max
        
        # Add penalty for very small distances to encourage better spread
        penalty = 0.0
        if d_min < 1e-10:
            return float('inf')
        elif d_min < 0.01:
            penalty = 1e15 * (1.0 / (d_min + 1e-12))
        
        # Return negative ratio (since we want to maximize) 
        # with penalty for very small distances
        return -ratio + penalty
    
    # Enhanced optimization with better restart strategy and bounds
    def optimize_with_strategy(starting_points, max_restarts=5):
        best_ratio = -float('inf')
        best_points = starting_points.copy()
        
        # Multiple restarts with different perturbation levels
        for restart in range(max_restarts):
            # Create perturbed version with adaptive perturbation
            perturbed = starting_points.copy()
            
            # Vary perturbation intensity based on restart number
            if restart == 0:
                # No perturbation for first run (baseline)
                pass
            elif restart == 1:
                # Small perturbations
                perturbed += np.random.normal(0, 0.002, perturbed.shape)
            elif restart == 2:
                # Medium perturbations
                perturbed += np.random.normal(0, 0.005, perturbed.shape)
            else:
                # Larger perturbations for later restarts
                perturbed += np.random.normal(0, 0.01, perturbed.shape)
            
            # Ensure within bounds
            perturbed = np.clip(perturbed, 0, 1)
            
            try:
                # Flatten points for optimization
                x0 = perturbed.flatten()
                
                # Set up bounds for all coordinates [0,1]
                bounds = [(0, 1) for _ in range(n * 3)]
                
                # Optimization options with better settings for speed
                options = {'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
                
                # Try different optimization methods with different settings
                methods_and_options = [
                    ('L-BFGS-B', {'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}),
                    ('SLSQP', {'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}),
                    ('TNC', {'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
                ]
                
                for method, method_options in methods_and_options:
                    try:
                        result = minimize(
                            improved_objective,
                            x0,
                            method=method,
                            bounds=bounds,
                            options=method_options,
                            tol=1e-10
                        )
                        
                        # Extract results and compute actual ratio
                        final_points = result.x.reshape(-1, 3)
                        final_distances = pdist(final_points)
                        d_min = np.min(final_distances)
                        d_max = np.max(final_distances)
                        
                        if d_max > 1e-12 and d_min > 1e-12:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points.copy()
                                
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        return best_ratio, best_points
    
    # Run optimizations in parallel for better performance
    # Limit number of jobs to ensure we stay within time budget
    results = Parallel(n_jobs=min(6, len(initial_configs)))(
        delayed(optimize_with_strategy)(init_config, max_restarts=3) 
        for init_config in initial_configs
    )
    
    # Find best result among all strategies
    best_ratio = -float('inf')
    best_points = None
    
    for ratio, points_result in results:
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points_result.copy()
    
    # Additional fine-tuning with local search using a more sophisticated approach
    if best_points is not None:
        # Try one more round of optimization with the best found solution
        try:
            # Apply a more aggressive optimization approach
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(n * 3)]
            options = {'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-12}
            
            # Use multiple methods for better chance of finding good solution
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            for method in methods:
                try:
                    result = minimize(
                        improved_objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-12
                    )
                    
                    final_points = result.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-12 and d_min > 1e-12:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_points = final_points.copy()
                            best_ratio = ratio
                except Exception:
                    continue
                    
        except Exception:
            pass
    
    # Final cleanup to ensure all points are within bounds
    if best_points is not None:
        best_points = np.clip(best_points, 0, 1)
    
    # Return the best configuration found
    return best_points if best_points is not None else np.random.rand(14, 3)


# EVOLVE-BLOCK-END
