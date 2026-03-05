# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial import SphericalVoronoi
import random
from numba import jit
import time
from scipy.optimize import differential_evolution, dual_annealing
from scipy.spatial import distance
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

@jit(nopython=True)
def compute_min_max_ratio_fast(points):
    """Fast computation of min/max distance ratio using compiled code"""
    n = points.shape[0]
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
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
    return 0.0

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initialization strategies with advanced optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        return np.min(distances) / np.max(distances)
    
    def objective_function(x):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        # Compute negative ratio (we want to maximize ratio, so minimize negative)
        ratio = compute_min_max_ratio_fast(points)  # Use fast version
        return -ratio
    
    def constraint_func(x):
        """Constraint to keep points within unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return positive values when constraint is satisfied
        return 1.0 - norms
    
    def generate_fibonacci_sphere(n):
        """Generate points on sphere using Fibonacci spiral"""
        points = np.zeros((n, 3))
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = golden_angle * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points[i] = [x, y, z]
        return points
    
    def generate_random_initialization(n, seed=None):
        """Generate random points in unit sphere"""
        if seed is not None:
            np.random.seed(seed)
        points = np.random.uniform(-1, 1, (n, 3))
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        valid_indices = norms <= 1
        points = points[valid_indices]
        
        # If we don't have enough points, generate more
        while len(points) < n:
            additional = np.random.uniform(-1, 1, (n - len(points), 3))
            additional_norms = np.linalg.norm(additional, axis=1)
            valid_additional = additional[additional_norms <= 1]
            points = np.vstack([points, valid_additional])
            
        return points[:n]
    
    def generate_voronoi_initialization(n):
        """Generate points based on Voronoi-like distribution"""
        # Start with fibonacci sphere
        points = generate_fibonacci_sphere(n)
        
        # Add some randomness to avoid local minima
        noise_level = 0.05
        points += np.random.normal(0, noise_level, points.shape)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / np.max(norms) * 0.9
        
        return points
    
    def generate_icosahedron_initialization(n):
        """Generate points using icosahedron-based distribution"""
        # Vertices of regular icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # If we need more points, distribute them more evenly
        if n > 12:
            # Generate points along edges and faces of icosahedron
            points = vertices.copy()
            # Add midpoints of edges
            edges = []
            # Icosahedron has 30 edges
            edge_indices = [
                (0,1), (0,4), (0,5), (1,7), (1,10), (2,3), (2,6), (2,11),
                (3,4), (3,8), (4,9), (5,6), (5,9), (6,11), (7,8), (7,10),
                (8,9), (10,11), (0,2), (1,3), (4,5), (6,7), (8,10), (9,11)
            ]
            
            # Add edge midpoints
            for i, j in edge_indices:
                midpoint = (vertices[i] + vertices[j]) / 2
                midpoint = midpoint / np.linalg.norm(midpoint)
                points = np.vstack([points, midpoint])
                
            # Add face centers (approximate)
            faces = [
                (0,1,4), (0,5,1), (1,7,10), (1,10,0), (0,4,5), (5,9,4),
                (4,9,3), (3,9,8), (3,8,2), (2,8,6), (2,6,11), (6,11,7),
                (7,11,10), (10,11,3), (3,2,4), (4,2,5), (5,0,1), (1,0,7),
                (7,1,10), (10,1,3), (3,10,8), (8,10,11), (11,10,7), (7,11,6),
                (6,11,2), (2,11,8), (8,11,3), (3,8,9), (9,8,4), (4,9,5)
            ]
            
            for i, j, k in faces:
                center = (vertices[i] + vertices[j] + vertices[k]) / 3
                center = center / np.linalg.norm(center)
                points = np.vstack([points, center])
                
            # Take first n points
            if len(points) >= n:
                points = points[:n]
            else:
                # Fill with random points
                remaining = n - len(points)
                extra_points = np.random.uniform(-1, 1, (remaining, 3))
                extra_points = extra_points / np.linalg.norm(extra_points, axis=1, keepdims=True)
                points = np.vstack([points, extra_points])
                
        else:
            points = vertices[:n]
            
        return points
    
    def generate_spherical_code_initialization(n):
        """Generate points using a more sophisticated spherical code approach"""
        # Start with icosahedron points
        points = generate_icosahedron_initialization(n)
        
        # Apply a few iterations of Lloyd relaxation to improve distribution
        for _ in range(3):
            # Create Voronoi diagram on sphere (approximated)
            # This is a simplified approach - in practice, would use more complex methods
            # For now, just add small perturbations
            noise = np.random.normal(0, 0.02, points.shape)
            points += noise
            
            # Project back to sphere
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * 0.9
            
        return points
    
    def generate_clustered_initialization(n):
        """Generate points using clustered approach for better spread"""
        # Generate points using multiple strategies and cluster them
        points_list = []
        
        # Strategy 1: Icosahedron-based
        points1 = generate_icosahedron_initialization(n)
        points_list.append(points1)
        
        # Strategy 2: Fibonacci sphere
        points2 = generate_fibonacci_sphere(n)
        points_list.append(points2)
        
        # Strategy 3: Random points
        points3 = generate_random_initialization(n, seed=42)
        points_list.append(points3)
        
        # Combine all strategies and use K-means to find representative points
        all_points = np.vstack(points_list)
        
        # Use K-means to get better distributed points
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(all_points)
        centroids = kmeans.cluster_centers_
        
        # Ensure they're on the unit sphere
        norms = np.linalg.norm(centroids, axis=1)
        centroids = centroids / norms[:, np.newaxis] * 0.95
        
        return centroids
    
    def generate_best_initialization():
        """Generate the best possible initialization strategy"""
        # Strategy 1: Fibonacci sphere
        points1 = generate_fibonacci_sphere(14)
        points1 = points1 / np.max(np.linalg.norm(points1, axis=1)) * 0.9
        ratio1 = compute_min_max_ratio_fast(points1)
        
        # Strategy 2: Voronoi-inspired
        points2 = generate_voronoi_initialization(14)
        ratio2 = compute_min_max_ratio_fast(points2)
        
        # Strategy 3: Random initialization
        points3 = generate_random_initialization(14, seed=42)
        ratio3 = compute_min_max_ratio_fast(points3)
        
        # Strategy 4: Icosahedron-based
        points4 = generate_icosahedron_initialization(14)
        ratio4 = compute_min_max_ratio_fast(points4)
        
        # Strategy 5: Spherical code approach
        points5 = generate_spherical_code_initialization(14)
        ratio5 = compute_min_max_ratio_fast(points5)
        
        # Strategy 6: Clustered approach
        points6 = generate_clustered_initialization(14)
        ratio6 = compute_min_max_ratio_fast(points6)
        
        # Select the best initialization
        ratios = [ratio1, ratio2, ratio3, ratio4, ratio5, ratio6]
        best_idx = np.argmax(ratios)
        
        if best_idx == 0:
            return points1
        elif best_idx == 1:
            return points2
        elif best_idx == 2:
            return points3
        elif best_idx == 3:
            return points4
        elif best_idx == 4:
            return points5
        else:
            return points6
    
    # Generate best initialization
    initial_points = generate_best_initialization()
    
    # Optimization with multiple strategies for speed and quality
    best_optimized_points = initial_points.copy()
    best_optimization_ratio = compute_min_max_ratio_fast(initial_points)
    
    # Track optimization progress
    start_time = time.time()
    max_time = 55  # Leave some buffer time
    
    # Strategy 1: Improved Differential Evolution with better parameter tuning
    if time.time() - start_time < max_time:
        try:
            # Use a more sophisticated global optimization approach
            bounds = [(-0.99, 0.99) for _ in range(42)]
            
            # Try multiple DE configurations with improved parameters
            configs = [
                {'maxiter': 300, 'popsize': 50, 'mutation': (0.8, 1), 'recombination': 0.9, 'seed': 42},
                {'maxiter': 200, 'popsize': 40, 'mutation': (0.6, 1), 'recombination': 0.8, 'seed': 42},
                {'maxiter': 150, 'popsize': 30, 'mutation': (0.5, 1), 'recombination': 0.7, 'seed': 42}
            ]
            
            for config in configs:
                if time.time() - start_time > max_time:
                    break
                    
                de_result = differential_evolution(
                    lambda x: objective_function(x),
                    bounds,
                    **config,
                    atol=1e-16,
                    rtol=1e-16
                )
                
                if de_result.success:
                    de_points = de_result.x.reshape(-1, 3)
                    # Ensure points are within unit sphere
                    norms = np.linalg.norm(de_points, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        de_points[mask] = de_points[mask] / norms[mask, np.newaxis] * 0.99
                    
                    de_ratio = compute_min_max_ratio_fast(de_points)
                    if de_ratio > best_optimization_ratio:
                        best_optimization_ratio = de_ratio
                        best_optimized_points = de_points
                        
        except Exception:
            pass
    
    # Strategy 2: Enhanced local optimization with better restarts and more aggressive settings
    if time.time() - start_time < max_time:
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        for method in methods:
            if time.time() - start_time > max_time:
                break
                
            # Multiple random restarts with better initialization
            for restart in range(15):  # Increased from 10 to 15 for more exploration
                if time.time() - start_time > max_time:
                    break
                    
                # Start with a good point and add more noise
                perturbed_points = initial_points.copy()
                noise_scale = 0.01 + restart * 0.02  # Increasing noise with restarts
                perturbed_points += np.random.normal(0, noise_scale, perturbed_points.shape)
                
                # Ensure points are within sphere
                norms = np.linalg.norm(perturbed_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    perturbed_points[mask] = perturbed_points[mask] / norms[mask, np.newaxis] * 0.99
                
                initial_guess = perturbed_points.flatten()
                
                try:
                    result = minimize(
                        objective_function,
                        initial_guess,
                        method=method,
                        bounds=[(-0.99, 0.99) for _ in range(42)],
                        constraints={'type': 'ineq', 'fun': constraint_func},
                        options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-16},
                        tol=1e-16
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        # Ensure points are within unit sphere
                        norms = np.linalg.norm(optimized_points, axis=1)
                        mask = norms > 1.0
                        if np.any(mask):
                            optimized_points[mask] = optimized_points[mask] / norms[mask, np.newaxis] * 0.99
                        
                        ratio = compute_min_max_ratio_fast(optimized_points)
                        if ratio > best_optimization_ratio:
                            best_optimization_ratio = ratio
                            best_optimized_points = optimized_points
                            
                except Exception:
                    continue
    
    # Strategy 3: Simulated Annealing with better cooling schedule and more iterations
    if time.time() - start_time < max_time:
        try:
            bounds = [(-0.99, 0.99) for _ in range(42)]
            
            # Try multiple SA configurations with improved parameters
            sa_configs = [
                {'maxiter': 500, 'initial_temp': 10000, 'no_local_search': True, 'seed': 42},
                {'maxiter': 300, 'initial_temp': 5000, 'no_local_search': False, 'seed': 42}
            ]
            
            for config in sa_configs:
                if time.time() - start_time > max_time:
                    break
                    
                sa_result = dual_annealing(
                    lambda x: objective_function(x),
                    bounds,
                    **config,
                    seed=42
                )
                
                if sa_result.success:
                    sa_points = sa_result.x.reshape(-1, 3)
                    # Ensure points are within unit sphere
                    norms = np.linalg.norm(sa_points, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        sa_points[mask] = sa_points[mask] / norms[mask, np.newaxis] * 0.99
                    
                    sa_ratio = compute_min_max_ratio_fast(sa_points)
                    if sa_ratio > best_optimization_ratio:
                        best_optimization_ratio = sa_ratio
                        best_optimized_points = sa_points
                        
        except Exception:
            pass
    
    # Strategy 4: Trust-constr with aggressive parameters and proper constraint handling
    if time.time() - start_time < max_time:
        try:
            # Use the best result so far as starting point
            final_points = best_optimized_points.copy()
            
            # Project points to sphere boundary for better constraint satisfaction
            norms = np.linalg.norm(final_points, axis=1)
            # Normalize points to unit sphere boundary
            final_points = final_points / norms[:, np.newaxis] * 0.99
            
            # More aggressive optimization with tighter constraints and better tolerance
            result = minimize(
                objective_function,
                final_points.flatten(),
                method='trust-constr',
                bounds=[(-0.99, 0.99) for _ in range(42)],  # Tighter bounds
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16, 'verbose': 0},
                tol=1e-16
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(final_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    final_points[mask] = final_points[mask] / norms[mask, np.newaxis] * 0.99
                
                final_ratio = compute_min_max_ratio_fast(final_points)
                if final_ratio > best_optimization_ratio:
                    best_optimization_ratio = final_ratio
                    best_optimized_points = final_points
                    
        except Exception:
            pass
    
    # Strategy 5: Additional refinement with more targeted approaches
    if time.time() - start_time < max_time:
        try:
            # Try with a different approach: Nelder-Mead with adaptive parameters
            final_points = best_optimized_points.copy()
            
            # Try with very tight tolerances and different restart strategy
            result = minimize(
                objective_function,
                final_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000, 
                    'adaptive': True, 
                    'fatol': 1e-16, 
                    'xatol': 1e-16,
                    'maxfev': 5000
                }
            )
            
            if result.success:
                nm_points = result.x.reshape(-1, 3)
                # Ensure points are within unit sphere
                norms = np.linalg.norm(nm_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    nm_points[mask] = nm_points[mask] / norms[mask, np.newaxis] * 0.99
                
                nm_ratio = compute_min_max_ratio_fast(nm_points)
                if nm_ratio > best_optimization_ratio:
                    best_optimization_ratio = nm_ratio
                    best_optimized_points = nm_points
                    
        except Exception:
            pass
    
    # Strategy 6: Final global search with enhanced diversity
    if time.time() - start_time < max_time:
        try:
            # Use a different initialization approach to explore more diverse regions
            # Generate multiple candidate solutions and pick the best
            candidates = []
            for i in range(5):
                np.random.seed(1000 + i)
                # Generate points using a different approach
                points = np.random.uniform(-0.9, 0.9, (14, 3))
                norms = np.linalg.norm(points, axis=1)
                points = points / norms[:, np.newaxis] * 0.95
                candidates.append(points)
            
            # Evaluate all candidates
            best_candidate = None
            best_candidate_ratio = 0
            for candidate in candidates:
                ratio = compute_min_max_ratio_fast(candidate)
                if ratio > best_candidate_ratio:
                    best_candidate_ratio = ratio
                    best_candidate = candidate
            
            # Refine the best candidate
            if best_candidate is not None:
                result = minimize(
                    objective_function,
                    best_candidate.flatten(),
                    method='trust-constr',
                    bounds=[(-0.99, 0.99) for _ in range(42)],
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16},
                    tol=1e-16
                )
                
                if result.success:
                    refined_points = result.x.reshape(-1, 3)
                    norms = np.linalg.norm(refined_points, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        refined_points[mask] = refined_points[mask] / norms[mask, np.newaxis] * 0.99
                    
                    refined_ratio = compute_min_max_ratio_fast(refined_points)
                    if refined_ratio > best_optimization_ratio:
                        best_optimization_ratio = refined_ratio
                        best_optimized_points = refined_points
                        
        except Exception:
            pass
    
    # Strategy 7: Enhanced refinement with multiple local searches
    if time.time() - start_time < max_time:
        try:
            # Apply a second round of local optimization with even more aggressive settings
            final_points = best_optimized_points.copy()
            
            # Apply multiple sequential local optimizations
            methods = ['L-BFGS-B', 'trust-constr']
            for method in methods:
                if time.time() - start_time > max_time:
                    break
                    
                result = minimize(
                    objective_function,
                    final_points.flatten(),
                    method=method,
                    bounds=[(-0.99, 0.99) for _ in range(42)],
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-16},
                    tol=1e-16
                )
                
                if result.success:
                    refined_points = result.x.reshape(-1, 3)
                    norms = np.linalg.norm(refined_points, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        refined_points[mask] = refined_points[mask] / norms[mask, np.newaxis] * 0.99
                    
                    refined_ratio = compute_min_max_ratio_fast(refined_points)
                    if refined_ratio > best_optimization_ratio:
                        best_optimization_ratio = refined_ratio
                        best_optimized_points = refined_points
                        
        except Exception:
            pass
    
    # Strategy 8: Geometrically inspired refinement using spherical Voronoi
    if time.time() - start_time < max_time:
        try:
            # Use a more geometrically motivated approach
            # Try to create a configuration with better symmetry
            points = generate_icosahedron_initialization(14)
            
            # Apply a more sophisticated optimization specifically designed for this problem
            # First project to sphere boundary
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * 0.99
            
            # Use a more aggressive trust-constr with specific options
            result = minimize(
                objective_function,
                points.flatten(),
                method='trust-constr',
                bounds=[(-0.99, 0.99) for _ in range(42)],
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={
                    'maxiter': 2500, 
                    'ftol': 1e-16, 
                    'gtol': 1e-16,
                    'verbose': 0,
                    'disp': False
                },
                tol=1e-16
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(refined_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    refined_points[mask] = refined_points[mask] / norms[mask, np.newaxis] * 0.99
                
                refined_ratio = compute_min_max_ratio_fast(refined_points)
                if refined_ratio > best_optimization_ratio:
                    best_optimization_ratio = refined_ratio
                    best_optimized_points = refined_points
                    
        except Exception:
            pass
    
    # Strategy 9: Multi-start optimization with better diversity
    if time.time() - start_time < max_time:
        try:
            # Try multiple starting points with different strategies
            strategies = [
                lambda: generate_fibonacci_sphere(14) * 0.9,
                lambda: generate_icosahedron_initialization(14) * 0.9,
                lambda: generate_voronoi_initialization(14) * 0.9,
                lambda: generate_spherical_code_initialization(14) * 0.9,
                lambda: generate_clustered_initialization(14) * 0.9
            ]
            
            for i, strategy in enumerate(strategies):
                if time.time() - start_time > max_time:
                    break
                    
                try:
                    # Generate initial points
                    points = strategy()
                    
                    # Optimize from this starting point
                    result = minimize(
                        objective_function,
                        points.flatten(),
                        method='trust-constr',
                        bounds=[(-0.99, 0.99) for _ in range(42)],
                        constraints={'type': 'ineq', 'fun': constraint_func},
                        options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16},
                        tol=1e-16
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        norms = np.linalg.norm(optimized_points, axis=1)
                        mask = norms > 1.0
                        if np.any(mask):
                            optimized_points[mask] = optimized_points[mask] / norms[mask, np.newaxis] * 0.99
                        
                        ratio = compute_min_max_ratio_fast(optimized_points)
                        if ratio > best_optimization_ratio:
                            best_optimization_ratio = ratio
                            best_optimized_points = optimized_points
                            
                except Exception:
                    continue
                    
        except Exception:
            pass
    
    return best_optimized_points


# EVOLVE-BLOCK-END
