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
from scipy.spatial import ConvexHull

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
    
    def generate_convex_hull_initialization(n):
        """Generate points using convex hull approach for better distribution"""
        # Start with a good base distribution
        points = generate_icosahedron_initialization(n)
        
        # Add some randomness and adjust to improve distribution
        for i in range(10):
            # Perturb points slightly
            noise = np.random.normal(0, 0.03, points.shape)
            points += noise
            
            # Project to sphere surface
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * 0.95
            
            # Remove any points that might be too close together
            distances = pdist(points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                if min_dist < 0.1:  # If points are too close, adjust
                    points += np.random.normal(0, 0.01, points.shape)
                    norms = np.linalg.norm(points, axis=1)
                    points = points / norms[:, np.newaxis] * 0.95
        
        return points
    
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
        
        # Strategy 7: Convex hull approach
        points7 = generate_convex_hull_initialization(14)
        ratio7 = compute_min_max_ratio_fast(points7)
        
        # Select the best initialization
        ratios = [ratio1, ratio2, ratio3, ratio4, ratio5, ratio6, ratio7]
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
        elif best_idx == 5:
            return points6
        else:
            return points7
    
    # Generate best initialization
    initial_points = generate_best_initialization()
    
    # Optimization with multiple strategies for speed and quality
    best_optimized_points = initial_points.copy()
    best_optimization_ratio = compute_min_max_ratio_fast(initial_points)
    
    # Track optimization progress
    start_time = time.time()
    max_time = 55  # Leave some buffer time
    
    # Strategy 1: Direct optimization with enhanced parameters
    if time.time() - start_time < max_time:
        try:
            # Use trust-constr with aggressive settings
            result = minimize(
                objective_function,
                initial_points.flatten(),
                method='trust-constr',
                bounds=[(-0.99, 0.99) for _ in range(42)],
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-18, 'verbose': 0},
                tol=1e-18
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
            pass
    
    # Strategy 2: Enhanced Differential Evolution with better diversity
    if time.time() - start_time < max_time:
        try:
            # Use a more sophisticated global optimization approach
            bounds = [(-0.99, 0.99) for _ in range(42)]
            
            # Try multiple DE configurations with improved parameters
            configs = [
                {'maxiter': 500, 'popsize': 60, 'mutation': (0.9, 1), 'recombination': 0.95, 'seed': 42},
                {'maxiter': 300, 'popsize': 50, 'mutation': (0.8, 1), 'recombination': 0.9, 'seed': 42},
            ]
            
            for config in configs:
                if time.time() - start_time > max_time:
                    break
                    
                de_result = differential_evolution(
                    lambda x: objective_function(x),
                    bounds,
                    **config,
                    atol=1e-18,
                    rtol=1e-18,
                    seed=42
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
    
    # Strategy 3: Simulated Annealing with better cooling schedule
    if time.time() - start_time < max_time:
        try:
            bounds = [(-0.99, 0.99) for _ in range(42)]
            
            # Try multiple SA configurations with improved parameters
            sa_configs = [
                {'maxiter': 1000, 'initial_temp': 50000, 'no_local_search': False, 'seed': 42},
                {'maxiter': 800, 'initial_temp': 20000, 'no_local_search': True, 'seed': 42}
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
    
    # Strategy 4: Multiple restarts with different methods
    if time.time() - start_time < max_time:
        try:
            # Try multiple restarts with different optimization methods
            methods = ['L-BFGS-B', 'TNC', 'SLSQP']
            restarts_per_method = 10
            
            for method in methods:
                if time.time() - start_time > max_time:
                    break
                    
                for restart in range(restarts_per_method):
                    if time.time() - start_time > max_time:
                        break
                        
                    # Use the best solution found so far as starting point
                    current_start = best_optimized_points.copy()
                    
                    # Add noise to escape local minima
                    noise = np.random.normal(0, 0.02, current_start.shape)
                    current_start += noise
                    
                    # Keep within sphere
                    norms = np.linalg.norm(current_start, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        current_start[mask] = current_start[mask] / norms[mask, np.newaxis] * 0.99
                    
                    initial_guess = current_start.flatten()
                    
                    result = minimize(
                        objective_function,
                        initial_guess,
                        method=method,
                        bounds=[(-0.99, 0.99) for _ in range(42)],
                        constraints={'type': 'ineq', 'fun': constraint_func},
                        options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18},
                        tol=1e-18
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
            pass
    
    # Strategy 5: Final refinement with hybrid approach
    if time.time() - start_time < max_time:
        try:
            # Try a more aggressive final optimization
            final_points = best_optimized_points.copy()
            
            # Perform multiple passes with different optimization techniques
            for iteration in range(3):
                if time.time() - start_time > max_time:
                    break
                    
                # First, try trust-constr
                result = minimize(
                    objective_function,
                    final_points.flatten(),
                    method='trust-constr',
                    bounds=[(-0.99, 0.99) for _ in range(42)],
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 1500, 'ftol': 1e-18, 'gtol': 1e-18, 'verbose': 0},
                    tol=1e-18
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    norms = np.linalg.norm(final_points, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        final_points[mask] = final_points[mask] / norms[mask, np.newaxis] * 0.99
                    
                    ratio = compute_min_max_ratio_fast(final_points)
                    if ratio > best_optimization_ratio:
                        best_optimization_ratio = ratio
                        best_optimized_points = final_points
                
                # Then try L-BFGS-B
                result = minimize(
                    objective_function,
                    final_points.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.99, 0.99) for _ in range(42)],
                    options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18},
                    tol=1e-18
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    norms = np.linalg.norm(final_points, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        final_points[mask] = final_points[mask] / norms[mask, np.newaxis] * 0.99
                    
                    ratio = compute_min_max_ratio_fast(final_points)
                    if ratio > best_optimization_ratio:
                        best_optimization_ratio = ratio
                        best_optimized_points = final_points
                        
        except Exception:
            pass
    
    # Strategy 6: Global search with different starting points
    if time.time() - start_time < max_time:
        try:
            # Generate completely different starting points
            for i in range(5):
                if time.time() - start_time > max_time:
                    break
                    
                # Use a completely different approach
                np.random.seed(1000 + i)
                points = np.random.uniform(-0.9, 0.9, (14, 3))
                norms = np.linalg.norm(points, axis=1)
                points = points / norms[:, np.newaxis] * 0.95
                
                # Optimize this one
                result = minimize(
                    objective_function,
                    points.flatten(),
                    method='trust-constr',
                    bounds=[(-0.99, 0.99) for _ in range(42)],
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18},
                    tol=1e-18
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
            pass
    
    # Strategy 7: Specialized approach for 14 points - using known good configurations
    if time.time() - start_time < max_time:
        try:
            # Try to improve using known patterns from similar problems
            # Generate a configuration inspired by 14-point spherical codes
            # Based on research into optimal point distributions
            
            # Start with a more systematic approach using icosahedral symmetry
            base_points = generate_icosahedron_initialization(14)
            
            # Adjust positions to maximize minimum distance
            # Use a simple gradient-based approach with constraints
            result = minimize(
                objective_function,
                base_points.flatten(),
                method='trust-constr',
                bounds=[(-0.99, 0.99) for _ in range(42)],
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18, 'verbose': 0},
                tol=1e-18
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
            pass
    
    return best_optimized_points


# EVOLVE-BLOCK-END
