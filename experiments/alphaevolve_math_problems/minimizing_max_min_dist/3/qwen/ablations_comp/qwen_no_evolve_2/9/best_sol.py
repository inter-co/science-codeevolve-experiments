# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')
from numba import jit
import time
from sklearn.cluster import KMeans

@jit(nopython=True)
def compute_distances_jit(points):
    """Compute pairwise distances efficiently using numba"""
    n = points.shape[0]
    distances = np.zeros((n * (n - 1) // 2,))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i, k] - points[j, k]
                dist += diff * diff
            distances[idx] = np.sqrt(dist)
            idx += 1
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid optimization approach combining global and local search methods with improved efficiency.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    
    # Improved initialization strategies with more geometric insight
    def init_fibonacci_sphere():
        """Initialize points using Fibonacci spiral on sphere"""
        points = np.zeros((n, d))
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = golden_angle * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points[i] = [x, y, z]
        return points / np.linalg.norm(points[0]) if np.linalg.norm(points[0]) > 0 else points
    
    def init_random_uniform():
        """Initialize points uniformly at random in unit cube"""
        np.random.seed(42)
        return np.random.uniform(-1, 1, (n, d))
    
    def init_cube_corners():
        """Initialize points at cube corners plus some interior points"""
        # Cube corners
        corners = np.array([
            [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1],
            [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]
        ])
        
        # Fill remaining points randomly
        remaining = n - len(corners)
        if remaining > 0:
            additional = np.random.uniform(-1, 1, (remaining, d))
            points = np.vstack([corners, additional])
        else:
            points = corners[:n]
            
        return points
    
    def init_icosahedron():
        """Initialize points based on icosahedron vertices"""
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0]) if np.linalg.norm(vertices[0]) > 0 else vertices
        
        # If we need more points, add random points on sphere
        if n > len(vertices):
            extra_points = np.random.randn(n - len(vertices), 3)
            extra_points = extra_points / np.linalg.norm(extra_points, axis=1, keepdims=True)
            points = np.vstack([vertices, extra_points])
        else:
            points = vertices[:n]
            
        return points
    
    def init_p24():
        """Initialize points based on 24-cell vertices pattern"""
        # Generate points in a way that maximizes spread
        # Using a combination of regular structures
        points = []
        
        # Add vertices of a regular octahedron
        octahedron = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ])
        
        points.extend(octahedron)
        
        # Add more points in between
        if len(points) < n:
            remaining = n - len(points)
            # Add random points on sphere
            for _ in range(remaining):
                point = np.random.randn(3)
                point = point / np.linalg.norm(point)
                points.append(point)
        
        return np.array(points[:n])
    
    def init_regular_tetrahedron():
        """Initialize points based on regular tetrahedron with additional points"""
        # Regular tetrahedron vertices
        tetrahedron = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ])
        
        # Normalize
        tetrahedron = tetrahedron / np.linalg.norm(tetrahedron[0])
        
        # Add more points to get 14 total
        if n > len(tetrahedron):
            extra_points = np.random.randn(n - len(tetrahedron), 3)
            extra_points = extra_points / np.linalg.norm(extra_points, axis=1, keepdims=True)
            points = np.vstack([tetrahedron, extra_points])
        else:
            points = tetrahedron[:n]
            
        return points
    
    def init_spherical_cluster():
        """Initialize points with better clustering distribution"""
        # Start with a good configuration based on known optimal patterns
        # This uses a combination of symmetry and optimization principles
        points = []
        
        # Add 8 vertices of a cube
        cube_vertices = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ])
        points.extend(cube_vertices)
        
        # Add 6 more points to form a more uniform distribution
        # These should be roughly at positions where they maximize minimum distances
        additional = np.array([
            [0, 0, 0.7], [0, 0, -0.7],
            [0.7, 0, 0], [-0.7, 0, 0],
            [0, 0.7, 0], [0, -0.7, 0]
        ])
        
        # Add remaining points randomly but ensuring good spread
        remaining = n - len(points)
        if remaining > 0:
            for _ in range(remaining):
                point = np.random.randn(3)
                point = point / np.linalg.norm(point)
                points.append(point)
        
        return np.array(points[:n])
    
    def init_expanded_pattern():
        """Initialize using a more sophisticated pattern based on known good configurations"""
        # Start with vertices of a cube and extend with symmetry-based points
        points = []
        
        # Cube corners
        cube_corners = np.array([
            [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1],
            [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]
        ])
        points.extend(cube_corners)
        
        # Add face centers
        face_centers = np.array([
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ])
        points.extend(face_centers)
        
        # Fill remaining spots with random points on sphere
        remaining = n - len(points)
        if remaining > 0:
            for _ in range(remaining):
                point = np.random.randn(3)
                point = point / np.linalg.norm(point)
                points.append(point)
                
        return np.array(points[:n])
    
    def init_golden_spiral():
        """Initialize using a golden spiral pattern on sphere"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        golden_angle = 2 * np.pi * (1 - 1/phi)
        
        # Create points along a spiral
        for i in range(n):
            # Distribute points along the z-axis
            z = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - z * z)
            
            # Golden angle spacing
            theta = golden_angle * i
            
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)
    
    def init_known_good_config():
        """Initialize with a known good configuration from literature"""
        # Based on research and known optimal configurations for 14 points
        # This configuration tries to balance symmetry and uniformity
        points = np.array([
            # Cube corners
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
            # Face centers  
            [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]
        ])
        
        # If we have more than 14 points, truncate
        if len(points) >= n:
            return points[:n]
        else:
            # Add remaining points randomly on sphere
            remaining = n - len(points)
            for _ in range(remaining):
                point = np.random.randn(3)
                point = point / np.linalg.norm(point)
                points = np.vstack([points, point])
            return points
    
    def init_better_spherical():
        """Initialize using a more sophisticated spherical distribution"""
        # Generate points that are more evenly distributed on sphere
        # Use a variant of the Fibonacci sphere approach but with better spacing
        points = []
        
        # Generate points using a more uniform method
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        golden_angle = 2 * np.pi * (1 - 1/phi)
        
        # Use a more controlled approach to avoid clustering
        for i in range(n):
            # Better distribution along z-axis
            z = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - z * z)
            
            # Use a more uniform angular spacing
            theta = golden_angle * i
            
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)
    
    def init_kmeans_refined():
        """Initialize using k-means clustering to find good starting points"""
        # Start with random points, then refine using k-means
        np.random.seed(42)
        initial_points = np.random.randn(n, 3)
        initial_points = initial_points / np.linalg.norm(initial_points, axis=1, keepdims=True)
        
        # Apply k-means clustering to find good distribution
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(initial_points)
        
        # Use cluster centers as starting points
        return kmeans.cluster_centers_
    
    # Try multiple initialization strategies
    initial_strategies = [
        init_fibonacci_sphere,
        init_random_uniform,
        init_cube_corners,
        init_icosahedron,
        init_p24,
        init_regular_tetrahedron,
        init_spherical_cluster,
        init_expanded_pattern,
        init_golden_spiral,
        init_known_good_config,
        init_better_spherical,
        init_kmeans_refined
    ]
    
    best_ratio = 0
    best_points = None
    
    # More sophisticated objective function with better numerical stability
    def objective_function(x_flat):
        """Objective function that maximizes the min/max distance ratio"""
        points = x_flat.reshape(-1, 3)
        
        # Compute distances efficiently using scipy
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero with small epsilon
        epsilon = 1e-15
        if max_dist <= epsilon:
            return 0
            
        # Return negative ratio since we want to maximize it
        return -min_dist / max_dist
    
    # Enhanced constraint handling with stricter enforcement
    def sphere_constraint(x_flat):
        """Constraint that keeps all points within unit sphere"""
        points = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return positive values when constraint satisfied (norm <= 1)
        return 1.0 - norms
    
    # Bounds for optimization (points in unit sphere)
    bounds = [(-1.0, 1.0) for _ in range(n * 3)]
    
    # Add constraint for sphere constraint
    cons = {'type': 'ineq', 'fun': sphere_constraint}
    
    # Use a more systematic optimization approach with better control
    np.random.seed(42)
    
    # Track timing to ensure we stay within budget
    start_time = time.time()
    
    # Try multiple optimization approaches with better parameter tuning
    attempts = 0
    max_attempts = 15  # Increased attempts for better chance
    
    # Strategy 1: Global optimization first
    try:
        # Use a more aggressive differential evolution approach
        de_result = differential_evolution(
            objective_function,
            bounds,
            maxiter=200,  # More iterations
            popsize=20,    # Larger population
            seed=42,
            disp=False,
            strategy='best1bin',
            atol=1e-12,   # Tighter absolute tolerance
            rtol=1e-12    # Tighter relative tolerance
        )
        
        if de_result.success:
            global_points = de_result.x.reshape(-1, 3)
            
            # Refine with local optimization
            result = minimize(
                objective_function,
                global_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15},
                tol=1e-15
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                final_distances = pdist(refined_points)
                if len(final_distances) > 0:
                    final_min = np.min(final_distances)
                    final_max = np.max(final_distances)
                    if final_max > 1e-15:
                        final_ratio = final_min / final_max
                        if final_ratio > best_ratio:
                            best_ratio = final_ratio
                            best_points = refined_points.copy()
                            # Early exit if we're close to target
                            if final_ratio > 0.48:
                                return best_points
    except Exception:
        pass
    
    # Strategy 2: Multiple local optimizations with different initializations
    while attempts < max_attempts and (time.time() - start_time) < 55:  # Leave some buffer
        try:
            # Select a random initialization strategy
            init_func = np.random.choice(initial_strategies)
            points = init_func()
            
            # Try optimization with different settings
            methods_and_configs = [
                ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}),
                ('TNC', {'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}),
                ('SLSQP', {'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            ]
            
            for method, options in methods_and_configs:
                if (time.time() - start_time) > 55:
                    break
                    
                try:
                    # Use a more robust optimizer with careful parameters
                    result = minimize(
                        objective_function,
                        points.flatten(),
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options=options,
                        tol=1e-15
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        
                        # Calculate final ratio
                        final_distances = pdist(optimized_points)
                        if len(final_distances) > 0:
                            final_min = np.min(final_distances)
                            final_max = np.max(final_distances)
                            if final_max > 1e-15:
                                final_ratio = final_min / final_max
                                
                                if final_ratio > best_ratio:
                                    best_ratio = final_ratio
                                    best_points = optimized_points.copy()
                                    # Early exit if we're approaching target
                                    if final_ratio > 0.47:
                                        return best_points
                                        
                except Exception:
                    continue
                    
        except Exception:
            pass
            
        attempts += 1
    
    # Strategy 3: Try a more focused optimization approach
    if best_points is None and (time.time() - start_time) < 55:
        try:
            # Try with a very good initialization from our best known pattern
            points = init_known_good_config()
            
            # Use multiple optimization methods in sequence
            methods = ['L-BFGS-B', 'TNC', 'SLSQP']
            for method in methods:
                if (time.time() - start_time) > 55:
                    break
                    
                try:
                    result = minimize(
                        objective_function,
                        points.flatten(),
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15},
                        tol=1e-15
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        final_distances = pdist(optimized_points)
                        if len(final_distances) > 0:
                            final_min = np.min(final_distances)
                            final_max = np.max(final_distances)
                            if final_max > 1e-15:
                                final_ratio = final_min / final_max
                                if final_ratio > best_ratio:
                                    best_ratio = final_ratio
                                    best_points = optimized_points.copy()
                                    if final_ratio > 0.47:
                                        return best_points
                except Exception:
                    continue
                    
        except Exception:
            pass
    
    # If still no success, return the best we have or a default pattern
    if best_points is not None:
        return best_points
    else:
        # Return the known good configuration as fallback
        return init_known_good_config()


# EVOLVE-BLOCK-END
