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
    
    # Enhanced initialization strategies with better geometric insight
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
    
    def init_known_optimal():
        """Use a known good configuration that's close to optimal"""
        # This is a carefully chosen configuration inspired by known solutions
        # for point distributions in 3D
        points = np.array([
            [0.000000, 0.000000, 1.000000],
            [0.000000, 0.000000, -1.000000],
            [0.000000, 0.707107, 0.707107],
            [0.000000, 0.707107, -0.707107],
            [0.000000, -0.707107, 0.707107],
            [0.000000, -0.707107, -0.707107],
            [0.707107, 0.000000, 0.707107],
            [0.707107, 0.000000, -0.707107],
            [-0.707107, 0.000000, 0.707107],
            [-0.707107, 0.000000, -0.707107],
            [0.707107, 0.707107, 0.000000],
            [0.707107, -0.707107, 0.000000],
            [-0.707107, 0.707107, 0.000000],
            [-0.707107, -0.707107, 0.000000]
        ])
        return points
    
    def init_voronoi_based():
        """Initialize using Voronoi-based distribution pattern"""
        # Start with icosahedron and refine
        points = init_icosahedron()
        
        # Add some strategic points for better distribution
        # Add vertices of a cube to create more balanced distribution
        cube = np.array([
            [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
            [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
        ])
        
        # Combine and normalize
        combined = np.vstack([points, cube[:n-len(points)]])
        combined = combined / np.linalg.norm(combined[0]) if np.linalg.norm(combined[0]) > 0 else combined
        
        # Randomize to avoid perfect patterns that might cause convergence issues
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, combined.shape)
        combined += noise
        
        # Normalize again to keep on sphere
        for i in range(len(combined)):
            combined[i] = combined[i] / np.linalg.norm(combined[i])
            
        return combined
    
    def init_kmeans_refined():
        """Initialize using k-means clustering for better distribution"""
        # Start with random points
        points = np.random.randn(n, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Use k-means to cluster points and then move them to cluster centers
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(points)
        refined_points = kmeans.cluster_centers_
        
        # Normalize the refined points
        refined_points = refined_points / np.linalg.norm(refined_points, axis=1, keepdims=True)
        return refined_points
    
    def init_hybrid_approach():
        """Hybrid initialization combining multiple strategies"""
        # Start with a good base pattern
        base_points = init_icosahedron()
        
        # Add some cube corner points for better coverage
        cube_points = np.array([
            [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
            [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
        ])
        
        # Combine and add some random points
        combined = np.vstack([base_points, cube_points[:n-len(base_points)]])
        
        # If we still don't have enough, add random points
        if len(combined) < n:
            extra = np.random.randn(n - len(combined), 3)
            extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
            combined = np.vstack([combined, extra])
            
        return combined[:n]
    
    # Try multiple initialization strategies, prioritizing the most promising ones
    initial_strategies = [
        init_hybrid_approach,
        init_voronoi_based,
        init_known_optimal,
        init_kmeans_refined,
        init_expanded_pattern,
        init_icosahedron,
        init_spherical_cluster,
        init_fibonacci_sphere,
        init_cube_corners,
        init_golden_spiral,
        init_p24,
        init_regular_tetrahedron
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
    
    # Strategy 1: Try different initializations with local optimization
    attempts = 0
    max_attempts = 15  # Reduced to allow more time for better optimization
    
    while attempts < max_attempts and (time.time() - start_time) < 55:  # Leave some buffer
        try:
            # Select a random initialization strategy
            init_func = np.random.choice(initial_strategies)
            points = init_func()
            
            # Use a more robust local optimization approach
            methods_and_configs = [
                ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}),
                ('TNC', {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}),
                ('SLSQP', {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
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
                                    if final_ratio > 0.48:
                                        return best_points
                                        
                except Exception:
                    continue
                    
        except Exception:
            pass
            
        attempts += 1
    
    # Strategy 2: Global optimization with better parameters
    if best_points is None and (time.time() - start_time) < 55:
        try:
            # Use a more robust differential evolution with better parameters
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,  # Reduced iterations to save time
                popsize=15,    # Smaller population for faster convergence
                seed=42,
                disp=False,
                strategy='best1bin',
                atol=1e-15,   # Tighter absolute tolerance
                rtol=1e-15    # Tighter relative tolerance
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
        except Exception:
            pass
    
    # Strategy 3: Final refinement with a known good starting point
    if best_points is None:
        try:
            # Start with our most promising configuration
            points = init_hybrid_approach()
            
            # Direct local optimization on this
            result = minimize(
                objective_function,
                points.flatten(),
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
        except Exception:
            pass
    
    # If still no success, return the best we have or default initialization
    if best_points is None:
        # Try the most promising initialization: hybrid approach
        points = init_hybrid_approach()
        return points
    
    return best_points


# EVOLVE-BLOCK-END
