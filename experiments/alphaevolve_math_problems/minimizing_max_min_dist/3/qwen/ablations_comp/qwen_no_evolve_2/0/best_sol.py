# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')
from numba import jit
import time
from sklearn.cluster import KMeans
import math

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
    
    # Specialized initialization for better results
    def init_symmetric_pattern():
        """Initialize with a symmetric pattern designed to maximize spread"""
        # This creates a configuration that's known to perform well
        # Based on mathematical analysis of optimal 14-point distributions
        points = []
        
        # Add vertices of a cube (8 points)
        cube_points = np.array([
            [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1],
            [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]
        ])
        points.extend(cube_points)
        
        # Add 6 points at positions that help balance distances
        # These are positioned to create a more uniform distribution
        extra_points = [
            [0, 0, 0.8], [0, 0, -0.8],  # Along z-axis
            [0.8, 0, 0], [-0.8, 0, 0],  # Along x-axis
            [0, 0.8, 0], [0, -0.8, 0]   # Along y-axis
        ]
        points.extend(extra_points)
        
        # Convert to numpy array
        points_array = np.array(points[:n])
        
        # Ensure they're on the unit sphere
        norms = np.linalg.norm(points_array, axis=1, keepdims=True)
        points_array = points_array / norms
        
        return points_array
    
    def init_optimized_grid():
        """Initialize with an optimized grid-like pattern"""
        # Create a pattern that balances uniformity and spread
        points = []
        
        # Add 8 corner points of a cube
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    points.append([x, y, z])
        
        # Add 6 face center points
        face_centers = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]
        points.extend(face_centers)
        
        # Adjust some points to improve distribution
        # Make sure we have exactly n points
        if len(points) > n:
            points = points[:n]
        elif len(points) < n:
            # Add more points in strategic locations
            remaining = n - len(points)
            for _ in range(remaining):
                # Add points in a way that maintains good spread
                point = np.random.randn(3)
                point = point / np.linalg.norm(point)
                points.append(point)
        
        return np.array(points)
    
    # New specialized initialization based on known good configurations
    def init_improved_symmetric():
        """Initialize with an improved symmetric pattern based on mathematical analysis"""
        # This is a carefully constructed pattern inspired by optimal 14-point configurations
        points = []
        
        # Add 8 cube corners
        cube_corners = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ])
        points.extend(cube_corners)
        
        # Add 6 face centers
        face_centers = np.array([
            [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]
        ])
        points.extend(face_centers)
        
        # Add 2 more points that are often found in optimal solutions
        # These help balance the minimum distance
        additional_points = [
            [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]
        ]
        points.extend(additional_points)
        
        # Normalize all points to unit sphere
        points_array = np.array(points[:n])
        norms = np.linalg.norm(points_array, axis=1, keepdims=True)
        points_array = points_array / norms
        
        return points_array
    
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
        init_kmeans_refined,
        init_symmetric_pattern,
        init_optimized_grid,
        init_improved_symmetric
    ]
    
    best_ratio = 0
    best_points = None
    
    # Optimized objective function with better numerical stability
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
        epsilon = 1e-12
        if max_dist <= epsilon:
            return 0
            
        # Return negative ratio since we want to maximize it
        # Added small regularization to avoid numerical issues
        return -(min_dist / (max_dist + epsilon))
    
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
    
    # Strategy 1: Focus on best performing initializations with aggressive optimization
    high_quality_initials = [
        init_improved_symmetric,
        init_symmetric_pattern,
        init_optimized_grid
    ]
    
    for init_func in high_quality_initials:
        if (time.time() - start_time) > 55:
            break
            
        try:
            points = init_func()
            
            # Use a more aggressive optimization approach
            # First, global optimization with better parameters
            try:
                # Use a more efficient global optimizer with fewer iterations
                de_result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=100,  # Reduced iterations to save time
                    popsize=10,    # Smaller population for faster convergence
                    seed=42,
                    disp=False,
                    strategy='best1bin',
                    atol=1e-12,
                    rtol=1e-12
                )
                
                if de_result.success:
                    global_points = de_result.x.reshape(-1, 3)
                    
                    # Then local refinement with multiple methods
                    methods_and_configs = [
                        ('L-BFGS-B', {'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-14}),
                        ('TNC', {'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-14}),
                        ('SLSQP', {'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-14})
                    ]
                    
                    for method, options in methods_and_configs:
                        if (time.time() - start_time) > 55:
                            break
                            
                        try:
                            result = minimize(
                                objective_function,
                                global_points.flatten(),
                                method=method,
                                bounds=bounds,
                                constraints=cons,
                                options=options,
                                tol=1e-14
                            )
                            
                            if result.success:
                                optimized_points = result.x.reshape(-1, 3)
                                final_distances = pdist(optimized_points)
                                if len(final_distances) > 0:
                                    final_min = np.min(final_distances)
                                    final_max = np.max(final_distances)
                                    if final_max > 1e-12:
                                        final_ratio = final_min / final_max
                                        if final_ratio > best_ratio:
                                            best_ratio = final_ratio
                                            best_points = optimized_points.copy()
                                            # Early exit if we're approaching target
                                            if final_ratio > 0.485:
                                                return best_points
                        except Exception:
                            continue
            except Exception:
                continue
                
        except Exception:
            continue
    
    # Strategy 2: Multiple local optimizations with different initializations
    attempts = 0
    max_attempts = 8  # Reduced attempts to save time
    
    while attempts < max_attempts and (time.time() - start_time) < 55:  # Leave some buffer
        if (time.time() - start_time) > 55:
            break
            
        try:
            # Select a random initialization strategy
            init_func = np.random.choice(initial_strategies)
            points = init_func()
            
            # Try optimization with different settings
            methods_and_configs = [
                ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}),
                ('TNC', {'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}),
                ('SLSQP', {'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
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
                        tol=1e-14
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        
                        # Calculate final ratio
                        final_distances = pdist(optimized_points)
                        if len(final_distances) > 0:
                            final_min = np.min(final_distances)
                            final_max = np.max(final_distances)
                            if final_max > 1e-12:
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
    
    # Strategy 3: Try a more focused optimization approach with enhanced constraints
    if best_points is None and (time.time() - start_time) < 55:
        try:
            # Try with a very good initialization from our best known pattern
            points = init_improved_symmetric()
            
            # Use a more aggressive single optimization with better parameters
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
                        options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14},
                        tol=1e-14
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        final_distances = pdist(optimized_points)
                        if len(final_distances) > 0:
                            final_min = np.min(final_distances)
                            final_max = np.max(final_distances)
                            if final_max > 1e-12:
                                final_ratio = final_min / final_max
                                if final_ratio > best_ratio:
                                    best_ratio = final_ratio
                                    best_points = optimized_points.copy()
                                    if final_ratio > 0.48:
                                        return best_points
                except Exception:
                    continue
                    
        except Exception:
            pass
    
    # If still no success, return the best we have or a default pattern
    if best_points is not None:
        return best_points
    else:
        # Return the improved symmetric pattern as fallback which tends to work well
        return init_improved_symmetric()


# EVOLVE-BLOCK-END
