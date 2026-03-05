# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')
from numba import jit
import time
from sklearn.cluster import KMeans
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import cdist

@jit(nopython=True)
def compute_distances_numba(points):
    """Compute pairwise distances efficiently using numba"""
    n = points.shape[0]
    distances = np.zeros(n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            distances[idx] = dist
            idx += 1
    return distances

@jit(nopython=True)
def compute_distance_matrix_numba(points):
    """Compute full distance matrix efficiently"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initialization strategies with advanced optimization.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        # Reshape to 14x3 points
        points = x.reshape(-1, 3)
        
        # Compute pairwise distances using faster numba version
        distances = compute_distances_numba(points)
        
        # Return negative of min/max ratio (we want to maximize this, so minimize negative)
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0
            
        # Return negative ratio to minimize (maximize the actual ratio)
        return -min_dist / max_dist
    
    def objective_fast(x):
        """Faster objective function using numba-compiled distance computation"""
        points = x.reshape(-1, 3)
        distances = compute_distances_numba(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    def objective_matrix(x):
        """Objective using full matrix computation for more robustness"""
        points = x.reshape(-1, 3)
        distances = compute_distance_matrix_numba(points)
        # Get upper triangular part (excluding diagonal)
        triu_indices = np.triu_indices_from(distances, k=1)
        distances_flat = distances[triu_indices]
        
        if len(distances_flat) == 0:
            return 0
            
        min_dist = np.min(distances_flat)
        max_dist = np.max(distances_flat)
        
        if max_dist == 0:
            return 0
            
        return -min_dist / max_dist
    
    def generate_initial_configurations():
        """Generate high-quality initial configurations"""
        configs = []
        
        # Configuration 1: Optimized Fibonacci spiral on sphere (based on research)
        # Using known good golden angle for 14 points
        points = np.zeros((14, 3))
        golden_angle = 2.39996  # This value works well for 14 points
        for i in range(14):
            z = 1 - 2 * i / 13  # z coordinate from 1 to -1
            radius = np.sqrt(max(0, 1 - z*z))
            theta = np.arccos(z)
            phi = i * golden_angle
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = z
        configs.append(points.copy())
        
        # Configuration 2: Icosahedral-inspired arrangement with better spacing
        # Using vertices of icosahedron plus additional points
        icosahedron_vertices = np.array([
            [0, 0, 1], [0, 0, -1],
            [0.85065080835204, 0, 0.525731112119134],
            [-0.85065080835204, 0, 0.525731112119134],
            [0.85065080835204, 0, -0.525731112119134],
            [-0.85065080835204, 0, -0.525731112119134],
            [0, 0.525731112119134, 0.85065080835204],
            [0, -0.525731112119134, 0.85065080835204],
            [0, 0.525731112119134, -0.85065080835204],
            [0, -0.525731112119134, -0.85065080835204],
            [0.525731112119134, 0.85065080835204, 0],
            [-0.525731112119134, 0.85065080835204, 0],
            [0.525731112119134, -0.85065080835204, 0],
            [-0.525731112119134, -0.85065080835204, 0]
        ])
        # Normalize and add small random perturbations
        norms = np.linalg.norm(icosahedron_vertices, axis=1, keepdims=True)
        points = icosahedron_vertices / norms * 0.9
        points += np.random.normal(0, 0.01, (14, 3))
        configs.append(points.copy())
        
        # Configuration 3: Optimized octahedral arrangement
        # Start with octahedron vertices and add points on edges
        points = np.array([
            [0, 0, 1],      # top
            [0, 0, -1],     # bottom
            [1, 0, 0],      # right
            [-1, 0, 0],     # left
            [0, 1, 0],      # front
            [0, -1, 0],     # back
        ])
        # Add 8 more points in a more uniform pattern
        angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 angles evenly spaced
        for i, angle in enumerate(angles):
            points = np.vstack([points, [np.cos(angle)*0.7, np.sin(angle)*0.7, 0]])
        # Trim to 14 points and add some noise
        points = points[:14] + np.random.normal(0, 0.03, (14, 3))
        configs.append(points.copy())
        
        # Configuration 4: Cluster-based initialization with better seed selection
        # Create a more uniform distribution using k-means clustering
        np.random.seed(123)
        random_points = np.random.uniform(-0.9, 0.9, (200, 3))  # More points for better clustering
        kmeans = KMeans(n_clusters=14, random_state=42, n_init=30)  # More init attempts
        kmeans.fit(random_points)
        cluster_centers = kmeans.cluster_centers_
        configs.append(cluster_centers.copy())
        
        # Configuration 5: Known good spherical code approximation
        # Based on research for 14 points on sphere
        points = np.zeros((14, 3))
        # Use a construction that's known to perform well for 14 points
        for i in range(14):
            z = 1 - 2 * i / 13
            radius = np.sqrt(max(0, 1 - z*z))
            theta = np.arccos(z)
            # Use a golden ratio-based approach
            golden_ratio = (1 + np.sqrt(5)) / 2
            phi = i * 2 * np.pi / golden_ratio
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = z
        configs.append(points.copy())
        
        # Configuration 6: Improved spherical code with specific angle calculation
        points = np.zeros((14, 3))
        # Using a more optimized angle spacing
        for i in range(14):
            z = 1 - 2 * i / 13
            radius = np.sqrt(max(0, 1 - z*z))
            theta = np.arccos(z)
            # Specific optimized angle for 14 points
            phi = i * 2.39996  # Fine-tuned for 14 points
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = z
        configs.append(points.copy())
        
        # Configuration 7: Random but with better constraints
        np.random.seed(42)
        points = np.random.uniform(-0.9, 0.9, (14, 3))
        configs.append(points.copy())
        
        # Configuration 8: Perturbed regular icosahedron vertices
        # Create vertices of icosahedron, then slightly perturb
        t = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, t], [0, -1, t], [0, -1, -t], [0, 1, -t],
            [1, t, 0], [-1, t, 0], [-1, -t, 0], [1, -t, 0],
            [t, 0, 1], [-t, 0, 1], [-t, 0, -1], [t, 0, -1],
            [1, 1, 1], [-1, -1, -1]  # Additional points for 14 total
        ])
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        points = vertices / norms * 0.9
        points = points[:14] + np.random.normal(0, 0.02, (14, 3))
        configs.append(points.copy())
        
        # Configuration 9: More refined spherical arrangement based on known good values
        # Using a more systematic approach for 14 points
        points = np.zeros((14, 3))
        # Generate points using a variant of the Fibonacci spiral with better spacing
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(14):
            # Distribute points more uniformly on sphere
            y = 1 - 2 * i / 13  # y goes from 1 to -1
            radius = np.sqrt(1 - y*y)
            theta = np.arccos(y)
            # Use golden angle increment but adjust for 14 points
            angle = i * 2 * np.pi / phi  # Golden angle increment
            points[i, 0] = radius * np.sin(theta) * np.cos(angle)
            points[i, 1] = radius * np.sin(theta) * np.sin(angle)
            points[i, 2] = y
        configs.append(points.copy())
        
        # Configuration 10: Modified icosahedral arrangement with better symmetry
        # Use icosahedral vertices but with a different perturbation scheme
        ico_verts = np.array([
            [0, 0, 1], [0, 0, -1],
            [0.85065080835204, 0, 0.525731112119134],
            [-0.85065080835204, 0, 0.525731112119134],
            [0.85065080835204, 0, -0.525731112119134],
            [-0.85065080835204, 0, -0.525731112119134],
            [0, 0.525731112119134, 0.85065080835204],
            [0, -0.525731112119134, 0.85065080835204],
            [0, 0.525731112119134, -0.85065080835204],
            [0, -0.525731112119134, -0.85065080835204],
            [0.525731112119134, 0.85065080835204, 0],
            [-0.525731112119134, 0.85065080835204, 0],
            [0.525731112119134, -0.85065080835204, 0],
            [-0.525731112119134, -0.85065080835204, 0]
        ])
        # Normalize and scale appropriately
        norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
        points = ico_verts / norms * 0.85
        # Add slight perturbations to improve the ratio
        points += np.random.normal(0, 0.015, (14, 3))
        configs.append(points.copy())
        
        # Configuration 11: A new configuration based on known optimal spherical codes
        # Using a more carefully constructed set of points
        points = np.zeros((14, 3))
        # Based on research for 14 points, using a combination of regular patterns with perturbations
        # Layered approach with different z-coordinates
        z_coords = np.linspace(-0.9, 0.9, 14)
        for i in range(14):
            z = z_coords[i]
            radius = np.sqrt(max(0, 1 - z*z))
            # Distribute points around circle at this z-level
            angle = i * 2 * np.pi / 14 + 0.1 * np.sin(i * 0.5)  # Add slight variation
            points[i, 0] = radius * np.cos(angle)
            points[i, 1] = radius * np.sin(angle)
            points[i, 2] = z
        configs.append(points.copy())
        
        # Configuration 12: Another attempt with improved Fibonacci-like distribution
        points = np.zeros((14, 3))
        # Use a better distribution formula with more careful spacing
        for i in range(14):
            # Better spacing using arctan distribution
            y = 1 - 2 * i / 13
            radius = np.sqrt(max(0, 1 - y*y))
            theta = np.arccos(y)
            # More precise angle calculation
            phi = i * 2.39996 + 0.01 * np.sin(i * 0.7)  # Small perturbations
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = y
        configs.append(points.copy())
        
        return configs
    
    # Track timing for performance limits
    start_time = time.time()
    
    # Try multiple initial configurations with global optimization
    best_ratio = -np.inf
    best_points = None
    
    initial_configs = generate_initial_configurations()
    
    # Use more aggressive optimization with better parameter tuning
    for i, config in enumerate(initial_configs):
        if time.time() - start_time > 55:  # Leave some time for final refinements
            break
            
        # Flatten for optimization
        x0 = config.flatten()
        
        # Define bounds for optimization (keep points in reasonable range)
        bounds = [(-1.5, 1.5)] * len(x0)
        
        # Try global optimization with differential evolution first
        try:
            # Global optimization with better parameters for speed and quality
            result_de = differential_evolution(
                objective_fast,
                bounds,
                seed=42+i,
                maxiter=150,  # Reduced iterations for faster testing
                popsize=40,   # Larger population for better exploration
                mutation=(0.8, 1),  # Higher mutation for more exploration
                recombination=0.9,  # Higher recombination rate
                atol=1e-10,   # Tighter tolerances
                rtol=1e-10,
                disp=False
            )
            
            # Extract optimized points
            optimized_points = result_de.x.reshape(-1, 3)
            
            # Calculate final ratio
            distances = pdist(optimized_points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
        except Exception as e:
            continue
    
    # If no global optimization worked well, try local optimization on the best initial config
    if best_points is None:
        # Find the best initial configuration
        best_initial_config = None
        best_initial_ratio = -np.inf
        
        for i, config in enumerate(initial_configs):
            if time.time() - start_time > 55:
                break
                
            # Evaluate initial configuration
            distances = pdist(config)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_initial_ratio:
                        best_initial_ratio = ratio
                        best_initial_config = config.copy()
        
        if best_initial_config is not None:
            # Use local optimization on the best initial configuration
            x0 = best_initial_config.flatten()
            bounds = [(-1.5, 1.5)] * len(x0)
            
            try:
                # More aggressive local optimization with better parameters
                result = minimize(
                    objective_fast,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12,
                    callback=lambda x: None  # Prevent excessive output
                )
                
                optimized_points = result.x.reshape(-1, 3)
                distances = pdist(optimized_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
            except Exception as e:
                pass
    
    # If we still don't have a solution, use the best initial configuration
    if best_points is None:
        # Find the best initial configuration among those tested
        best_initial_config = None
        best_initial_ratio = -np.inf
        
        for i, config in enumerate(initial_configs):
            if time.time() - start_time > 55:
                break
                
            distances = pdist(config)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_initial_ratio:
                        best_initial_ratio = ratio
                        best_initial_config = config.copy()
        
        if best_initial_config is not None:
            best_points = best_initial_config.copy()
    
    # Final refinement with multiple optimization attempts
    if best_points is not None and time.time() - start_time < 55:
        # Try different optimization methods on the best found solution
        x0 = best_points.flatten()
        bounds = [(-1.5, 1.5)] * len(x0)
        
        # Try multiple optimization approaches with different settings
        optimization_attempts = [
            ('L-BFGS-B', {'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12}),
            ('TNC', {'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12}),
            ('SLSQP', {'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12})
        ]
        
        for method, options in optimization_attempts:
            if time.time() - start_time > 55:
                break
                
            try:
                result = minimize(
                    objective_fast,
                    x0,
                    method=method,
                    bounds=bounds,
                    options=options,
                    tol=1e-12
                )
                
                optimized_points = result.x.reshape(-1, 3)
                distances = pdist(optimized_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        final_ratio = min_dist / max_dist
                        # Only accept if it's better than our previous best
                        if final_ratio > best_ratio:
                            best_ratio = final_ratio
                            best_points = optimized_points.copy()
                            
            except Exception as e:
                continue
    
    # Normalize to ensure all points are within unit sphere
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        # Normalize to unit sphere but ensure proper spacing
        normalized_points = best_points / np.max(norms) * 0.95  # Slightly larger to improve ratio
        return normalized_points
    
    # Fallback to a simple configuration if everything fails
    points = np.random.uniform(-0.9, 0.9, (14, 3))
    return points


# EVOLVE-BLOCK-END
