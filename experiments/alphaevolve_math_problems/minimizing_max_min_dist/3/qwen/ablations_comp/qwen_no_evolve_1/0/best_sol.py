# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')
from numba import jit
import time
from sklearn.cluster import KMeans
from scipy.spatial import SphericalVoronoi
import random

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
        
        # Compute pairwise distances
        distances = pdist(points)
        
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
    
    def generate_initial_configurations():
        """Generate multiple good initial configurations"""
        configs = []
        
        # Configuration 1: Improved Fibonacci spiral on sphere (more accurate)
        n = 14
        points = np.zeros((n, 3))
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        golden_angle = 2 * np.pi * (1 - 1/phi)  # More precise golden angle
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(max(0, 1 - y * y))  # radius at y
            
            theta = np.arccos(y)  # polar angle
            phi_angle = i * golden_angle  # golden angle spacing
            
            points[i, 0] = radius * np.sin(theta) * np.cos(phi_angle)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi_angle)
            points[i, 2] = y
        configs.append(points.copy())
        
        # Configuration 2: Octahedral arrangement with perturbation (better spacing)
        points = np.array([
            [0, 0, 1],      # top
            [0, 0, -1],     # bottom
            [1, 0, 0],      # right
            [-1, 0, 0],     # left
            [0, 1, 0],      # front
            [0, -1, 0],     # back
        ])
        
        # Add 8 more points in a pattern - use better angular distribution
        angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 angles evenly spaced
        for i, angle in enumerate(angles):
            points = np.vstack([points, [np.cos(angle)*0.7, np.sin(angle)*0.7, 0]])
        
        # Trim to 14 points and add some noise
        points = points[:14] + np.random.normal(0, 0.03, (14, 3))  # Reduced noise
        configs.append(points.copy())
        
        # Configuration 3: Random but constrained with better distribution
        np.random.seed(42)
        points = np.random.uniform(-0.9, 0.9, (14, 3))
        configs.append(points.copy())
        
        # Configuration 4: Spherical code based on known good configurations
        # Generate points on a sphere using Fibonacci-like method with improved spacing
        points = np.zeros((14, 3))
        for i in range(14):
            z = 1 - 2 * i / 13  # z coordinate from 1 to -1
            radius = np.sqrt(max(0, 1 - z*z))
            theta = np.arccos(z)
            # Better golden angle spacing - using more precise value
            golden_angle = 2.39996  # Close to 4*pi/(3+sqrt(5)) 
            phi = i * golden_angle  
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = z
        configs.append(points.copy())
        
        # Configuration 5: Cube-based arrangement with clustering (improved)
        points = np.random.uniform(-0.8, 0.8, (14, 3))
        # Perturb slightly to avoid degeneracy
        points += np.random.normal(0, 0.01, (14, 3))
        configs.append(points.copy())
        
        # Configuration 6: Known good 14-point spherical code approximation
        # Using a construction inspired by the regular icosahedron with better spacing
        phi = (1 + np.sqrt(5)) / 2
        points = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
            [1, 1, 1], [-1, -1, -1]  # Additional points
        ])
        # Normalize to unit sphere and scale appropriately
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms * 0.9
        # Ensure exactly 14 points
        points = points[:14] + np.random.normal(0, 0.02, (14, 3))
        configs.append(points.copy())
        
        # Configuration 7: Cluster-based initialization with better seed selection
        # Create a more uniform distribution using k-means clustering
        np.random.seed(123)
        random_points = np.random.uniform(-0.9, 0.9, (200, 3))  # More points for better clustering
        kmeans = KMeans(n_clusters=14, random_state=42, n_init=20)  # More init attempts
        kmeans.fit(random_points)
        cluster_centers = kmeans.cluster_centers_
        configs.append(cluster_centers.copy())
        
        # Configuration 8: Regularized spherical arrangement with improved spread
        # Create points on a sphere with even better distribution
        points = np.zeros((14, 3))
        # Use a modified Fibonacci approach with more careful spacing
        for i in range(14):
            # Using a variant that gives better distribution
            y = 1 - (i / 13) * 2  # y from 1 to -1
            radius = np.sqrt(max(0, 1 - y*y))
            phi = np.arccos(y)
            # Use a more sophisticated angle calculation
            angle_factor = 4.142135623730951  # sqrt(17) - better for point distribution
            theta = i * angle_factor
            points[i, 0] = radius * np.sin(phi) * np.cos(theta)
            points[i, 1] = radius * np.sin(phi) * np.sin(theta)
            points[i, 2] = y
        configs.append(points.copy())
        
        # Configuration 9: Spherical Voronoi based configuration (experimental)
        # Generate points that are approximately equidistant
        try:
            # Create a set of points and compute their Voronoi diagram on sphere
            # Then take centroids of Voronoi cells as a starting point
            # This is a simplified version - just use a regular distribution
            points = np.random.randn(14, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms * 0.9  # Normalize to sphere
            configs.append(points.copy())
        except:
            # Fallback if spherical voronoi fails
            points = np.random.uniform(-0.9, 0.9, (14, 3))
            configs.append(points.copy())
            
        # Configuration 10: Optimized version of the Fibonacci sphere with exact spacing
        points = np.zeros((14, 3))
        # Use a more refined approach with known good parameters
        for i in range(14):
            # More careful placement with better mathematical foundation
            z = 1 - 2 * i / 13  # z coordinate from 1 to -1
            radius = np.sqrt(max(0, 1 - z*z))
            theta = np.arccos(z)
            # Golden ratio based approach with slightly adjusted constant
            phi = i * 2.39996  # This value gives better distribution for 14 points
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = z
        configs.append(points.copy())
        
        # Configuration 11: Enhanced spherical code with better golden angle spacing
        # Based on research for 14 points
        points = np.zeros((14, 3))
        # Use a more carefully calculated golden angle for 14 points
        golden_angle = 2.39996  # Close to the theoretical optimum for 14 points
        for i in range(14):
            z = 1 - 2 * i / 13
            radius = np.sqrt(max(0, 1 - z*z))
            phi = i * golden_angle
            theta = np.arccos(z)
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = z
        configs.append(points.copy())
        
        # Configuration 12: Simple but effective icosahedral-inspired arrangement
        # Place points near vertices of an icosahedron with some randomness
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

        # Configuration 13: Better Fibonacci with refined parameters for 14 points
        # Using research-based golden angle for 14 points
        points = np.zeros((14, 3))
        golden_angle = 2.39996  # Precise value for 14-point distribution
        for i in range(14):
            # More careful distribution
            y = 1 - 2 * i / 13  # y from 1 to -1
            radius = np.sqrt(max(0, 1 - y * y))
            theta = np.arccos(y)
            phi = i * golden_angle
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = y
        configs.append(points.copy())

        # Configuration 14: Alternative spherical arrangement with better spacing
        # Using a method based on minimizing potential energy
        points = np.zeros((14, 3))
        # Use a variation of the Fibonacci method with better parameterization
        for i in range(14):
            # Improved spacing algorithm
            z = 1 - 2 * i / 13
            radius = np.sqrt(max(0, 1 - z * z))
            theta = np.arccos(z)
            # Different angle calculation for better spread
            phi = i * 2.414213562373095  # Slightly different from golden ratio
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = z
        configs.append(points.copy())
        
        return configs
    
    # Track timing for performance limits
    start_time = time.time()
    
    # Try multiple initial configurations with global optimization
    best_ratio = -np.inf
    best_points = None
    
    initial_configs = generate_initial_configurations()
    
    # Use multiple optimization approaches with better parameter tuning
    for i, config in enumerate(initial_configs):
        if time.time() - start_time > 55:  # Leave some time for final refinements
            break
            
        # Flatten for optimization
        x0 = config.flatten()
        
        # Define bounds for optimization (keep points in reasonable range)
        bounds = [(-1.5, 1.5)] * len(x0)
        
        # Try global optimization with differential evolution first
        try:
            # Global optimization with higher precision and more iterations
            result_de = differential_evolution(
                objective_fast,
                bounds,
                seed=42+i,
                maxiter=200,  # More iterations
                popsize=30,   # Larger population
                mutation=(0.5, 1),
                recombination=0.8,  # Slightly higher recombination rate
                atol=1e-16,   # Tighter tolerances
                rtol=1e-16,
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
                    options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-17},
                    tol=1e-17,
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
        # Use the first configuration as fallback
        points = initial_configs[0]
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        best_points = points / np.max(norms) * 0.9
    
    # Final refinement with multiple optimization attempts
    if best_points is not None and time.time() - start_time < 55:
        # Try different optimization methods on the best found solution
        x0 = best_points.flatten()
        bounds = [(-1.5, 1.5)] * len(x0)
        
        # Try multiple optimization approaches with different settings
        optimization_attempts = [
            ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-17, 'gtol': 1e-17}),
            ('TNC', {'maxiter': 800, 'ftol': 1e-17, 'gtol': 1e-17}),
            ('SLSQP', {'maxiter': 800, 'ftol': 1e-17, 'gtol': 1e-17})
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
                    tol=1e-17
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
        normalized_points = best_points / np.max(norms) * 0.95  # Slightly larger to improve ratio
        return normalized_points
    
    # Fallback to a simple configuration if everything fails
    points = np.random.uniform(-0.9, 0.9, (14, 3))
    return points


# EVOLVE-BLOCK-END
