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
import math
from scipy.spatial import distance_matrix
from itertools import combinations
import random
from scipy.spatial.transform import Rotation as R

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
        """Generate multiple high-quality initial configurations"""
        configs = []
        
        # Configuration 1: Optimized Fibonacci spiral on sphere (using better constants)
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
        
        # Configuration 2: Icosahedral-inspired arrangement with better spacing
        # Using vertices of icosahedron and adding more points strategically
        icosahedron_vertices = np.array([
            [0, 1, 1.618], [0, -1, 1.618], [0, 1, -1.618], [0, -1, -1.618],
            [1.618, 0, 1], [-1.618, 0, 1], [1.618, 0, -1], [-1.618, 0, -1],
            [1, 1.618, 0], [-1, 1.618, 0], [1, -1.618, 0], [-1, -1.618, 0]
        ])
        norms = np.linalg.norm(icosahedron_vertices, axis=1, keepdims=True)
        icosahedron_points = icosahedron_vertices / norms * 0.9
        # Take first 12 points and add 2 more strategic points
        points = np.vstack([icosahedron_points[:12], 
                           [[0, 0, 0.9], [0, 0, -0.9]]]) + np.random.normal(0, 0.02, (14, 3))
        configs.append(points.copy())
        
        # Configuration 3: Random with better spatial distribution
        np.random.seed(42)
        points = np.random.uniform(-0.9, 0.9, (14, 3))
        configs.append(points.copy())
        
        # Configuration 4: Improved spherical code based on known good configurations
        points = np.zeros((14, 3))
        for i in range(14):
            z = 1 - 2 * i / 13  # z coordinate from 1 to -1
            radius = np.sqrt(max(0, 1 - z*z))
            theta = np.arccos(z)
            # Better golden angle spacing - use a more refined constant
            golden_angle = 2.399963229728653  # More precise value
            phi = i * golden_angle  
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = z
        configs.append(points.copy())
        
        # Configuration 5: Octahedral arrangement with perturbation (better spacing)
        # Start with vertices of octahedron and add points
        points = np.array([
            [0, 0, 1],      # top
            [0, 0, -1],     # bottom
            [1, 0, 0],      # right
            [-1, 0, 0],     # left
            [0, 1, 0],      # front
            [0, -1, 0],     # back
        ])
        
        # Add 8 more points in a pattern - use better distribution
        angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 angles
        for i, angle in enumerate(angles):
            points = np.vstack([points, [np.cos(angle)*0.7, np.sin(angle)*0.7, 0]])
        
        # Trim to 14 points and add some noise
        points = points[:14] + np.random.normal(0, 0.05, (14, 3))
        configs.append(points.copy())
        
        # Configuration 6: Cluster-based initialization with better distribution
        # Create a more uniform distribution using k-means clustering
        np.random.seed(123)
        random_points = np.random.uniform(-0.9, 0.9, (100, 3))
        kmeans = KMeans(n_clusters=14, random_state=42, n_init=10, init='k-means++')
        kmeans.fit(random_points)
        cluster_centers = kmeans.cluster_centers_
        configs.append(cluster_centers.copy())
        
        # Configuration 7: Cube-based arrangement with clustering (improved)
        points = np.random.uniform(-0.8, 0.8, (14, 3))
        # Perturb slightly to avoid degeneracy
        points += np.random.normal(0, 0.02, (14, 3))
        configs.append(points.copy())
        
        # Configuration 8: Better Fibonacci approach with improved spacing
        points = np.zeros((14, 3))
        for i in range(14):
            y = 1 - (i / 13) * 2  # y from 1 to -1
            radius = np.sqrt(max(0, 1 - y*y))
            phi = np.arccos(y)
            # Use more precise golden angle
            golden_ratio = (1 + np.sqrt(5)) / 2
            angle = i * 2 * np.pi / golden_ratio
            points[i, 0] = radius * np.sin(phi) * np.cos(angle)
            points[i, 1] = radius * np.sin(phi) * np.sin(angle)
            points[i, 2] = y
        configs.append(points.copy())
        
        # Configuration 9: Regularized spherical arrangement with better spread
        points = np.zeros((14, 3))
        # Use a different approach to get better spread
        for i in range(14):
            y = 1 - (i / 13) * 2  # y from 1 to -1
            radius = np.sqrt(max(0, 1 - y*y))
            phi = np.arccos(y)
            # Use a modified angle to distribute points more evenly
            theta = i * 2.4  # Adjusted angle for better distribution
            points[i, 0] = radius * np.sin(phi) * np.cos(theta)
            points[i, 1] = radius * np.sin(phi) * np.sin(theta)
            points[i, 2] = y
        configs.append(points.copy())
        
        # Configuration 10: Structured arrangement with better geometry
        points = np.zeros((14, 3))
        # Place 6 points on a cube face, 4 points on the opposite face, and 4 more
        # Top face
        points[:4] = [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
        # Bottom face
        points[4:8] = [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]
        # Remaining 6 points - distributed more evenly
        points[8:14] = np.random.uniform(-0.5, 0.5, (6, 3))
        configs.append(points.copy())
        
        # Configuration 11: Improved icosahedral construction with better spacing
        # Based on vertices of icosahedron with better spacing
        vertices = np.array([
            [0, 1, 1.618], [0, -1, 1.618], [0, 1, -1.618], [0, -1, -1.618],
            [1.618, 0, 1], [-1.618, 0, 1], [1.618, 0, -1], [-1.618, 0, -1],
            [1, 1.618, 0], [-1, 1.618, 0], [1, -1.618, 0], [-1, -1.618, 0],
            [0, 0, 2.618], [0, 0, -2.618]
        ])
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms * 0.8
        points = vertices[:14] + np.random.normal(0, 0.03, (14, 3))
        configs.append(points.copy())
        
        # Configuration 12: Alternative spherical code approach
        # Generate points using a modified spherical Fibonacci method
        points = np.zeros((14, 3))
        for i in range(14):
            z = 1 - 2 * i / 13
            radius = np.sqrt(max(0, 1 - z*z))
            theta = np.arccos(z)
            phi = i * 2.399963229728653  # Golden angle
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = z
        configs.append(points.copy())
        
        # Configuration 13: Randomized grid approach
        # Create a more structured randomization
        np.random.seed(43)
        points = np.zeros((14, 3))
        for i in range(14):
            points[i] = [random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9)]
        configs.append(points.copy())
        
        # Configuration 14: Hybrid approach combining multiple strategies
        # Mix of Fibonacci and clustered points
        points = np.zeros((14, 3))
        # First 8 points from Fibonacci-like distribution
        for i in range(8):
            y = 1 - (i / 7) * 2
            radius = np.sqrt(max(0, 1 - y*y))
            theta = np.arccos(y)
            phi = i * 2.399963229728653
            points[i, 0] = radius * np.sin(theta) * np.cos(phi)
            points[i, 1] = radius * np.sin(theta) * np.sin(phi)
            points[i, 2] = y
        # Last 6 points from random distribution
        points[8:] = np.random.uniform(-0.8, 0.8, (6, 3))
        configs.append(points.copy())
        
        # Configuration 15: Rotated Fibonacci approach for diversity
        points = np.zeros((14, 3))
        rotation = R.from_euler('xyz', [random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi)], degrees=False)
        for i in range(14):
            y = 1 - (i / 13) * 2  # y from 1 to -1
            radius = np.sqrt(max(0, 1 - y*y))
            phi = np.arccos(y)
            golden_ratio = (1 + np.sqrt(5)) / 2
            angle = i * 2 * np.pi / golden_ratio
            point = np.array([radius * np.sin(phi) * np.cos(angle), 
                             radius * np.sin(phi) * np.sin(angle), 
                             y])
            rotated_point = rotation.apply(point)
            points[i] = rotated_point
        configs.append(points.copy())
        
        # Configuration 16: Spherical Voronoi inspired arrangement
        # Use points that form a nearly regular structure
        points = np.zeros((14, 3))
        # Place points in layers
        layers = 4
        points_per_layer = 14 // layers + 1
        layer_idx = 0
        for layer in range(layers):
            # Determine layer height
            z = 1 - 2 * layer / (layers - 1) if layers > 1 else 0
            radius = np.sqrt(max(0, 1 - z*z))
            # Distribute points around the circle
            points_in_layer = min(points_per_layer, 14 - layer_idx)
            for i in range(points_in_layer):
                angle = 2 * np.pi * i / points_in_layer
                points[layer_idx + i] = [radius * np.cos(angle), radius * np.sin(angle), z]
            layer_idx += points_in_layer
            if layer_idx >= 14:
                break
        # Add some random noise to improve distribution
        points += np.random.normal(0, 0.03, (14, 3))
        configs.append(points.copy())
        
        # Configuration 17: Known good 14-point spherical code (from literature)
        # This is a well-known configuration that performs well
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.951056516295154, 0.0, 0.309016994374947],
            [-0.951056516295154, 0.0, 0.309016994374947],
            [0.0, 0.951056516295154, 0.309016994374947],
            [0.0, -0.951056516295154, 0.309016994374947],
            [0.951056516295154, 0.0, -0.309016994374947],
            [-0.951056516295154, 0.0, -0.309016994374947],
            [0.0, 0.951056516295154, -0.309016994374947],
            [0.0, -0.951056516295154, -0.309016994374947],
            [0.587785252292473, 0.809016994374947, 0.0],
            [-0.587785252292473, 0.809016994374947, 0.0],
            [0.587785252292473, -0.809016994374947, 0.0],
            [-0.587785252292473, -0.809016994374947, 0.0]
        ])
        configs.append(points.copy())
        
        # Configuration 18: Modified icosahedral with better distribution
        # Create points based on icosahedron vertices but with more even distribution
        icosahedron_points = np.array([
            [0, 1, 1.618], [0, -1, 1.618], [0, 1, -1.618], [0, -1, -1.618],
            [1.618, 0, 1], [-1.618, 0, 1], [1.618, 0, -1], [-1.618, 0, -1],
            [1, 1.618, 0], [-1, 1.618, 0], [1, -1.618, 0], [-1, -1.618, 0]
        ])
        norms = np.linalg.norm(icosahedron_points, axis=1, keepdims=True)
        icosahedron_points = icosahedron_points / norms * 0.8
        # Create a better arrangement by taking 12 vertices and adjusting their positions
        points = icosahedron_points.copy()
        # Add 2 more points at poles
        points = np.vstack([points, [[0, 0, 0.9], [0, 0, -0.9]]])
        # Trim to 14 points if needed
        points = points[:14] + np.random.normal(0, 0.03, (14, 3))
        configs.append(points.copy())
        
        return configs
    
    # Track timing for performance limits
    start_time = time.time()
    
    # Try multiple initial configurations with global optimization
    best_ratio = -np.inf
    best_points = None
    
    initial_configs = generate_initial_configurations()
    
    # Use a more targeted approach with fewer but better initial configurations
    # Focus on the most promising ones first
    selected_configs = initial_configs[:10]  # Use only the first 10 for efficiency
    
    # Use multiple optimization approaches with better parameter tuning
    for i, config in enumerate(selected_configs):
        if time.time() - start_time > 55:  # Leave some time for final refinements
            break
            
        # Flatten for optimization
        x0 = config.flatten()
        
        # Define bounds for optimization (keep points in reasonable range)
        bounds = [(-1.5, 1.5)] * len(x0)
        
        # Try global optimization with differential evolution first
        try:
            # Global optimization with tuned parameters for better convergence
            result_de = differential_evolution(
                objective_fast,
                bounds,
                seed=42+i,
                maxiter=200,  # Reduced iterations for speed
                popsize=30,   # Smaller population for faster convergence
                mutation=(0.5, 1),
                recombination=0.7,
                atol=1e-12,   # Less stringent tolerance for faster execution
                rtol=1e-12,
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
        
        for i, config in enumerate(selected_configs):
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
                    options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12},
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
        # Find the best initial configuration from all tries
        best_initial_config = None
        best_initial_ratio = -np.inf
        
        for i, config in enumerate(selected_configs):
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
            best_points = best_initial_config.copy()
    
    # Final refinement with multiple optimization attempts
    if best_points is not None and time.time() - start_time < 55:
        # Try different optimization methods on the best found solution
        x0 = best_points.flatten()
        bounds = [(-1.5, 1.5)] * len(x0)
        
        # Try multiple optimization approaches with different settings
        optimization_attempts = [
            ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}),
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
        normalized_points = best_points / np.max(norms) * 0.9
        return normalized_points
    
    # Fallback to a simple configuration if everything fails
    points = np.random.uniform(-0.9, 0.9, (14, 3))
    return points


# EVOLVE-BLOCK-END
