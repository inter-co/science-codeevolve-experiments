# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import warnings
from scipy.optimize import dual_annealing
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import numba

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@numba.jit(nopython=True)
def compute_distances_numba(points):
    """Fast computation of pairwise distances using numba"""
    n = points.shape[0]
    distances = np.zeros((n * (n - 1) // 2,))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            distances[idx] = np.sqrt(dx * dx + dy * dy)
            idx += 1
    return distances

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def objective(points_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = points_flat.reshape(-1, 2)
        
        # Use fast distance computation
        try:
            distances = compute_distances_numba(points)
        except:
            # Fallback to scipy version
            distances = pdist(points)
        
        if len(distances) == 0 or np.isnan(distances).any() or np.isinf(distances).any():
            return 1e10  # Large positive value for invalid configuration
            
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero or near-zero values
        if d_max <= 1e-15 or d_min <= 1e-15:
            return 1e10  # Large positive value for invalid configuration
            
        # Return negative ratio (since we're minimizing)
        # Use a small epsilon to prevent numerical issues
        epsilon = 1e-12
        return -(d_min + epsilon) / (d_max + epsilon)
    
    # Improved initialization strategies
    def generate_hexagonal_grid():
        """Generate hexagonal grid configuration"""
        points = []
        rows = 4
        cols = 4
        
        # Create a staggered grid pattern (hexagonal-like)
        for i in range(rows):
            for j in range(cols):
                # Staggered rows
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + offset + 0.5) / cols
                y = (i + 0.5) / rows
                
                # Add small random perturbation
                x += np.random.normal(0, 0.01, 1)[0]
                y += np.random.normal(0, 0.01, 1)[0]
                
                # Clamp to [0,1] range
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
        
        return np.array(points)
    
    def generate_golden_spiral():
        """Generate golden spiral configuration"""
        points = []
        n = 16
        golden_angle = 2.399963229728653  # ~2π(1 - 1/φ) where φ is golden ratio
        
        for i in range(n):
            r = np.sqrt(i / (n - 1)) * 0.4  # Radial scaling
            theta = i * golden_angle
            
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            
            # Add noise to avoid degeneracy
            x += np.random.normal(0, 0.005, 1)[0]
            y += np.random.normal(0, 0.005, 1)[0]
            
            # Clamp to [0,1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            points.append([x, y])
            
        return np.array(points)
    
    def generate_circular_arrangement():
        """Generate circular arrangement with perturbations"""
        points = []
        n = 16
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        for angle in angles:
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            
            # Add noise
            x += np.random.normal(0, 0.01, 1)[0]
            y += np.random.normal(0, 0.01, 1)[0]
            
            # Clamp to [0,1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            points.append([x, y])
            
        return np.array(points)
    
    def generate_perturbed_grid():
        """Generate grid with perturbations"""
        points = []
        # Create a 4x4 grid with slight perturbations
        for i in range(4):
            for j in range(4):
                x = j / 3.0
                y = i / 3.0
                
                # Add small random perturbation
                x += np.random.normal(0, 0.03, 1)[0]
                y += np.random.normal(0, 0.03, 1)[0]
                
                # Clamp to [0,1]
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
                
        return np.array(points)
    
    def generate_clustered_config():
        """Generate configuration using k-means clustering to avoid clustering"""
        # Start with random points
        points = np.random.rand(16, 2)
        
        # Apply k-means to create more uniform distribution
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)
        
        # Adjust positions to spread out clusters
        adjusted_points = []
        for i in range(4):
            cluster_mask = labels == i
            cluster_points = points[cluster_mask]
            if len(cluster_points) > 0:
                # Move cluster center to the middle of the cluster
                center = np.mean(cluster_points, axis=0)
                # Spread points around the center
                for pt in cluster_points:
                    direction = pt - center
                    # Scale to move away from center
                    scaled_direction = direction * 0.3
                    new_point = center + scaled_direction
                    # Clamp to [0,1]
                    new_point[0] = np.clip(new_point[0], 0, 1)
                    new_point[1] = np.clip(new_point[1], 0, 1)
                    adjusted_points.append(new_point)
        
        # Fill remaining points if needed
        while len(adjusted_points) < 16:
            adjusted_points.append(np.random.rand(2))
            
        return np.array(adjusted_points[:16])
    
    def generate_fibonacci_sphere():
        """Generate points using Fibonacci sphere method for better distribution"""
        points = []
        n = 16
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = np.sqrt(n * np.pi) * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Map to 2D with some noise
            x = 0.5 + 0.4 * x
            y = 0.5 + 0.4 * y
            
            # Add noise to avoid degeneracy
            x += np.random.normal(0, 0.005, 1)[0]
            y += np.random.normal(0, 0.005, 1)[0]
            
            # Clamp to [0,1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            points.append([x, y])
            
        return np.array(points)
    
    def generate_regular_polygon():
        """Generate points on regular polygon vertices"""
        points = []
        n = 16
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # Place points on a circle with some randomness
        for angle in angles:
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            
            # Add small random noise
            x += np.random.normal(0, 0.003, 1)[0]
            y += np.random.normal(0, 0.003, 1)[0]
            
            # Clamp to [0,1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            points.append([x, y])
            
        return np.array(points)
    
    def generate_voronoi_like():
        """Generate points inspired by Voronoi diagram principles"""
        # Start with a regular grid and perturb
        points = []
        for i in range(4):
            for j in range(4):
                x = j / 3.0 + np.random.uniform(-0.05, 0.05)
                y = i / 3.0 + np.random.uniform(-0.05, 0.05)
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
        return np.array(points)
    
    def generate_better_initialization():
        """Generate a better initial configuration using a combination of methods"""
        # Start with a regular grid and add small perturbations
        points = []
        for i in range(4):
            for j in range(4):
                x = j / 3.0 + np.random.uniform(-0.02, 0.02)
                y = i / 3.0 + np.random.uniform(-0.02, 0.02)
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
        return np.array(points)
    
    def generate_concentric_circles():
        """Generate points in concentric circles for better spacing"""
        points = []
        n = 16
        
        # Distribute points in concentric rings
        radii = [0.2, 0.3, 0.4, 0.45]  # Different ring radii
        points_per_ring = [4, 4, 4, 4]  # Points per ring
        
        for i, (radius, count) in enumerate(zip(radii, points_per_ring)):
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for angle in angles:
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                # Add noise to avoid degeneracy
                x += np.random.normal(0, 0.005, 1)[0]
                y += np.random.normal(0, 0.005, 1)[0]
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
                
        return np.array(points[:16])
    
    def generate_best_initialization():
        """Generate the best initial configuration based on known good patterns"""
        # Try multiple good starting configurations and pick the best one
        configs = [
            generate_hexagonal_grid(),
            generate_golden_spiral(),
            generate_circular_arrangement(),
            generate_perturbed_grid(),
            generate_regular_polygon(),
            generate_concentric_circles()
        ]
        
        best_config = None
        best_ratio = -1e10
        
        for config in configs:
            try:
                distances = pdist(config)
                if len(distances) > 0 and not np.isnan(distances).any() and not np.isinf(distances).any():
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    if d_max > 1e-15 and d_min > 1e-15:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_config = config.copy()
            except:
                continue
                
        if best_config is not None:
            return best_config
        else:
            # Fallback to random
            return np.random.rand(16, 2)
    
    # Generate a single high-quality initial configuration
    initial_points = generate_best_initialization()
    initial_flat = initial_points.flatten()
    
    best_ratio = -1e10
    best_points = None
    
    # Strategy 1: Differential Evolution with optimized parameters
    try:
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]
        
        # Use differential evolution with aggressive parameters
        result = differential_evolution(
            objective,
            bounds,
            maxiter=1000,
            popsize=100,
            seed=42,
            disp=False,
            atol=1e-18,
            rtol=1e-18,
            strategy='best1bin'
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            
            # Validate result
            distances = pdist(final_points)
            if len(distances) > 0 and not np.isnan(distances).any() and not np.isinf(distances).any():
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 1e-15 and d_min > 1e-15:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
    except Exception as e:
        pass
    
    # Strategy 2: Dual Annealing with more iterations and better parameters
    if best_points is None:
        try:
            bounds = [(0, 1) for _ in range(32)]
            
            result = dual_annealing(
                objective,
                bounds,
                maxiter=3000,
                initial_temp=2000,
                seed=42,
                no_local_search=True
            )
            
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            
            # Validate and compare
            distances = pdist(final_points)
            if len(distances) > 0 and not np.isnan(distances).any() and not np.isinf(distances).any():
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 1e-15 and d_min > 1e-15:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_points = final_points
                        
        except Exception as e:
            pass
    
    # Strategy 3: Local optimization with multiple restarts
    if best_points is not None:
        try:
            initial_flat = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            # Use multiple local optimization strategies with aggressive settings
            strategies = [
                ('L-BFGS-B', {'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18}),
                ('TNC', {'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}),
                ('SLSQP', {'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16})
            ]
            
            for method, options in strategies:
                try:
                    result = minimize(
                        objective,
                        initial_flat,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-20
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        
                        # Validate and compare
                        distances = pdist(final_points)
                        if len(distances) > 0 and not np.isnan(distances).any() and not np.isinf(distances).any():
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            if d_max > 1e-15 and d_min > 1e-15:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = final_points
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
    
    # Strategy 4: Multi-start approach with different initializations
    if best_points is None:
        # Try a few more diverse initialization strategies
        test_configs = []
        
        # Generate a variety of configurations
        for i in range(5):
            # Random initialization
            rand_config = np.random.rand(16, 2)
            test_configs.append(rand_config)
            
            # Hexagonal grid
            hex_config = generate_hexagonal_grid()
            test_configs.append(hex_config)
            
            # Golden spiral
            golden_config = generate_golden_spiral()
            test_configs.append(golden_config)
        
        # Evaluate all configurations
        for config in test_configs:
            try:
                distances = pdist(config)
                if len(distances) > 0 and not np.isnan(distances).any() and not np.isinf(distances).any():
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    if d_max > 1e-15 and d_min > 1e-15:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = config.copy()
            except:
                continue
    
    # Final fallback: return the best initial configuration
    if best_points is None:
        return initial_points
    
    return best_points


# EVOLVE-BLOCK-END
