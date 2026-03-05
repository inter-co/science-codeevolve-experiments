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
from scipy.spatial.distance import squareform

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@numba.jit(nopython=True)
def compute_min_max_distances_numba(points):
    """Compute min and max distances efficiently using numba"""
    n = points.shape[0]
    if n < 2:
        return 0.0, 1.0
    
    min_dist = 1e10
    max_dist = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    return min_dist, max_dist

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
            min_dist, max_dist = compute_min_max_distances_numba(points)
        except:
            # Fallback to scipy version
            distances = pdist(points)
            if len(distances) == 0:
                return 1e10
            min_dist = np.min(distances)
            max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 1e-12:
            return 1e10
            
        # Return negative ratio (since we're minimizing)
        # Use a small epsilon to prevent numerical issues
        epsilon = 1e-12
        return -(min_dist + epsilon) / (max_dist + epsilon)
    
    def objective_with_penalty(points_flat):
        """Objective function with penalty for edge cases"""
        points = points_flat.reshape(-1, 2)
        
        # Use fast distance computation
        try:
            min_dist, max_dist = compute_min_max_distances_numba(points)
        except:
            # Fallback to scipy version
            distances = pdist(points)
            if len(distances) == 0:
                return 1e10
            min_dist = np.min(distances)
            max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 1e-12:
            return 1e10
            
        # Penalize configurations where points are too close together
        if min_dist < 1e-6:
            return 1e10  # Invalid configuration
            
        # Return negative ratio (since we're minimizing)
        # Use a small epsilon to prevent numerical issues
        epsilon = 1e-12
        return -(min_dist + epsilon) / (max_dist + epsilon)
    
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
    
    def generate_symmetric_arrangement():
        """Generate symmetric arrangement based on known good solutions"""
        # Start with a square grid with some adjustments
        points = []
        # Create a 4x4 grid but adjust for better spacing
        for i in range(4):
            for j in range(4):
                x = j / 3.0
                y = i / 3.0
                
                # Add strategic perturbations to improve distribution
                if i == 0 or i == 3:
                    x += np.random.normal(0, 0.02, 1)[0] if j % 2 == 0 else np.random.normal(0, 0.01, 1)[0]
                elif i == 1 or i == 2:
                    x += np.random.normal(0, 0.01, 1)[0] if j % 2 == 0 else np.random.normal(0, 0.02, 1)[0]
                
                y += np.random.normal(0, 0.01, 1)[0]
                
                # Clamp to [0,1]
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
                
        return np.array(points)
    
    def generate_better_grid():
        """Generate a better grid arrangement with optimized spacing"""
        # Create a grid that's slightly optimized for spacing
        points = []
        # Use a more structured approach with specific spacing
        for i in range(4):
            for j in range(4):
                # Create a more evenly spaced grid with some perturbations
                x = (j + 0.5) / 4.0
                y = (i + 0.5) / 4.0
                
                # Add small perturbations to avoid regular patterns
                if i % 2 == 0:
                    x += np.random.normal(0, 0.015, 1)[0]
                else:
                    y += np.random.normal(0, 0.015, 1)[0]
                
                # Clamp to [0,1]
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
                
        return np.array(points)
    
    def generate_refined_spiral():
        """Generate a refined spiral arrangement"""
        points = []
        n = 16
        golden_angle = 2.399963229728653  # ~2π(1 - 1/φ) where φ is golden ratio
        
        # Use a more sophisticated spiral with radial variation
        for i in range(n):
            # Radial scaling with a twist to distribute points better
            r = 0.4 * (1 - np.exp(-i * 0.3))  # Exponential increase to fill space
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
    
    def generate_kmeans_plus_plus():
        """Generate configuration using k-means++ like approach"""
        # Start with random points
        points = np.random.rand(16, 2)
        
        # Use a simple iterative approach to spread points
        for _ in range(100):
            # Compute distances
            distances = pdist(points)
            if len(distances) == 0:
                break
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            # If already well distributed, stop early
            if min_dist > 0.1 and max_dist < 0.8:
                break
                
            # Move points apart slightly
            for i in range(16):
                # Find nearest neighbor
                nearest_idx = -1
                nearest_dist = 1e10
                for j in range(16):
                    if i != j:
                        dist = np.sqrt((points[i, 0] - points[j, 0])**2 + (points[i, 1] - points[j, 1])**2)
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_idx = j
                
                if nearest_idx >= 0 and nearest_dist < 0.2:
                    # Move point away from its nearest neighbor
                    dx = points[i, 0] - points[nearest_idx, 0]
                    dy = points[i, 1] - points[nearest_idx, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 1e-10:
                        # Normalize and scale
                        dx /= dist
                        dy /= dist
                        # Move away by a small amount
                        points[i, 0] += dx * 0.01
                        points[i, 1] += dy * 0.01
                        
                        # Clamp to [0,1]
                        points[i, 0] = np.clip(points[i, 0], 0, 1)
                        points[i, 1] = np.clip(points[i, 1], 0, 1)
        
        return points
    
    def generate_optimized_config():
        """Generate an optimized configuration with better distribution"""
        # Start with a good known configuration - regular grid with perturbations
        points = []
        # Create a 4x4 grid with strategic perturbations
        for i in range(4):
            for j in range(4):
                x = j / 3.0 + np.random.normal(0, 0.02, 1)[0]
                y = i / 3.0 + np.random.normal(0, 0.02, 1)[0]
                # Ensure points stay in valid range
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
        return np.array(points)
    
    # Generate a few carefully selected initial configurations
    configs = [
        generate_hexagonal_grid(),      # Hexagonal grid
        generate_golden_spiral(),       # Golden spiral
        generate_circular_arrangement(), # Circular arrangement
        generate_perturbed_grid(),      # Perturbed grid
        generate_clustered_config(),    # Clustered config
        generate_fibonacci_sphere(),    # Fibonacci sphere
        generate_regular_polygon(),     # Regular polygon
        generate_voronoi_like(),        # Voronoi-inspired
        generate_better_initialization(), # Better initialization
        generate_concentric_circles(),  # Concentric circles
        generate_symmetric_arrangement(), # Symmetric arrangement
        generate_better_grid(),         # Better grid
        generate_refined_spiral(),      # Refined spiral
        generate_kmeans_plus_plus(),    # K-means++ inspired
        generate_optimized_config(),    # Optimized config
        np.random.rand(16, 2)           # Random configuration
    ]
    
    best_ratio = 0
    best_points = None
    start_time = time.time()
    
    # Try different optimization approaches with better parameter tuning
    # Focus on fewer, better strategies rather than many diverse ones
    for i, initial_config in enumerate(configs):
        if time.time() - start_time > 55:  # Leave some time for final steps
            break
            
        initial_flat = initial_config.flatten()
        
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            # Strategy 1: Differential Evolution with better parameters
            for restart in range(2):  # Reduced restarts for faster execution
                try:
                    # Add small random perturbation to initial guess for diversity
                    perturbed_initial = initial_flat + np.random.normal(0, 0.01, 32)
                    perturbed_initial = np.clip(perturbed_initial, 0, 1)
                    
                    # Use differential evolution with higher quality settings
                    result = differential_evolution(
                        objective_with_penalty,
                        bounds,
                        maxiter=1000,  # Reduced iterations to save time
                        popsize=50,    # Moderate population size
                        seed=42+i+restart,
                        disp=False,
                        atol=1e-14,    # Tighter tolerances
                        rtol=1e-14,
                        strategy='best1bin'
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        
                        # Validate result
                        distances = pdist(final_points)
                        if len(distances) > 0:
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            if d_max > 1e-12:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = final_points.copy()
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    # If no good solution found, try global optimization with better parameters
    if best_points is None and time.time() - start_time < 50:
        try:
            # Use dual annealing for global optimization with more iterations
            bounds = [(0, 1) for _ in range(32)]
            
            result = dual_annealing(
                objective_with_penalty,
                bounds,
                maxiter=3000,   # Reduced iterations for speed
                initial_temp=2000,  # Lower initial temperature
                seed=42,
                no_local_search=True
            )
            
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            
            # Validate and compare
            distances = pdist(final_points)
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 1e-12:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_points = final_points
                        
        except Exception as e:
            pass
    
    # If still no good solution, try a hybrid approach with more aggressive local search
    if best_points is None and time.time() - start_time < 50:
        try:
            # Start with golden spiral as baseline
            initial_points = generate_golden_spiral()
            initial_flat = initial_points.flatten()
            
            # Use L-BFGS-B with multiple restarts and better tolerance
            for restart in range(3):  # Reduced restarts
                try:
                    # Add small random perturbation
                    perturbed_initial = initial_flat + np.random.normal(0, 0.01, 32)
                    perturbed_initial = np.clip(perturbed_initial, 0, 1)
                    
                    result = minimize(
                        objective_with_penalty,
                        perturbed_initial,
                        method='L-BFGS-B',
                        bounds=[(0, 1) for _ in range(32)],
                        options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13},  # Reduced iterations
                        tol=1e-15
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        
                        # Validate and compare
                        distances = pdist(final_points)
                        if len(distances) > 0:
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            if d_max > 1e-12:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = final_points.copy()
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
    
    # Final refinement with local optimization using a more robust approach
    if best_points is not None and time.time() - start_time < 55:
        try:
            # Try to improve with local optimization using the best configuration
            initial_flat = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            # Use only one local optimization strategy to save time
            strategies = [
                ('L-BFGS-B', {'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13})
            ]
            
            for method, options in strategies:
                if time.time() - start_time > 55:
                    break
                try:
                    result = minimize(
                        objective_with_penalty,
                        initial_flat,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-15
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        
                        # Validate and compare
                        distances = pdist(final_points)
                        if len(distances) > 0:
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            if d_max > 1e-12:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = final_points
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
    
    # Final validation check - try one more global optimization approach
    if best_points is None and time.time() - start_time < 50:
        try:
            # Try basin hopping which often works well for this type of problem
            from scipy.optimize import basinhopping
            
            # Start with a good configuration
            initial_points = generate_optimized_config()
            initial_flat = initial_points.flatten()
            
            def callback(x, f, accepted):
                pass  # No callback needed
            
            result = basinhopping(
                objective_with_penalty,
                initial_flat,
                niter=100,  # Reduced iterations for speed
                T=1.0,      # Temperature
                stepsize=0.1,  # Step size
                seed=42,
                callback=callback
            )
            
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            
            # Validate and compare
            distances = pdist(final_points)
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 1e-12:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_points = final_points
                        
        except Exception as e:
            pass
    
    # Final validation check
    if best_points is None:
        # Last resort: return the optimized configuration
        return generate_optimized_config()
    
    return best_points


# EVOLVE-BLOCK-END
