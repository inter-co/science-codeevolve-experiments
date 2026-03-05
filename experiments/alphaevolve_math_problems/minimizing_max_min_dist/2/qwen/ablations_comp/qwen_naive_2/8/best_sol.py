# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import warnings
from typing import Tuple
from numba import jit
import time
from sklearn.cluster import KMeans
from scipy.spatial import distance
import random
from joblib import Parallel, delayed
import multiprocessing

@jit(nopython=True)
def compute_distances_fast(points):
    """Fast computation of pairwise distances using Numba"""
    n = points.shape[0]
    distances = np.zeros(n * (n - 1) // 2)
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
    Uses advanced optimization techniques with better initialization and convergence strategies.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    
    def compute_distances(points):
        """Compute all pairwise distances efficiently"""
        if len(points) < 2:
            return np.array([])
        return compute_distances_fast(points)
    
    def objective_ratio(points_flat):
        """Objective function to maximize ratio of min/max distances"""
        points = points_flat.reshape(-1, 2)
        distances = compute_distances(points)
        
        if len(distances) == 0 or len(distances) < 2:
            return -np.inf
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 1e-12:
            return -np.inf
            
        return min_dist / max_dist
    
    def energy_objective(points_flat):
        """Energy-based objective (minimize negative ratio)"""
        points = points_flat.reshape(-1, 2)
        distances = compute_distances(points)
        
        if len(distances) == 0 or len(distances) < 2:
            return np.inf
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 1e-12:
            return np.inf
            
        # We want to maximize min/max ratio, so we minimize -ratio
        ratio = min_dist / max_dist
        return -ratio
    
    def get_initial_points_fibonacci() -> np.ndarray:
        """Generate initial points using Fibonacci-like distribution"""
        points = []
        # Use Fibonacci-inspired distribution for good coverage
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            # Better Fibonacci distribution in 2D
            theta = i * 2 * np.pi / golden_ratio
            r = np.sqrt(i / (n - 1)) if n > 1 else 0.5
            x = 0.5 + r * np.cos(theta) * 0.4
            y = 0.5 + r * np.sin(theta) * 0.4
            points.append([x, y])
        
        points = np.array(points)
        # Clip to [0,1] range
        points = np.clip(points, 0, 1)
        return points
    
    def get_initial_points_hexagonal() -> np.ndarray:
        """Generate high-quality initial point configurations using hexagonal packing"""
        points = []
        # Create hexagonal lattice pattern with 4x4 grid
        rows = 4
        cols = 4
        spacing = 0.8 / (rows - 1) if rows > 1 else 0.8
        offset = spacing / 2 if cols > 1 else 0.4
        
        for i in range(rows):
            for j in range(cols):
                if len(points) < n:
                    x = 0.1 + j * spacing
                    y = 0.1 + i * spacing + (offset if i % 2 == 1 else 0)
                    points.append([x, y])
        
        # Adjust to exactly n points and normalize
        points = np.array(points[:n])
        if len(points) < n:
            # Fill remaining points with random points
            extra_points = np.random.rand(n - len(points), 2) * 0.8 + 0.1
            points = np.vstack([points, extra_points])
        else:
            points = points[:n]
            
        # Normalize to [0,1]x[0,1] range
        points = np.clip(points, 0, 1)
        
        # Add slight perturbations to avoid degenerate cases
        points += np.random.normal(0, 0.005, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    def get_initial_points_spiral() -> np.ndarray:
        """Generate spiral-based initial configuration"""
        points = np.zeros((n, 2))
        # Use a spiral pattern that spreads points evenly
        for i in range(n):
            angle = 2 * np.pi * i / n * 4  # More turns for better distribution
            radius = 0.4 * (i / (n - 1)) + 0.1  # Spread points radially
            points[i] = [
                0.5 + radius * np.cos(angle),
                0.5 + radius * np.sin(angle)
            ]
        
        # Add small random perturbations
        points += np.random.normal(0, 0.01, points.shape) * 0.5
        points = np.clip(points, 0, 1)
        
        return points
    
    def get_initial_points_grid() -> np.ndarray:
        """Generate grid-based initial configuration"""
        points = []
        # Create a 4x4 grid pattern
        for i in range(4):
            for j in range(4):
                if len(points) < n:
                    x = 0.1 + 0.8 * j / 3
                    y = 0.1 + 0.8 * i / 3
                    points.append([x, y])
        
        points = np.array(points[:n])
        if len(points) < n:
            extra_points = np.random.rand(n - len(points), 2) * 0.8 + 0.1
            points = np.vstack([points, extra_points])
        else:
            points = points[:n]
            
        points = np.clip(points, 0, 1)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    def get_initial_points_kmeans() -> np.ndarray:
        """Generate initial points using k-means clustering approach"""
        # Generate random points and then cluster them to create well-distributed points
        random_points = np.random.rand(32, 2) * 0.8 + 0.1  # [0.1, 0.9] range
        
        # Use k-means to find centers of 16 clusters
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(random_points)
        
        # Get cluster centers as initial points
        points = kmeans.cluster_centers_
        
        # Ensure they're within bounds
        points = np.clip(points, 0, 1)
        
        # Add small random perturbations
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    def get_initial_points_custom() -> np.ndarray:
        """Generate a custom initial configuration based on known good patterns"""
        # Create a configuration inspired by optimal point distributions
        points = []
        
        # Place some points near corners and center
        corner_points = [
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
            [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]
        ]
        
        # Add more points in a structured way
        for i in range(8):
            if len(points) < n:
                points.append(corner_points[i % len(corner_points)])
        
        # Fill remaining spots with random but distributed points
        while len(points) < n:
            x = 0.1 + np.random.random() * 0.8
            y = 0.1 + np.random.random() * 0.8
            points.append([x, y])
        
        points = np.array(points[:n])
        
        # Add slight random perturbations
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    def get_initial_points_poisson() -> np.ndarray:
        """Generate points using a Poisson disk sampling approach"""
        # Start with a simple grid and add jitter
        points = []
        grid_size = 4
        cell_size = 0.8 / (grid_size - 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < n:
                    x = 0.1 + j * cell_size + np.random.uniform(-cell_size/4, cell_size/4)
                    y = 0.1 + i * cell_size + np.random.uniform(-cell_size/4, cell_size/4)
                    points.append([x, y])
        
        points = np.array(points[:n])
        points = np.clip(points, 0, 1)
        return points
    
    def get_initial_points_sphere() -> np.ndarray:
        """Generate points using spherical distribution (inspired by sphere packing)"""
        # Generate points on a sphere using Fibonacci method, then project to 2D
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = golden_angle * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Project to 2D (simplest projection)
            points.append([0.5 + x * 0.4, 0.5 + z * 0.4])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    def optimize_with_multiple_strategies() -> Tuple[np.ndarray, float]:
        """Try multiple optimization strategies and return best result"""
        best_points = None
        best_ratio = -np.inf
        
        # Strategy 1: Multiple differential evolution runs with different parameters
        try:
            bounds = [(0, 1) for _ in range(2*n)]
            
            # Try multiple initial configurations with DE optimization
            initial_configs = [
                get_initial_points_fibonacci(),
                get_initial_points_hexagonal(),
                get_initial_points_spiral(), 
                get_initial_points_grid(),
                get_initial_points_kmeans(),
                get_initial_points_custom(),
                get_initial_points_poisson(),
                get_initial_points_sphere()
            ]
            
            # Use parallel processing for multiple DE runs
            def run_de_optimization(initial_points, seed_val):
                try:
                    res_de = differential_evolution(
                        energy_objective,
                        bounds,
                        seed=seed_val,
                        maxiter=500,
                        popsize=40,
                        mutation=(0.5, 1),
                        recombination=0.7,
                        atol=1e-16,
                        rtol=1e-16,
                        disp=False
                    )
                    if res_de.success:
                        optimized_points = res_de.x.reshape(-1, 2)
                        optimized_points = np.clip(optimized_points, 0, 1)
                        return optimized_points
                except Exception:
                    return None
                return None
            
            # Run DE optimizations in parallel
            results = Parallel(n_jobs=min(multiprocessing.cpu_count(), 4))(
                delayed(run_de_optimization)(initial_points, 42 + i)
                for i, initial_points in enumerate(initial_configs)
            )
            
            # Process results
            for optimized_points in results:
                if optimized_points is not None:
                    final_distances = compute_distances(optimized_points)
                    if len(final_distances) > 0:
                        min_dist = np.min(final_distances)
                        max_dist = np.max(final_distances)
                        if max_dist > 1e-15:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
        except Exception as e:
            pass
        
        # Strategy 2: Local optimization with better convergence
        if best_points is None:
            # Try with multiple starting configurations using multiple methods
            initial_configs = [
                get_initial_points_fibonacci(),
                get_initial_points_custom(),
                get_initial_points_poisson(),
                get_initial_points_sphere()
            ]
            
            # Try multiple optimization methods with better convergence control
            methods = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']
            max_iter = 500
            
            for i, initial_points in enumerate(initial_configs):
                for restart in range(3):
                    np.random.seed(42 + i * 10 + restart)
                    
                    # Add noise to break symmetry
                    noisy_points = initial_points + np.random.normal(0, 0.02, initial_points.shape)
                    noisy_points = np.clip(noisy_points, 0, 1)
                    
                    try:
                        bounds = [(0, 1) for _ in range(2*n)]
                        
                        for method in methods:
                            try:
                                res = minimize(
                                    energy_objective,
                                    noisy_points.flatten(),
                                    method=method,
                                    bounds=bounds,
                                    options={
                                        'maxiter': max_iter, 
                                        'ftol': 1e-16, 
                                        'gtol': 1e-16,
                                        'disp': False
                                    }
                                )
                                
                                if res.success:
                                    optimized_points = res.x.reshape(-1, 2)
                                    optimized_points = np.clip(optimized_points, 0, 1)
                                    
                                    final_distances = compute_distances(optimized_points)
                                    if len(final_distances) > 0:
                                        min_dist = np.min(final_distances)
                                        max_dist = np.max(final_distances)
                                        if max_dist > 1e-15:
                                            ratio = min_dist / max_dist
                                            if ratio > best_ratio:
                                                best_ratio = ratio
                                                best_points = optimized_points.copy()
                            except Exception:
                                continue
                    except Exception:
                        continue
        
        # Strategy 3: Hybrid approach combining global and local search
        if best_points is None:
            try:
                # Start with a good configuration
                points = get_initial_points_fibonacci()
                
                # First do a coarse optimization with DE
                bounds = [(0, 1) for _ in range(2*n)]
                res_de = differential_evolution(
                    energy_objective,
                    bounds,
                    seed=42,
                    maxiter=200,
                    popsize=30,
                    mutation=(0.8, 1),
                    recombination=0.9,
                    disp=False
                )
                
                if res_de.success:
                    points = res_de.x.reshape(-1, 2)
                    points = np.clip(points, 0, 1)
                
                # Then fine-tune with local optimization
                res_local = minimize(
                    energy_objective,
                    points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-16}
                )
                
                if res_local.success:
                    optimized_points = res_local.x.reshape(-1, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    
                    final_distances = compute_distances(optimized_points)
                    if len(final_distances) > 0:
                        min_dist = np.min(final_distances)
                        max_dist = np.max(final_distances)
                        if max_dist > 1e-15:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
            except Exception:
                pass
        
        # If still no good solution, return the fibonacci pattern
        if best_points is None:
            points = get_initial_points_fibonacci()
            return points, 0.0
        
        return best_points, best_ratio
    
    # Execute optimization with timeout protection
    start_time = time.time()
    
    # Execute optimization
    final_points, ratio = optimize_with_multiple_strategies()
    
    # Final verification and cleanup
    if final_points is not None:
        # Ensure we have valid points
        final_points = np.clip(final_points, 0, 1)
        # Verify the ratio calculation
        distances = compute_distances(final_points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 1e-15:
                calculated_ratio = min_dist / max_dist
                # If our ratio is significantly worse than what we computed,
                # recompute more carefully
                if abs(calculated_ratio - ratio) > 1e-6:
                    ratio = calculated_ratio
    
    return final_points


# EVOLVE-BLOCK-END
