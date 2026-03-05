# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from numba import jit
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


@jit(nopython=True)
def compute_min_max_ratio_jit(points):
    """JIT compiled version for faster computation of min/max distance ratio"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq)
            
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    if max_dist <= 0:
        return -np.inf
    return min_dist / max_dist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses advanced optimization techniques with multiple restarts and better initialization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    
    def objective(x_flat):
        """Objective function: negative ratio of min/max distances (to maximize ratio)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Use JIT compiled version for performance
        ratio = compute_min_max_ratio_jit(points)
        
        # Return negative ratio (since we want to maximize ratio, minimize negative ratio)
        return -ratio if ratio != -np.inf else 1e10
    
    def generate_initial_points():
        """Generate better initial points using a more strategic approach"""
        # Strategy 1: Try to create a configuration that avoids clustering
        # Start with a hexagonal-like pattern for good distribution
        
        # Create points in a circular pattern with some randomness
        angles = np.linspace(0, 2*np.pi, 16)
        radii = np.ones(16) * 0.4  # Radius slightly less than 0.5 to keep away from edges
        
        # Add some variation to make it more generic
        radii += np.random.normal(0, 0.05, 16)
        radii = np.clip(radii, 0.05, 0.45)
        
        x = 0.5 + radii * np.cos(angles)
        y = 0.5 + radii * np.sin(angles)
        
        points = np.column_stack([x, y])
        
        # Add small random noise to break symmetries
        points += np.random.normal(0, 0.01, points.shape)
        
        # Clip to bounds
        points = np.clip(points, 0, 1)
        return points
    
    def generate_multiple_starts(n_starts=10):
        """Generate multiple starting points and run optimization from each"""
        best_ratio = -np.inf
        best_points = None
        best_result = None
        
        for start_idx in range(n_starts):
            # Generate different initial configurations
            if start_idx == 0:
                # First start: hexagonal pattern
                initial_points = generate_initial_points()
            elif start_idx == 1:
                # Second start: random points
                initial_points = np.random.rand(16, 2)
            elif start_idx == 2:
                # Third start: grid pattern with some noise
                grid_size = 4
                x = np.linspace(0.05, 0.95, grid_size)
                y = np.linspace(0.05, 0.95, grid_size)
                X, Y = np.meshgrid(x, y)
                points = np.column_stack([X.ravel(), Y.ravel()])[:16]
                points += np.random.normal(0, 0.02, points.shape)
                points = np.clip(points, 0, 1)
                initial_points = points
            else:
                # Later starts: perturbed versions of previous best or random
                if best_points is not None and start_idx < 7:
                    initial_points = best_points + np.random.normal(0, 0.01, (16, 2))
                    initial_points = np.clip(initial_points, 0, 1)
                else:
                    initial_points = np.random.rand(16, 2)
            
            initial_flat = initial_points.flatten()
            
            # Define bounds for coordinates
            bounds = [(0, 1) for _ in range(32)]
            
            # Try different optimization methods
            methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
            
            for method in methods_to_try:
                try:
                    result = minimize(
                        objective,
                        initial_flat,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-7},
                        callback=None
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 2)
                        optimized_points = np.clip(optimized_points, 0, 1)
                        
                        # Evaluate final ratio
                        final_ratio = compute_min_max_ratio_jit(optimized_points)
                        
                        if final_ratio > best_ratio:
                            best_ratio = final_ratio
                            best_points = optimized_points.copy()
                            best_result = result
                        
                except Exception as e:
                    continue
                    
        return best_points if best_points is not None else generate_initial_points()
    
    # Run multiple optimization attempts with more aggressive settings
    optimized_points = generate_multiple_starts(n_starts=8)
    
    # Final aggressive refinement with high iteration count
    initial_flat = optimized_points.flatten()
    bounds = [(0, 1) for _ in range(32)]
    
    try:
        result = minimize(
            objective,
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10},
            callback=None
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 2)
            refined_points = np.clip(refined_points, 0, 1)
            
            # Verify improvement
            original_ratio = compute_min_max_ratio_jit(optimized_points)
            refined_ratio = compute_min_max_ratio_jit(refined_points)
            
            if refined_ratio > original_ratio:
                optimized_points = refined_points
                
    except Exception:
        pass
    
    return optimized_points


# EVOLVE-BLOCK-END
