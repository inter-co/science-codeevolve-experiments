# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import warnings
from scipy.optimize import differential_evolution
from numba import jit
import time


@jit(nopython=True)
def compute_distances_jit(points):
    """Compute pairwise distances efficiently using Numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    n = 16
    np.random.seed(42)
    
    # Strategy 1: Better initialization - use a more systematic approach
    # Start with a regular pattern and perturb appropriately
    # Create a grid-like structure with some randomness
    grid_size = 4
    points_grid = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points_grid) < n:
                # Add some jitter to make it non-regular
                x = (j + 0.5 + np.random.normal(0, 0.1)) / grid_size
                y = (i + 0.5 + np.random.normal(0, 0.1)) / grid_size
                points_grid.append([x, y])
    
    # Ensure we have exactly n points
    initial_points = np.array(points_grid[:n])
    
    # Clip to [0,1] range
    initial_points = np.clip(initial_points, 0, 1)
    
    # Strategy 2: Improved objective function for maximization
    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Use efficient distance computation
        distances = pdist(points)
        
        # Numerical safety
        distances = np.maximum(distances, 1e-12)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # To maximize min/max ratio, we want to minimize 1/(min/max) = max/min
        # But we want to maximize min/max, so we should minimize max/min
        # However, we want to maximize min/max, so we want to minimize max/min
        # So we return max_dist / min_dist to minimize the inverse ratio
        # But to maximize min/max, we minimize max/min, so we return max_dist / min_dist
        # This is correct: when max/min is small, min/max is large
        if min_dist < 1e-12:
            return float('inf')  # Penalize invalid configurations
        return max_dist / min_dist  # Minimizing this maximizes min/max ratio
    
    # Strategy 3: Enhanced optimization with better convergence control
    best_ratio = 0
    best_points = initial_points.copy()
    
    # Approach 1: Differential Evolution with better parameters
    try:
        bounds = [(0, 1) for _ in range(2*n)]
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-12,
            mutation=(0.8, 1.0),
            recombination=0.9,
            strategy='best1bin'
        )
        
        if de_result.success:
            optimized_points = de_result.x.reshape(-1, 2)
            # Calculate final ratio
            dist_matrix = squareform(pdist(optimized_points))
            min_dist = np.min(dist_matrix[dist_matrix > 0])
            max_dist = np.max(dist_matrix)
            
            if max_dist > 0 and min_dist / max_dist > best_ratio:
                best_ratio = min_dist / max_dist
                best_points = optimized_points.copy()
                
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
    
    # Approach 2: Multiple local optimizations with better restart strategy
    methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
    for method in methods_to_try:
        for restart in range(3):  # Fewer restarts for speed
            # Generate random restart points with better spread
            x0 = initial_points.flatten() + np.random.normal(0, 0.02, 2*n)
            x0 = np.clip(x0, 0, 1)
            
            bounds = [(0, 1) for _ in range(2*n)]
            
            try:
                result = minimize(
                    objective, 
                    x0, 
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-13}
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    # Calculate final ratio
                    dist_matrix = squareform(pdist(optimized_points))
                    min_dist = np.min(dist_matrix[dist_matrix > 0])
                    max_dist = np.max(dist_matrix)
                    
                    if max_dist > 0 and min_dist / max_dist > best_ratio:
                        best_ratio = min_dist / max_dist
                        best_points = optimized_points.copy()
                        
            except Exception as e:
                continue
    
    # Approach 3: Gradient-based optimization with adaptive learning rate
    if best_ratio < 0.20:  # Only refine if not already quite good
        # Use a more sophisticated optimization approach
        points = best_points.copy()
        
        # Adam-style optimization with adaptive learning rate
        learning_rate = 0.005
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        m = np.zeros_like(points)  # First moment
        v = np.zeros_like(points)  # Second moment
        
        for iteration in range(1000):
            # Compute distances
            dist_matrix = squareform(pdist(points))
            dist_matrix = np.maximum(dist_matrix, 1e-12)
            
            # Compute gradient approximation using finite differences
            grad = np.zeros_like(points)
            eps = 1e-6
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        diff = points[i] - points[j]
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            # Approximate gradient of max_dist / min_dist
                            # Simplified approach: gradient of distance function
                            grad[i] += diff / (dist * (dist + 1e-12))
                            grad[j] -= diff / (dist * (dist + 1e-12))
            
            # Apply Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**(iteration+1))
            v_hat = v / (1 - beta2**(iteration+1))
            
            points -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Keep within bounds
            points = np.clip(points, 0, 1)
            
            # Reduce learning rate over time
            if iteration % 100 == 0 and iteration > 0:
                learning_rate *= 0.95
            
            # Occasionally reinitialize some points to escape local minima
            if iteration % 300 == 0 and iteration > 0:
                mask = np.random.random(n) < 0.05
                points[mask] = np.random.uniform(0, 1, (np.sum(mask), 2))
        
        # Final evaluation
        dist_matrix = squareform(pdist(points))
        min_dist = np.min(dist_matrix[dist_matrix > 0])
        max_dist = np.max(dist_matrix)
        final_ratio = min_dist / max_dist
        
        if final_ratio > best_ratio:
            best_points = points
    
    # Strategy 4: Final refinement using more aggressive optimization
    try:
        # Create a refined version with more careful optimization
        bounds = [(0, 1) for _ in range(2*n)]
        x0 = best_points.flatten()
        
        # Use L-BFGS-B with very strict tolerances
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-16}
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            dist_matrix = squareform(pdist(optimized_points))
            min_dist = np.min(dist_matrix[dist_matrix > 0])
            max_dist = np.max(dist_matrix)
            final_ratio = min_dist / max_dist
            
            if max_dist > 0 and final_ratio > best_ratio:
                best_points = optimized_points
                
    except Exception as e:
        pass
    
    # Strategy 5: Try a completely different initialization if we're still not good enough
    if best_ratio < 0.20:
        # Try a better initialization strategy - circle packing approach
        try:
            # Try placing points in a circular pattern with some randomness
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            # Vary radii slightly to create a more dispersed pattern
            radii = 0.4 + 0.1 * np.sin(np.arange(n) * 2 * np.pi / n)
            
            circle_points = np.column_stack([
                0.5 + radii * np.cos(angles),
                0.5 + radii * np.sin(angles)
            ])
            
            # Add noise to break symmetry
            circle_points += np.random.normal(0, 0.01, (n, 2))
            circle_points = np.clip(circle_points, 0, 1)
            
            # Optimize this initialization
            bounds = [(0, 1) for _ in range(2*n)]
            x0 = circle_points.flatten()
            
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-13}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                dist_matrix = squareform(pdist(optimized_points))
                min_dist = np.min(dist_matrix[dist_matrix > 0])
                max_dist = np.max(dist_matrix)
                final_ratio = min_dist / max_dist
                
                if max_dist > 0 and final_ratio > best_ratio:
                    best_points = optimized_points
                    
        except Exception as e:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
