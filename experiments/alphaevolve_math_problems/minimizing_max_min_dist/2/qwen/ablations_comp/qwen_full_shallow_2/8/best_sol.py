# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and gradient-based optimization
    with focus on computational efficiency and quality results.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    d = 2
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 1e-12:
            return 0
        
        return d_min / d_max

    def objective(params):
        """Objective function to minimize (negative of min/max ratio)"""
        points = params.reshape((n, d))
        distances = pdist(points)
        
        if len(distances) == 0:
            return 1e10
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 1e-12:
            return 1e10
        
        # Return negative ratio to maximize it (minimize negative)
        return -d_min / d_max

    def initialize_hexagonal_pattern():
        """Initialize points in a hexagonal pattern for better starting distribution"""
        # Create a 4x4 hexagonal grid pattern
        points = []
        spacing = 0.25
        
        for i in range(4):
            for j in range(4):
                if len(points) < n:
                    x = j * spacing + (i % 2) * spacing/2
                    y = i * spacing * np.sqrt(3)/2
                    points.append([x, y])
        
        # Normalize to fit in unit square and center properly
        points = np.array(points[:n])
        if len(points) > 0:
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)
            # Avoid division by zero
            range_x = max_x - min_x if max_x - min_x > 1e-10 else 1
            range_y = max_y - min_y if max_y - min_y > 1e-10 else 1
            
            points[:, 0] = (points[:, 0] - min_x) / range_x * 0.8 + 0.1
            points[:, 1] = (points[:, 1] - min_y) / range_y * 0.8 + 0.1
        
        return points

    # Initialize with hexagonal pattern (more reliable than complex patterns)
    points = initialize_hexagonal_pattern()
    
    # Multi-start optimization with fewer restarts for better performance
    best_ratio = 0
    best_points = points.copy()
    
    # Use only 8 restarts instead of 15 to reduce computation time
    for restart in range(8):
        np.random.seed(42 + restart * 100)
        
        # Start with slightly perturbed version of hexagonal pattern
        initial_points = points + np.random.normal(0, 0.03, (n, d))
        
        # Ensure points stay within bounds after perturbation
        initial_points = np.clip(initial_points, 0, 1)
        
        # Try multiple optimization methods for robustness
        methods_to_try = ['L-BFGS-B', 'SLSQP']
        
        for method in methods_to_try:
            try:
                result = minimize(
                    objective,
                    initial_points.flatten(),
                    method=method,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                if result.success:
                    optimized_points = result.x.reshape((n, d))
                    ratio = compute_min_max_ratio(optimized_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
            except Exception:
                continue
    
    # Final refinement with gradient-based optimization
    try:
        final_result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if final_result.success:
            best_points = final_result.x.reshape((n, d))
    except Exception:
        pass
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
