# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and gradient-based optimization.

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
    
    # Initialize with a hexagonal pattern like inspiration programs
    def initialize_hexagonal_pattern():
        # Create a 4x4 grid pattern for 16 points
        rows, cols = 4, 4
        points = []
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n:
                    break
                # Hexagonal offset
                x = j + 0.5 * (i % 2)
                y = i * np.sqrt(3) / 2
                points.append([x, y])
        
        # Normalize to fit in unit square [0,1] x [0,1]
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
    
    # Start with hexagonal pattern
    points = initialize_hexagonal_pattern()
    
    # Multi-start optimization with different initializations
    best_ratio = 0
    best_points = points.copy()
    
    # Try multiple random restarts (15 as suggested in inspiration 2)
    for restart in range(15):
        np.random.seed(42 + restart * 100)
        
        # Slightly perturb the hexagonal pattern (perturbation from inspiration 2: 0.03)
        initial_points = points + np.random.normal(0, 0.03, (n, d))
        
        # Ensure points stay within bounds
        initial_points = np.clip(initial_points, 0, 1)
        
        # Try both optimization methods for robustness
        methods_to_try = ['L-BFGS-B', 'SLSQP']
        
        for method in methods_to_try:
            try:
                result = minimize(
                    objective,
                    initial_points.flatten(),
                    method=method,
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                if result.success:
                    optimized_points = result.x.reshape((n, d))
                    ratio = compute_min_max_ratio(optimized_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
            except Exception:
                continue
    
    # Final refinement with high precision optimization
    try:
        final_result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if final_result.success:
            best_points = final_result.x.reshape((n, d))
    except Exception:
        pass
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    # One final check to make sure we have a valid result
    if not np.isfinite(best_points).all():
        # Fallback to regular hexagon if something went wrong
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.mean(points, axis=0)
        best_points = (points - center) * 0.4 + 0.5
    
    return best_points


# EVOLVE-BLOCK-END
