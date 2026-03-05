# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses cube-based initialization and robust optimization with multiple restarts.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x.reshape(14, 3)
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return float('inf')  # Very bad objective value
            
        # Return negative ratio (since we want to maximize)
        return -d_min / d_max
    
    # Better initialization based on cube vertices and symmetric points
    def get_better_initial_configuration():
        # Start with vertices of a cube (8 points) 
        points = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    points.append([i, j, k])
        
        # Add points along axes and diagonals for better distribution
        points.extend([
            [1.5, 0, 0], [-1.5, 0, 0],  # x-axis
            [0, 1.5, 0], [0, -1.5, 0],  # y-axis  
            [0, 0, 1.5], [0, 0, -1.5],  # z-axis
            [0.707, 0.707, 0.707], [-0.707, -0.707, -0.707],  # diagonals
            [0.707, -0.707, 0.707], [-0.707, 0.707, -0.707]   # more diagonals
        ])
        
        # Keep only first 14 points and normalize
        points = np.array(points[:14], dtype=float)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        return points
    
    # Get initial configuration
    initial_points = get_better_initial_configuration()
    
    # Flatten for optimization
    x0 = initial_points.flatten()
    
    # Use enhanced restart strategy with multiple configurations and methods
    best_points = None
    best_ratio = -np.inf
    
    # Enhanced restart strategy with better parameter combinations from inspirations
    restart_configs = [
        (0.02, 750),   # Medium perturbation, moderate iterations
        (0.03, 1000),  # Larger perturbation, more iterations  
        (0.025, 1000), # Balanced approach
    ]
    
    # Try multiple optimization methods for better convergence
    methods = ['L-BFGS-B', 'SLSQP']
    
    for restart_idx, (perturbation, max_iter) in enumerate(restart_configs):
        np.random.seed(42 + restart_idx)
        
        # Slightly perturb initial solution for each restart
        x_start = x0 + np.random.normal(0, perturbation, len(x0))
        # Ensure points stay on unit sphere after perturbation
        x_start = x_start.reshape(14, 3)
        norms = np.linalg.norm(x_start, axis=1, keepdims=True)
        x_start = (x_start / norms).flatten()
        
        try:
            # Try multiple methods for better convergence
            for method in methods:
                result = minimize(
                    objective,
                    x_start,
                    method=method,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )
                
                if result.success:
                    optimized_points = result.x.reshape(14, 3)
                    # Evaluate the result
                    distances = pdist(optimized_points)
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    
                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
        except Exception:
            continue
    
    # If no good optimization was found, return the best initial configuration
    if best_points is None:
        return initial_points
    
    return best_points


# EVOLVE-BLOCK-END
