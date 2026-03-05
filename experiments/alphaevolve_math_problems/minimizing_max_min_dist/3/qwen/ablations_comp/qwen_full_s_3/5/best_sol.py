# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.distance import pdist, squareform
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    
    # Define objective function to maximize min/max ratio
    def objective(x):
        # Reshape flat array back to points
        points = x.reshape(n, d)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return -np.inf
            
        # Avoid division by zero
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return -np.inf
            
        # We want to maximize min/max ratio, so we minimize -min/max ratio
        return -min_dist / max_dist
    
    # Define constraint function for bounded domain [0,1]^3
    def constraint_func(x):
        points = x.reshape(n, d)
        # Ensure all points are within [0,1]^3
        return np.concatenate([
            points.flatten() - 1,  # x <= 1
            -points.flatten()     # x >= 0
        ])
    
    # Use a more sophisticated optimization approach
    # Start with a good initial configuration based on known optimal patterns
    np.random.seed(42)
    
    # Try multiple starting points with different strategies
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Random initialization with energy minimization
    for _ in range(5):
        # Generate random points in [0,1]^3
        initial_points = np.random.rand(n, d)
        
        # Flatten for optimization
        x0 = initial_points.flatten()
        
        # Use dual annealing for global optimization
        try:
            result = dual_annealing(
                objective, 
                bounds=[(0, 1) for _ in range(n * d)],
                maxiter=1000,
                seed=42,
                no_local_search=True
            )
            
            if result.success:
                points = result.x.reshape(n, d)
                distances = pdist(points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = points.copy()
        except:
            continue
    
    # Strategy 2: Use a known good starting configuration and refine
    if best_points is None:
        # Generate a configuration inspired by sphere packing
        # Use Fibonacci spiral approach for better distribution
        golden_ratio = (1 + np.sqrt(5)) / 2
        points = []
        for i in range(n):
            theta = np.arccos(-1 + (2 * i) / (n - 1))
            phi = np.sqrt(n * np.pi) * golden_ratio * i
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        
        # Normalize to [0,1]^3
        points = np.array(points)
        points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
        best_points = points
    
    # Refine with local optimization if needed
    if best_points is not None:
        # Convert to flattened form for scipy optimization
        x0 = best_points.flatten()
        
        # Try several local optimization approaches
        for method in ['L-BFGS-B', 'TNC']:
            try:
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=[(0, 1) for _ in range(n * d)],
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                if result.success:
                    points = result.x.reshape(n, d)
                    distances = pdist(points)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = points.copy()
            except:
                continue
    
    # Final fallback to a good configuration
    if best_points is None:
        # Create a simple good configuration
        # Based on known good distributions for small numbers of points
        points = np.random.rand(n, d)
        best_points = points
    
    return best_points


# EVOLVE-BLOCK-END
