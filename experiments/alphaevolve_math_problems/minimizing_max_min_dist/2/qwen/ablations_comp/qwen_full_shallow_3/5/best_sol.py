# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction and robust optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        distances = distances[distances > 0]  # Remove zero distances
        
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def objective_function(params):
        """Objective function to minimize (negative ratio)."""
        # Reshape parameters into points
        points = params.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        # Return negative because we want to maximize the ratio
        return -ratio
    
    def constraint_bounds(x_flat):
        points = x_flat.reshape(-1, 2)
        # Ensure all points are within [0,1] x [0,1]
        return np.concatenate([
            points[:, 0],           # x-coordinates >= 0
            1 - points[:, 0],       # x-coordinates <= 1
            points[:, 1],           # y-coordinates >= 0
            1 - points[:, 1]        # y-coordinates <= 1
        ])
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Strategy 1: Golden spiral (from inspiration 1) - most reliable starting point
    n = 16
    points = np.zeros((n, 2))
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    for i in range(n):
        angle = 2 * np.pi * i / phi
        radius = np.sqrt(i / (n - 1)) if i < n - 1 else 1.0
        points[i] = [
            0.5 + 0.4 * radius * np.cos(angle),
            0.5 + 0.4 * radius * np.sin(angle)
        ]
    
    # Add structured perturbations to escape local optima
    points += np.random.normal(0, 0.02, points.shape)
    points = np.clip(points, 0, 1)
    
    # Strategy 2: Hexagonal grid (from inspiration 2) - good alternative
    points_hex = []
    sqrt3 = np.sqrt(3)
    for i in range(4):
        for j in range(4):
            offset = 0.5 if i % 2 == 1 else 0.0
            x = j * 0.25 + offset * 0.25 + np.random.normal(0, 0.01)
            y = i * 0.25 * sqrt3 / 2 + np.random.normal(0, 0.01)
            points_hex.append([x, y])
    
    points_hex = np.array(points_hex[:16])
    points_hex = np.clip(points_hex, 0, 1)
    
    # Strategy 3: Grid-based (from inspiration 1) - simple but effective
    points_grid = []
    for i in range(4):
        for j in range(4):
            x = (i + 0.5) / 4.0
            y = (j + 0.5) / 4.0
            points_grid.append([x, y])
    
    points_grid = np.array(points_grid)
    points_grid += np.random.normal(0, 0.02, points_grid.shape)
    points_grid = np.clip(points_grid, 0, 1)
    
    best_points = None
    best_ratio = -float('inf')
    
    # Try optimization with the best initial configurations
    initial_configs = [points, points_hex, points_grid]
    
    # Use only the most robust optimization method for speed and reliability
    method = 'SLSQP'
    options = {'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
    
    for initial_points in initial_configs:
        try:
            x0 = initial_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            cons = {'type': 'ineq', 'fun': constraint_bounds}
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective_function,
                    x0,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options=options
                )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    # Final polishing with L-BFGS-B for potentially better results
    if best_points is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective_function,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(32)],
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                )
            
            if result.success:
                polished_points = result.x.reshape(-1, 2)
                polished_points = np.clip(polished_points, 0, 1)
                ratio = compute_min_max_ratio(polished_points)
                if ratio > best_ratio:
                    best_points = polished_points
                    
        except Exception:
            pass
    
    # Return the best configuration found
    if best_points is None:
        return points  # Fallback to golden spiral
    
    return best_points


# EVOLVE-BLOCK-END
