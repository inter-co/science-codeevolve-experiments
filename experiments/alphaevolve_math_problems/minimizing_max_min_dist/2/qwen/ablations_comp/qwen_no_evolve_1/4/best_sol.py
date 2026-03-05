# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def objective(x_flat):
        """Objective function to maximize the min/max distance ratio"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return -np.inf
            
        # Return negative because we're minimizing in scipy
        return -min_dist / max_dist
    
    def constraint_func(x_flat):
        """Constraint function to keep points within unit square"""
        points = x_flat.reshape(-1, 2)
        # Ensure all points are within [0,1] x [0,1]
        return np.concatenate([
            points[:, 0],  # x coordinates
            points[:, 1],  # y coordinates
            1 - points[:, 0],  # 1 - x coordinates
            1 - points[:, 1]   # 1 - y coordinates
        ])
    
    # Initialize points randomly within unit square
    np.random.seed(42)
    initial_points = np.random.rand(16, 2)
    initial_flat = initial_points.flatten()
    
    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(32)]
    
    # Define constraints: all points must be in [0,1] x [0,1]
    constraints = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Use scipy's minimize with SLSQP method
    start_time = time.time()
    
    # Try multiple optimization runs with different initializations
    best_ratio = -np.inf
    best_points = None
    
    for attempt in range(5):
        # Slightly perturb the initial points for different attempts
        np.random.seed(42 + attempt)
        perturbed_initial = initial_points + np.random.normal(0, 0.01, (16, 2))
        # Keep within bounds
        perturbed_initial = np.clip(perturbed_initial, 0, 1)
        perturbed_flat = perturbed_initial.flatten()
        
        try:
            result = minimize(
                objective,
                perturbed_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                distances = pdist(final_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
        except Exception as e:
            continue
            
        # Early termination if we've been running too long
        if time.time() - start_time > 55:  # Leave some buffer for finalization
            break
    
    # If no good solution found, fall back to the original approach but with better initialization
    if best_points is None:
        # Better approach: use a known good configuration and optimize from there
        # Start with a regular grid pattern and perturb slightly
        grid_points = []
        for i in range(4):
            for j in range(4):
                grid_points.append([i/3, j/3])  # 4x4 grid scaled to [0,1]
        
        grid_points = np.array(grid_points)
        # Add some randomness
        np.random.seed(42)
        noisy_points = grid_points + np.random.normal(0, 0.05, (16, 2))
        noisy_points = np.clip(noisy_points, 0, 1)
        
        # Optimize from this starting point
        start_flat = noisy_points.flatten()
        
        try:
            result = minimize(
                objective,
                start_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                best_points = result.x.reshape(-1, 2)
            else:
                best_points = noisy_points
                
        except:
            best_points = noisy_points
    
    return best_points


# EVOLVE-BLOCK-END
