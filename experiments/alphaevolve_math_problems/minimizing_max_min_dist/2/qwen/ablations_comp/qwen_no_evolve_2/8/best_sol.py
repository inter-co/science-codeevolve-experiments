# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(points_flat):
        """Objective function to maximize min/max distance ratio"""
        points = points_flat.reshape(-1, 2)
        # Compute pairwise distances
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio since we want to maximize
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    def constraint_func(points_flat):
        """Ensure points stay within [0,1] x [0,1]"""
        points = points_flat.reshape(-1, 2)
        # Return positive values when constraints are satisfied
        return np.concatenate([
            points[:, 0],                    # x >= 0
            1 - points[:, 0],                # x <= 1
            points[:, 1],                    # y >= 0
            1 - points[:, 1]                 # y <= 1
        ])
    
    # Start with a good initial configuration based on hexagonal packing
    # Arrange points in a roughly hexagonal pattern
    n = 16
    points_init = []
    
    # Create a hexagonal-like arrangement
    rows = 4
    cols = 4
    spacing_x = 1.0 / (cols - 1)
    spacing_y = 1.0 / (rows - 1)
    
    for i in range(rows):
        for j in range(cols):
            if len(points_init) < n:
                x = j * spacing_x + (i % 2) * spacing_x * 0.5  # Offset every other row
                y = i * spacing_y
                points_init.append([max(0.01, min(0.99, x)), max(0.01, min(0.99, y))])
    
    # Use multiple random starts to find better solutions
    best_ratio = -float('inf')
    best_points = None
    
    # Try several different initial configurations
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        
        # Start with hexagonal pattern
        points_current = np.array(points_init[:n])
        
        # Add some noise to break symmetry
        noise = np.random.normal(0, 0.01, points_current.shape)
        points_current += noise
        points_current = np.clip(points_current, 0, 1)
        
        # Optimize using scipy
        try:
            # Flatten points for optimization
            points_flat = points_current.flatten()
            
            # Define bounds (points must stay in [0,1] x [0,1])
            bounds = [(0, 1) for _ in range(2*n)]
            
            # Define constraints
            cons = {'type': 'ineq', 'fun': constraint_func}
            
            # Optimize
            result = minimize(
                objective,
                points_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                distances = pdist(optimized_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
        except Exception:
            continue
    
    # If no good solution found, return the hexagonal pattern
    if best_points is None:
        best_points = np.array(points_init[:n])
    
    return best_points


# EVOLVE-BLOCK-END
