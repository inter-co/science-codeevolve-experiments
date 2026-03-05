# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import random


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
        
        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio (since we want to maximize)
        if max_dist == 0:
            return float('inf')
        return -min_dist / max_dist
    
    def constraint_func(x_flat):
        """Constraint function to keep points within [0,1] x [0,1]"""
        points = x_flat.reshape(-1, 2)
        # Ensure all points are within bounds
        return np.concatenate([
            points[:, 0],  # x coordinates
            points[:, 1],  # y coordinates
            1 - points[:, 0],  # 1 - x coordinates
            1 - points[:, 1]   # 1 - y coordinates
        ])
    
    def energy_minimization(x_flat):
        """Energy-based approach: minimize repulsive forces between points"""
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Energy is sum of inverse squared distances (repulsive force)
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        energy = np.sum(1.0 / (distances ** 2))
        
        return energy
    
    # Multi-start approach with different initialization strategies
    best_points = None
    best_ratio = -float('inf')
    
    # Strategy 1: Grid-based initialization
    grid_points = []
    n_grid = int(np.ceil(np.sqrt(16)))
    for i in range(n_grid):
        for j in range(n_grid):
            if len(grid_points) < 16:
                grid_points.append([i/(n_grid-1) if n_grid > 1 else 0.5, 
                                  j/(n_grid-1) if n_grid > 1 else 0.5])
    
    # Strategy 2: Random perturbations of grid
    random_seeds = [42, 123, 456, 789, 999]
    
    for seed in random_seeds:
        np.random.seed(seed)
        
        # Try different initializations
        initial_points = np.array(grid_points[:16]) + np.random.normal(0, 0.05, (16, 2))
        
        # Clip to valid range
        initial_points = np.clip(initial_points, 0, 1)
        
        # Optimize using scipy minimize
        try:
            result = minimize(
                objective,
                initial_points.flatten(),
                method='SLSQP',
                constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                # Compute actual ratio
                distances = pdist(final_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
        except Exception:
            continue
    
    # If no good solution found, use simple grid
    if best_points is None:
        # Create a more systematic grid arrangement
        grid_size = 4
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                points.append([(i+0.5)/grid_size, (j+0.5)/grid_size])
        best_points = np.array(points[:16])
    
    return best_points


# EVOLVE-BLOCK-END
