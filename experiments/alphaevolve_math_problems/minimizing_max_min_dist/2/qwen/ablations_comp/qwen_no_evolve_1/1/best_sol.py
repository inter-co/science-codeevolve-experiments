# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses simulated annealing with custom energy function for optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # Initialize points randomly within unit square
    points = np.random.rand(n, d)
    
    # Simulated Annealing parameters
    initial_temp = 1.0
    final_temp = 1e-6
    alpha = 0.995
    max_iter = 10000
    
    # Current best solution
    best_points = points.copy()
    best_ratio = calculate_min_max_ratio(points)
    
    temp = initial_temp
    
    for iteration in range(max_iter):
        # Create neighbor solution by perturbing one point
        new_points = points.copy()
        point_idx = np.random.randint(0, n)
        
        # Perturb the selected point
        new_points[point_idx] += np.random.normal(0, 0.01, d)
        
        # Keep points within unit square
        new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
        new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)
        
        # Calculate new ratio
        new_ratio = calculate_min_max_ratio(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_ratio > best_ratio or np.random.rand() < math.exp((new_ratio - best_ratio) / temp):
            points = new_points
            if new_ratio > best_ratio:
                best_points = new_points.copy()
                best_ratio = new_ratio
        
        # Cool down temperature
        temp = max(final_temp, temp * alpha)
    
    return best_points


def calculate_min_max_ratio(points):
    """
    Calculate the ratio of minimum to maximum distance between all point pairs.
    """
    if len(points) < 2:
        return 0
    
    # Calculate pairwise distances
    distances = pdist(points)
    
    # Get min and max distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Avoid division by zero
    if max_dist == 0:
        return 0
    
    return min_dist / max_dist


# EVOLVE-BLOCK-END
