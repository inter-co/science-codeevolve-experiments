# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid optimization approach combining geometric insights with simulated annealing.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # Initialize points using a hexagonal grid pattern for good starting configuration
    # This provides a reasonable initial distribution that's likely to be better than random
    points = np.zeros((n, d))
    
    # Create a hexagonal lattice pattern
    rows = 4
    cols = 4
    spacing = 1.0
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < n:
                # Offset every other row for hexagonal packing
                x = j * spacing + (i % 2) * spacing * 0.5
                y = i * spacing * math.sqrt(3) / 2
                points[idx] = [x, y]
                idx += 1
    
    # Normalize to unit square [0,1] x [0,1]
    points[:, 0] = points[:, 0] / max(1e-10, np.max(points[:, 0]))
    points[:, 1] = points[:, 1] / max(1e-10, np.max(points[:, 1]))
    
    # Apply simulated annealing optimization
    best_points = points.copy()
    best_ratio = calculate_min_max_ratio(best_points)
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 1e-8
    alpha = 0.999
    max_iter = 50000
    
    for iteration in range(max_iter):
        # Generate neighbor solution by perturbing one point
        new_points = best_points.copy()
        point_idx = np.random.randint(0, n)
        
        # Perturb the selected point
        new_points[point_idx] += np.random.normal(0, 0.01, 2)
        
        # Keep points within bounds [0,1] x [0,1]
        new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
        new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)
        
        # Calculate new ratio
        new_ratio = calculate_min_max_ratio(new_points)
        
        # Accept or reject the new solution
        if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) / T):
            best_points = new_points
            best_ratio = new_ratio
            
            # Occasionally reset temperature to escape local minima
            if iteration % 1000 == 0 and T > T_min:
                T *= alpha
                
        # Cool down temperature
        if T > T_min:
            T *= alpha
    
    return best_points


def calculate_min_max_ratio(points: np.ndarray) -> float:
    """
    Calculate the ratio of minimum to maximum distance between all point pairs.
    
    Args:
        points: np.ndarray of shape (n, 2) containing 2D coordinates
        
    Returns:
        float: ratio of minimum to maximum distance
    """
    if len(points) < 2:
        return 0.0
    
    # Calculate pairwise distances
    distances = pdist(points)
    
    # Get min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    # Avoid division by zero
    if d_max <= 1e-10:
        return 0.0
        
    return d_min / d_max


# EVOLVE-BLOCK-END
