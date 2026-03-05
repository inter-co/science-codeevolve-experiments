# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a simulated annealing approach with custom energy function targeting min/max distance ratio.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    n = 16
    d = 2
    
    # Initialize points randomly within unit square
    points = np.random.rand(n, d)
    
    # Simulated Annealing parameters
    max_iter = 50000
    temp = 1.0
    cooling_rate = 0.9995
    min_temp = 1e-8
    
    # Track best solution
    best_points = points.copy()
    best_ratio = 0
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0
            
        return min_dist / max_dist
    
    def apply_move(points, step_size=0.01):
        """Apply a small random perturbation to one point"""
        new_points = points.copy()
        idx = random.randint(0, len(new_points) - 1)
        
        # Generate small random displacement
        displacement = np.random.normal(0, step_size, d)
        new_points[idx] += displacement
        
        # Keep points within bounds [0,1]
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        
        return new_points
    
    # Main optimization loop
    for iteration in range(max_iter):
        # Apply move
        new_points = apply_move(points)
        
        # Calculate ratios
        old_ratio = compute_min_max_ratio(points)
        new_ratio = compute_min_max_ratio(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_ratio > old_ratio:
            points = new_points
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
        else:
            # Accept with probability based on temperature
            delta = new_ratio - old_ratio
            if random.random() < np.exp(delta / temp):
                points = new_points
        
        # Cool down
        temp *= cooling_rate
        
        # Early stopping condition
        if temp < min_temp:
            break
    
    return best_points


# EVOLVE-BLOCK-END
