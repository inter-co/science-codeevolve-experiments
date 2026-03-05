# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a physics-inspired optimization approach with simulated annealing.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # Initialize points randomly within unit square
    points = np.random.rand(n, d)
    
    # Physics-inspired optimization with cooling schedule
    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 1e-6
    iterations_per_temp = 100
    
    best_points = points.copy()
    best_ratio = 0
    
    # Helper function to calculate min/max distance ratio
    def calculate_ratio(points):
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0
    
    # Helper function to apply boundary constraints
    def constrain_points(points):
        points = np.clip(points, 0, 1)
        return points
    
    # Optimization loop
    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            # Make a small perturbation
            new_points = points.copy()
            idx = np.random.randint(0, n)
            # Perturb one point slightly
            new_points[idx] += np.random.normal(0, 0.01, d)
            
            # Constrain points to unit square
            new_points = constrain_points(new_points)
            
            # Calculate ratios
            current_ratio = calculate_ratio(points)
            new_ratio = calculate_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < math.exp((new_ratio - current_ratio) / temperature):
                points = new_points
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()
        
        temperature *= cooling_rate
    
    return best_points


# EVOLVE-BLOCK-END
