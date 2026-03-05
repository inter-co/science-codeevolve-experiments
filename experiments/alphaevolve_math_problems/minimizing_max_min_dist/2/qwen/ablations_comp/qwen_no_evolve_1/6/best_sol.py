# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a simulated annealing approach with energy minimization to find optimal point placement.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # Initialize points randomly within [0,1] x [0,1]
    points = np.random.rand(n, d)
    
    # Energy function that we want to minimize
    # We want to maximize min/max distance ratio, so we minimize 1/(min/max) = max/min
    def energy(points):
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if min_dist == 0:
            return float('inf')  # Avoid division by zero
            
        return max_dist / min_dist
    
    # Simulated Annealing parameters
    initial_temp = 1.0
    final_temp = 0.001
    alpha = 0.995
    max_iter = 10000
    
    current_points = points.copy()
    current_energy = energy(current_points)
    best_points = current_points.copy()
    best_energy = current_energy
    
    temp = initial_temp
    
    # Track time to ensure we don't exceed 60 seconds
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start_time > 55:  # Leave some buffer time
            break
            
        # Create neighbor solution by perturbing one point
        new_points = current_points.copy()
        idx = np.random.randint(0, n)
        # Perturb one point slightly
        new_points[idx] += np.random.normal(0, 0.01, d)
        # Keep within bounds [0,1]
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        
        new_energy = energy(new_points)
        
        # Accept or reject the new solution
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
        else:
            # Accept with probability based on temperature
            if np.random.rand() < np.exp(-(new_energy - current_energy) / temp):
                current_points = new_points
                current_energy = new_energy
        
        # Update best solution
        if current_energy < best_energy:
            best_points = current_points.copy()
            best_energy = current_energy
            
        # Cool down
        temp *= alpha
    
    return best_points


# EVOLVE-BLOCK-END
