# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a physics-based optimization approach with energy minimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    
    # Initialize points randomly within unit square
    n = 16
    points = np.random.rand(n, 2)
    
    # Physics-based optimization parameters
    max_iterations = 10000
    learning_rate = 0.01
    temperature = 1.0
    cooling_rate = 0.9995
    
    best_ratio = 0
    best_points = points.copy()
    
    # Energy function: minimize negative of min/max ratio (equivalent to maximizing ratio)
    def compute_ratio(points):
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    # Gradient-based optimization
    for iteration in range(max_iterations):
        # Compute current ratio
        current_ratio = compute_ratio(points)
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()
        
        # Compute gradients using finite differences
        gradients = np.zeros_like(points)
        epsilon = 1e-6
        
        for i in range(n):
            for j in range(2):  # x and y coordinates
                # Perturb point
                points_plus = points.copy()
                points_minus = points.copy()
                points_plus[i, j] += epsilon
                points_minus[i, j] -= epsilon
                
                # Compute gradient estimate
                ratio_plus = compute_ratio(points_plus)
                ratio_minus = compute_ratio(points_minus)
                gradients[i, j] = (ratio_plus - ratio_minus) / (2 * epsilon)
        
        # Update points with gradient descent
        points += learning_rate * gradients
        
        # Keep points within bounds [0,1]
        points = np.clip(points, 0, 1)
        
        # Simulated annealing cooling
        temperature *= cooling_rate
        
        # Occasionally do random perturbations to escape local optima
        if iteration % 100 == 0 and iteration > 0:
            points += np.random.normal(0, 0.001, (n, 2))
            points = np.clip(points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
