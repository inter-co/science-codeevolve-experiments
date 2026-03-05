# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization with simulated annealing optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0.0
    
    def energy_function(points, alpha=1.0):
        """
        Energy function that encourages good point distribution.
        Penalizes configurations with small minimum distances and large maximum distances.
        """
        if len(points) < 2:
            return float('inf')
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return float('inf')
            
        # We want to maximize d_min/d_max, so we minimize -d_min/d_max
        # Add penalty terms to encourage uniformity
        ratio = d_min / d_max
        return -ratio + alpha * (d_max - d_min) / d_max
    
    def perturb_point(point, step_size=0.01):
        """Perturb a single point with bounded random movement."""
        new_point = point + np.random.uniform(-step_size, step_size, 2)
        # Keep within unit square
        new_point[0] = np.clip(new_point[0], 0, 1)
        new_point[1] = np.clip(new_point[1], 0, 1)
        return new_point
    
    # Initialize points using a hexagonal grid pattern for better starting configuration
    n = 16
    points = np.zeros((n, 2))
    
    # Create a hexagonal-like arrangement
    rows = 4
    cols = 4
    spacing_x = 1.0 / (cols - 1)
    spacing_y = 1.0 / (rows - 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < n:
                # Add slight jitter to avoid perfect grid
                x = j * spacing_x + np.random.normal(0, 0.01)
                y = i * spacing_y + np.random.normal(0, 0.01)
                points[idx] = [np.clip(x, 0, 1), np.clip(y, 0, 1)]
                idx += 1
    
    # Simulated Annealing optimization
    current_points = points.copy()
    current_ratio = compute_min_max_ratio(current_points)
    best_points = current_points.copy()
    best_ratio = current_ratio
    
    # Parameters for simulated annealing
    temp = 0.1
    cooling_rate = 0.9995
    min_temp = 1e-6
    max_iterations = 50000
    
    for iteration in range(max_iterations):
        if temp < min_temp:
            break
            
        # Choose random point to perturb
        point_idx = random.randint(0, n-1)
        old_point = current_points[point_idx].copy()
        
        # Perturb the point
        new_point = perturb_point(old_point)
        current_points[point_idx] = new_point
        
        # Compute new ratio
        new_ratio = compute_min_max_ratio(current_points)
        
        # Accept or reject the move
        if new_ratio > current_ratio:
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = current_points.copy()
        else:
            # Metropolis acceptance criterion
            delta = new_ratio - current_ratio
            if np.random.random() < np.exp(delta / temp):
                current_ratio = new_ratio
            else:
                # Revert the change
                current_points[point_idx] = old_point
        
        # Cool down temperature
        temp *= cooling_rate
    
    return best_points


# EVOLVE-BLOCK-END
