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
    
    np.random.seed(42)
    n = 16
    
    # Initialize points in a structured way - hexagonal grid pattern
    # This provides a good starting configuration
    points = np.zeros((n, 2))
    
    # Create a hexagonal grid pattern that fits well in [0,1] x [0,1]
    rows = 4
    cols = 4
    row_spacing = 1.0 / (rows - 1) if rows > 1 else 1.0
    col_spacing = 1.0 / (cols - 1) if cols > 1 else 1.0
    
    # Hexagonal offset pattern
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < n:
                x = j * col_spacing
                y = i * row_spacing
                # Add slight offset for hexagonal arrangement
                if i % 2 == 1:
                    x += col_spacing * 0.5
                points[i * cols + j] = [x, y]
    
    # Ensure all points are within bounds
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    points[:, 1] = np.clip(points[:, 1], 0, 1)
    
    # Simulated Annealing optimization
    def calculate_min_max_ratio(pts):
        """Calculate the ratio of minimum to maximum distances"""
        if len(pts) < 2:
            return 0
        distances = pdist(pts)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0
    
    # Initial ratio
    current_ratio = calculate_min_max_ratio(points)
    
    # Simulated Annealing parameters
    temperature = 1.0
    cooling_rate = 0.999
    min_temperature = 1e-8
    max_iterations = 10000
    
    best_points = points.copy()
    best_ratio = current_ratio
    
    # Optimization loop
    for iteration in range(max_iterations):
        if temperature < min_temperature:
            break
            
        # Create a small perturbation
        new_points = points.copy()
        
        # Select random point to perturb
        idx = random.randint(0, n - 1)
        # Perturb in small random direction
        new_points[idx] += np.random.normal(0, 0.01, 2)
        
        # Keep within bounds
        new_points[idx][0] = np.clip(new_points[idx][0], 0, 1)
        new_points[idx][1] = np.clip(new_points[idx][1], 0, 1)
        
        # Calculate new ratio
        new_ratio = calculate_min_max_ratio(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_ratio > current_ratio:
            points = new_points
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_points = new_points.copy()
                best_ratio = new_ratio
        else:
            # Accept with probability based on temperature
            delta = new_ratio - current_ratio
            if random.random() < np.exp(delta / temperature):
                points = new_points
                current_ratio = new_ratio
        
        # Cool down
        temperature *= cooling_rate
    
    return best_points


# EVOLVE-BLOCK-END
