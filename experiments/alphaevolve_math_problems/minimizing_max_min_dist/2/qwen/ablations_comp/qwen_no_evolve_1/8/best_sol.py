# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining hexagonal grid initialization with simulated annealing optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
            
        # Calculate all pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        # Avoid division by zero
        if dmax == 0:
            return 0.0
            
        return dmin / dmax
    
    def objective_function(points):
        """Objective function to maximize (negative because we minimize in scipy)."""
        return -calculate_min_max_ratio(points)
    
    # Initialize points using a hexagonal grid pattern for good starting configuration
    # This provides a more uniform distribution than pure randomness
    n = 16
    
    # Create a hexagonal grid pattern
    # We'll place points in a roughly hexagonal arrangement
    rows = 4
    cols = 4
    points = []
    
    # Hexagonal grid with slight perturbation
    for i in range(rows):
        for j in range(cols):
            if i % 2 == 0:
                x = j * 1.0 + 0.5
                y = i * 0.866  # sqrt(3)/2
            else:
                x = j * 1.0 + 1.0
                y = i * 0.866
            points.append([x, y])
    
    # Normalize to [0,1] x [0,1] and add some randomness
    points = np.array(points[:n])
    
    # Normalize to unit square
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    
    if x_range > 0:
        points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
    if y_range > 0:
        points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
    
    # Add small random perturbations to avoid degenerate cases
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    
    # Clip to [0,1] bounds
    points = np.clip(points, 0, 1)
    
    # Apply simulated annealing optimization
    # Current best solution
    current_points = points.copy()
    current_ratio = calculate_min_max_ratio(current_points)
    
    # Simulated Annealing parameters
    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 1e-8
    iterations_per_temp = 100
    
    best_points = current_points.copy()
    best_ratio = current_ratio
    
    # Optimization loop
    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            # Create neighbor solution by perturbing one point
            new_points = current_points.copy()
            
            # Choose random point to perturb
            idx = random.randint(0, n-1)
            
            # Small random perturbation
            new_points[idx, 0] += np.random.normal(0, 0.001)
            new_points[idx, 1] += np.random.normal(0, 0.001)
            
            # Keep within bounds
            new_points[idx, 0] = np.clip(new_points[idx, 0], 0, 1)
            new_points[idx, 1] = np.clip(new_points[idx, 1], 0, 1)
            
            # Calculate new ratio
            new_ratio = calculate_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.random() < np.exp((new_ratio - current_ratio) / temperature):
                current_points = new_points
                current_ratio = new_ratio
                
                # Update best solution
                if current_ratio > best_ratio:
                    best_points = current_points.copy()
                    best_ratio = current_ratio
        
        temperature *= cooling_rate
    
    return best_points


# EVOLVE-BLOCK-END
