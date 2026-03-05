# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel simulated annealing approach with adaptive cooling schedule.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # Initialize points randomly within unit square
    points = np.random.rand(n, d)
    
    # Energy function: negative of min/max ratio (we want to maximize ratio, so minimize negative)
    def energy(points):
        distances = pdist(points)
        if len(distances) == 0:
            return float('inf')
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return float('inf')
        return -d_min / d_max
    
    # Calculate initial energy
    current_energy = energy(points)
    
    # Simulated Annealing parameters
    temperature = 1.0
    min_temperature = 1e-8
    cooling_rate = 0.9995
    max_iterations = 100000
    
    # Store best solution found
    best_points = points.copy()
    best_energy = current_energy
    
    # Main optimization loop
    for iteration in range(max_iterations):
        # Cool down temperature
        temperature *= cooling_rate
        
        if temperature < min_temperature:
            break
            
        # Create new candidate solution by perturbing one point
        new_points = points.copy()
        point_idx = random.randint(0, n-1)
        
        # Perturb the selected point
        new_points[point_idx] += np.random.normal(0, 0.01, d)
        
        # Keep points within bounds [0,1]
        new_points[point_idx] = np.clip(new_points[point_idx], 0, 1)
        
        # Calculate new energy
        new_energy = energy(new_points)
        
        # Accept or reject the new solution
        if new_energy < current_energy:
            # Always accept better solutions
            points = new_points
            current_energy = new_energy
        else:
            # Accept worse solutions with probability based on temperature
            delta = new_energy - current_energy
            acceptance_prob = np.exp(-delta / temperature)
            if random.random() < acceptance_prob:
                points = new_points
                current_energy = new_energy
        
        # Update best solution if needed
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()
    
    return best_points


# EVOLVE-BLOCK-END
