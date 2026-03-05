# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization with simulated annealing optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    
    # Phase 1: Geometric initialization using hexagonal packing approximation
    # Arrange points in a roughly hexagonal pattern to get good initial distribution
    points = np.zeros((n, 2))
    
    # Create a grid-like structure with some randomness to avoid regular patterns
    rows = 4
    cols = 4
    row_spacing = 1.0 / (rows + 1)
    col_spacing = 1.0 / (cols + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < n:
                # Add slight jitter to avoid perfect grid
                y = (i + 1) * row_spacing + np.random.normal(0, 0.01)
                x = (j + 1) * col_spacing + np.random.normal(0, 0.01)
                # Ensure points stay within bounds
                points[idx] = [max(0.01, min(0.99, x)), max(0.01, min(0.99, y))]
                idx += 1
    
    # Phase 2: Optimization using simulated annealing with custom objective
    # Energy function based on inverse of min/max distance ratio
    def energy_function(points_array):
        # Compute pairwise distances
        distances = pdist(points_array)
        if len(distances) == 0:
            return float('inf')
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return float('inf')
            
        # We want to maximize min_dist / max_dist, so minimize the negative ratio
        ratio = min_dist / max_dist
        return -ratio
    
    # Simulated annealing parameters
    current_points = points.copy()
    current_energy = energy_function(current_points)
    
    # Annealing schedule
    temperature = 1.0
    cooling_rate = 0.999
    min_temperature = 1e-6
    max_iterations = 50000
    
    # Track best solution
    best_points = current_points.copy()
    best_energy = current_energy
    
    # Optimization iterations
    for iteration in range(max_iterations):
        if temperature < min_temperature:
            break
            
        # Generate neighbor solution by perturbing one point
        neighbor_points = current_points.copy()
        point_idx = np.random.randint(0, n)
        
        # Perturb the selected point
        delta = np.random.normal(0, 0.005, 2)
        neighbor_points[point_idx] += delta
        
        # Keep within bounds
        neighbor_points[point_idx] = np.clip(neighbor_points[point_idx], 0, 1)
        
        # Calculate energy of neighbor
        neighbor_energy = energy_function(neighbor_points)
        
        # Accept or reject based on Metropolis criterion
        if neighbor_energy < current_energy:
            # Always accept better solutions
            current_points = neighbor_points
            current_energy = neighbor_energy
        else:
            # Accept worse solutions with probability based on temperature
            acceptance_prob = math.exp(-(neighbor_energy - current_energy) / temperature)
            if np.random.random() < acceptance_prob:
                current_points = neighbor_points
                current_energy = neighbor_energy
        
        # Update best solution
        if current_energy < best_energy:
            best_points = current_points.copy()
            best_energy = current_energy
            
        # Cool down
        temperature *= cooling_rate
    
    # Final refinement with gradient-based optimization on the best solution
    # Using simple local search with small perturbations
    refined_points = best_points.copy()
    for _ in range(1000):
        best_refined = refined_points.copy()
        best_energy = energy_function(refined_points)
        
        # Try small perturbations to each point
        for i in range(n):
            test_points = refined_points.copy()
            delta = np.random.normal(0, 0.001, 2)
            test_points[i] += delta
            test_points[i] = np.clip(test_points[i], 0, 1)
            
            test_energy = energy_function(test_points)
            if test_energy < best_energy:
                best_energy = test_energy
                best_refined = test_points
        
        refined_points = best_refined
    
    return refined_points


# EVOLVE-BLOCK-END
