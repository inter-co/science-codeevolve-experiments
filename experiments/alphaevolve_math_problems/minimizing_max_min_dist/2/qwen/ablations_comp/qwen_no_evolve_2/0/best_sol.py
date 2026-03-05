# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel simulated annealing approach with energy-based optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # Initialize points randomly within unit square
    points = np.random.rand(n, d)
    
    # Energy function that penalizes both very small and very large distances
    def energy_function(points):
        # Calculate pairwise distances
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return float('inf')
            
        # Ratio to maximize
        ratio = min_dist / max_dist
        
        # Return negative because we want to maximize (min/max ratio)
        return -ratio
    
    # Simulated Annealing parameters
    temp = 1.0
    cooling_rate = 0.999
    min_temp = 1e-8
    max_iterations = 10000
    
    # Store best solution
    best_points = points.copy()
    best_energy = energy_function(points)
    
    # Main optimization loop
    for iteration in range(max_iterations):
        # Create a new candidate solution by perturbing one point
        candidate_points = points.copy()
        point_idx = np.random.randint(0, n)
        
        # Perturb the selected point with small random displacement
        displacement = np.random.normal(0, 0.01, d)
        candidate_points[point_idx] += displacement
        
        # Keep points within [0,1] bounds
        candidate_points = np.clip(candidate_points, 0, 1)
        
        # Calculate energies
        current_energy = energy_function(points)
        candidate_energy = energy_function(candidate_points)
        
        # Accept or reject based on Metropolis criterion
        if candidate_energy < current_energy:
            # Always accept better solutions
            points = candidate_points
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_points = candidate_points.copy()
        else:
            # Accept worse solutions with probability based on temperature
            delta_energy = candidate_energy - current_energy
            acceptance_prob = math.exp(-delta_energy / temp)
            if np.random.random() < acceptance_prob:
                points = candidate_points
                
        # Cool down temperature
        temp *= cooling_rate
        
        # Stop if temperature gets too low
        if temp < min_temp:
            break
    
    # Final refinement with gradient-based optimization
    # Using a simple gradient descent approach on the energy function
    for _ in range(1000):
        # Calculate current distances
        distances = pdist(best_points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            break
            
        # Simple gradient descent step
        grad = np.zeros_like(best_points)
        eps = 1e-6
        
        for i in range(n):
            for j in range(d):
                # Compute numerical gradient
                test_points = best_points.copy()
                test_points[i, j] += eps
                test_points = np.clip(test_points, 0, 1)
                
                new_energy = - (np.min(pdist(test_points)) / np.max(pdist(test_points)))
                old_energy = - (np.min(pdist(best_points)) / np.max(pdist(best_points)))
                
                grad[i, j] = (new_energy - old_energy) / eps
                
        # Update points
        learning_rate = 0.01
        best_points -= learning_rate * grad
        
        # Ensure points stay within bounds
        best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
