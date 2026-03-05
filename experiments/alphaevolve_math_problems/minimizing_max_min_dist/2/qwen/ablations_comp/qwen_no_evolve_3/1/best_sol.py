# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a simulated annealing approach with energy minimization to find optimal configuration.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # Initialize points randomly within [0,1] x [0,1]
    points = np.random.rand(n, d)
    
    # Energy function: minimize -log(min_distance/max_distance) 
    # This is equivalent to maximizing min_distance/max_distance
    def compute_energy(points):
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return float('inf')
            
        # Use log-transformed ratio to avoid numerical issues
        ratio = min_dist / max_dist
        if ratio <= 0:
            return float('inf')
            
        # Return negative log to convert maximization to minimization
        return -np.log(ratio)
    
    # Gradient-based optimization helper
    def gradient_descent_step(points, learning_rate=0.01, max_attempts=100):
        old_energy = compute_energy(points)
        
        # Try small perturbations
        for _ in range(max_attempts):
            # Create a copy and perturb one point at a time
            new_points = points.copy()
            
            # Randomly select a point to move
            idx = np.random.randint(0, n)
            
            # Small random perturbation
            perturbation = np.random.normal(0, 0.001, d)
            new_points[idx] += perturbation
            
            # Keep within bounds [0,1]
            new_points[idx] = np.clip(new_points[idx], 0, 1)
            
            new_energy = compute_energy(new_points)
            
            if new_energy < old_energy:
                points = new_points
                old_energy = new_energy
                
        return points
    
    # Simulated Annealing approach
    current_points = points.copy()
    best_points = current_points.copy()
    best_energy = compute_energy(current_points)
    
    # Annealing parameters
    temp = 1.0
    cooling_rate = 0.999
    min_temp = 1e-6
    max_iterations = 50000
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > 55:  # Leave 5 seconds for final cleanup
            break
            
        # Generate neighbor solution
        new_points = current_points.copy()
        
        # Perturb one point
        idx = np.random.randint(0, n)
        new_points[idx] += np.random.normal(0, 0.01, d)
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        
        # Calculate energies
        current_energy = compute_energy(current_points)
        new_energy = compute_energy(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_energy < current_energy:
            current_points = new_points
            if new_energy < best_energy:
                best_points = new_points.copy()
                best_energy = new_energy
        else:
            # Accept with probability based on temperature
            if np.random.rand() < np.exp(-(new_energy - current_energy) / temp):
                current_points = new_points
        
        # Cool down
        temp *= cooling_rate
        if temp < min_temp:
            temp = min_temp
    
    # Final refinement using gradient descent
    refined_points = best_points.copy()
    for _ in range(1000):
        refined_points = gradient_descent_step(refined_points)
    
    return refined_points


# EVOLVE-BLOCK-END
