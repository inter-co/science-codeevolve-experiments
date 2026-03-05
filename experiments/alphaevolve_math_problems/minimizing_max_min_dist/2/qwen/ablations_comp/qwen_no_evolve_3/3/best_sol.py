# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses energy-based optimization with simulated annealing approach.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # Initialize points randomly within unit square
    points = np.random.rand(n, d)
    
    # Energy-based optimization with simulated annealing
    # Objective: maximize min_distance / max_distance ratio
    
    def compute_distances(points):
        """Compute all pairwise distances"""
        distances = pdist(points)
        return distances
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance"""
        distances = compute_distances(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0
    
    def energy_function(points):
        """Energy function that penalizes small distances and large distances"""
        distances = compute_distances(points)
        if len(distances) == 0:
            return float('inf')
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # We want to maximize min_dist/max_dist, so we minimize the negative of this
        # But also penalize when distances are too small (clustering) or too large (spread out)
        if max_dist <= 0:
            return float('inf')
        
        ratio = min_dist / max_dist
        # Return negative ratio since we want to maximize it
        return -ratio
    
    # Simulated Annealing parameters
    temp = 1.0
    min_temp = 1e-8
    cooling_rate = 0.999
    max_iter = 10000
    
    best_points = points.copy()
    best_ratio = compute_min_max_ratio(points)
    
    # Optimization loop
    for iteration in range(max_iter):
        # Create neighbor solution by perturbing one point
        new_points = points.copy()
        idx = np.random.randint(0, n)
        # Small random perturbation
        new_points[idx] += np.random.normal(0, 0.01, d)
        # Keep within bounds [0,1]
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        
        # Accept or reject based on Metropolis criterion
        current_ratio = compute_min_max_ratio(points)
        new_ratio = compute_min_max_ratio(new_points)
        
        # Accept if better, or with probability based on temperature
        if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
            points = new_points
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
        
        # Cool down
        temp *= cooling_rate
        if temp < min_temp:
            temp = min_temp
    
    # Final refinement with gradient-based approach for the best solution
    # Using a simple gradient ascent method
    for _ in range(1000):
        current_ratio = compute_min_max_ratio(best_points)
        
        # Compute gradients numerically
        epsilon = 1e-6
        gradients = np.zeros_like(best_points)
        
        for i in range(n):
            for j in range(d):
                # Perturb point i,j slightly
                test_points_plus = best_points.copy()
                test_points_minus = best_points.copy()
                test_points_plus[i, j] += epsilon
                test_points_minus[i, j] -= epsilon
                
                # Ensure bounds
                test_points_plus[i, j] = np.clip(test_points_plus[i, j], 0, 1)
                test_points_minus[i, j] = np.clip(test_points_minus[i, j], 0, 1)
                
                ratio_plus = compute_min_max_ratio(test_points_plus)
                ratio_minus = compute_min_max_ratio(test_points_minus)
                
                gradients[i, j] = (ratio_plus - ratio_minus) / (2 * epsilon)
        
        # Update points using gradient ascent
        learning_rate = 0.01
        best_points += learning_rate * gradients
        
        # Keep within bounds
        best_points = np.clip(best_points, 0, 1)
        
        # Early stopping if improvement is minimal
        new_ratio = compute_min_max_ratio(best_points)
        if abs(new_ratio - current_ratio) < 1e-10:
            break
    
    return best_points


# EVOLVE-BLOCK-END
