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
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def energy_function(points):
        """Energy function that penalizes configurations with poor min/max ratios"""
        ratio = compute_min_max_ratio(points)
        # We want to maximize ratio, so we minimize -ratio (or equivalently maximize ratio)
        # But for optimization purposes, we'll work with the negative ratio
        return -ratio
    
    def get_neighbors(current_points, step_size=0.05):
        """Generate neighbor points by perturbing one point at a time"""
        neighbors = []
        num_points = len(current_points)
        
        for i in range(num_points):
            # Create a copy of current points
            new_points = current_points.copy()
            
            # Perturb one point
            delta_x = random.uniform(-step_size, step_size)
            delta_y = random.uniform(-step_size, step_size)
            
            # Ensure point stays within bounds [0,1]x[0,1]
            new_x = max(0, min(1, current_points[i][0] + delta_x))
            new_y = max(0, min(1, current_points[i][1] + delta_y))
            
            new_points[i] = [new_x, new_y]
            neighbors.append(new_points)
            
        return neighbors
    
    # Initialize with a good geometric configuration (hexagonal-like pattern)
    # Arrange points in a roughly hexagonal pattern
    points = np.zeros((16, 2))
    
    # Create a more structured initial layout
    row_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    col_indices = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    
    # Generate initial points in a hexagonal-like grid pattern
    for i in range(16):
        row = row_indices[i]
        col = col_indices[i]
        
        # Distribute points in a way that mimics hexagonal packing
        x = 0.1 + 0.8 * (col + 0.5 * (row % 2)) / 5.0
        y = 0.1 + 0.8 * row / 6.0
        
        points[i] = [x, y]
    
    # Add some randomness to avoid local minima
    np.random.seed(42)
    points += np.random.normal(0, 0.02, points.shape)
    
    # Clip to bounds
    points = np.clip(points, 0, 1)
    
    # Simulated Annealing optimization
    current_energy = energy_function(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Parameters for simulated annealing
    temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 1e-8
    max_iterations = 50000
    
    for iteration in range(max_iterations):
        # Generate neighbors
        neighbors = get_neighbors(points, step_size=0.02)
        
        # Evaluate neighbors
        neighbor_energies = [energy_function(neighbor) for neighbor in neighbors]
        
        # Find best neighbor
        best_neighbor_idx = np.argmin(neighbor_energies)
        best_neighbor_energy = neighbor_energies[best_neighbor_idx]
        best_neighbor = neighbors[best_neighbor_idx]
        
        # Accept or reject based on Metropolis criterion
        if best_neighbor_energy < current_energy:
            points = best_neighbor
            current_energy = best_neighbor_energy
        else:
            # Accept with probability based on temperature
            delta_energy = best_neighbor_energy - current_energy
            acceptance_prob = np.exp(-delta_energy / temperature)
            if random.random() < acceptance_prob:
                points = best_neighbor
                current_energy = best_neighbor_energy
        
        # Update best solution
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()
        
        # Cool down
        temperature *= cooling_rate
        
        # Early stopping
        if temperature < min_temperature:
            break
    
    return best_points


# EVOLVE-BLOCK-END
