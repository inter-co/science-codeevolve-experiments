# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a simulated annealing approach with energy minimization to find optimal configuration.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distances between all pairs."""
        if len(points) < 2:
            return 0
        
        # Calculate pairwise distances
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0
            
        return min_dist / max_dist
    
    def energy_function(points):
        """Energy function that penalizes small distances and rewards large ones.
        We want to maximize min/max ratio, so we minimize a negative version of this."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return float('inf')
            
        # We want to maximize min/max ratio, so we minimize -ratio
        ratio = min_dist / max_dist
        return -ratio
    
    def get_neighbors(current_points, step_size=0.05):
        """Generate neighbor points by perturbing one point at a time."""
        neighbors = []
        n_points = len(current_points)
        
        for i in range(n_points):
            # Create a copy of current points
            new_points = current_points.copy()
            
            # Perturb one point in both x and y directions
            for _ in range(2):  # Try two perturbations per point
                dx = random.uniform(-step_size, step_size)
                dy = random.uniform(-step_size, step_size)
                
                # Apply perturbation
                new_points[i, 0] += dx
                new_points[i, 1] += dy
                
                # Keep within bounds [0,1]
                new_points[i, 0] = max(0, min(1, new_points[i, 0]))
                new_points[i, 1] = max(0, min(1, new_points[i, 1]))
                
                neighbors.append(new_points.copy())
                
        return neighbors
    
    # Initialize with a good starting configuration
    # Start with a hexagonal-like pattern that's known to be effective
    np.random.seed(42)
    
    # Create initial configuration - start with a regular grid pattern with some noise
    points = np.zeros((16, 2))
    
    # Arrange points in a roughly hexagonal pattern
    row = 0
    col = 0
    for i in range(16):
        # Distribute points in a way that mimics hexagonal packing
        if i == 0:
            points[i] = [0.5, 0.5]  # Center point
        else:
            # Place points in a pattern that tries to maximize spacing
            angle = 2 * math.pi * (i - 1) / 15  # Distribute around circle
            radius = 0.4  # Keep points within reasonable bounds
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points[i] = [x, y]
    
    # Add slight randomization to avoid local minima
    for i in range(16):
        points[i, 0] += random.uniform(-0.05, 0.05)
        points[i, 1] += random.uniform(-0.05, 0.05)
    
    # Keep points within bounds
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    points[:, 1] = np.clip(points[:, 1], 0, 1)
    
    # Simulated Annealing parameters
    current_points = points.copy()
    best_points = current_points.copy()
    current_energy = energy_function(current_points)
    best_energy = current_energy
    
    # Annealing schedule
    temperature = 1.0
    cooling_rate = 0.999
    min_temperature = 1e-6
    max_iterations = 10000
    
    # Optimization loop
    for iteration in range(max_iterations):
        # Generate neighbors
        neighbors = get_neighbors(current_points, step_size=temperature * 0.1)
        
        if len(neighbors) == 0:
            break
            
        # Evaluate neighbors
        neighbor_energies = [energy_function(neighbor) for neighbor in neighbors]
        
        # Find best neighbor
        best_neighbor_idx = np.argmin(neighbor_energies)
        best_neighbor_energy = neighbor_energies[best_neighbor_idx]
        best_neighbor = neighbors[best_neighbor_idx]
        
        # Accept or reject based on Metropolis criterion
        if best_neighbor_energy < current_energy:
            # Always accept better solutions
            current_points = best_neighbor
            current_energy = best_neighbor_energy
        else:
            # Accept worse solutions with probability based on temperature
            delta_energy = best_neighbor_energy - current_energy
            acceptance_probability = math.exp(-delta_energy / temperature)
            if random.random() < acceptance_probability:
                current_points = best_neighbor
                current_energy = best_neighbor_energy
        
        # Update best solution
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = current_points.copy()
        
        # Cool down
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
            
        # Early stopping condition
        if iteration > 1000 and abs(best_energy - current_energy) < 1e-8:
            break
    
    # Final validation
    final_ratio = calculate_min_max_ratio(best_points)
    return best_points


# EVOLVE-BLOCK-END
