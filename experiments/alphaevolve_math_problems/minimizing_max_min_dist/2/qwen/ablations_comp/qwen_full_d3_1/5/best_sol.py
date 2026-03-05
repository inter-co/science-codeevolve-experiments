# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical initialization with advanced optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        # Find min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return 0.0
            
        return d_min / d_max
    
    def objective_function(points):
        """Objective function to maximize (negative because we minimize in optimization)."""
        return -compute_min_max_ratio(points)
    
    def neighbor_step(points, step_size=0.05):
        """Generate a neighboring solution by perturbing one point."""
        new_points = points.copy()
        # Choose a random point to perturb
        idx = random.randint(0, len(points) - 1)
        # Add small random perturbation
        new_points[idx] += np.random.normal(0, step_size, 2)
        # Keep points within [0,1] bounds
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        return new_points
    
    def acceptance_probability(old_energy, new_energy, temperature):
        """Calculate probability of accepting worse solution."""
        if new_energy < old_energy:
            return 1.0
        return math.exp((old_energy - new_energy) / temperature)
    
    # Initialize with a sophisticated mathematical construction
    np.random.seed(42)
    
    # Use a high-quality hexagonal lattice approach similar to successful programs
    # Create points in a hexagonal pattern that naturally provides good dispersion
    points = []
    rows = 4
    cols = 4
    
    for i in range(rows):
        for j in range(cols):
            x = j + (i % 2) * 0.5
            y = i * math.sqrt(3) / 2
            points.append([x, y])
    
    points = np.array(points)
    
    # Normalize to fit in [0,1] square properly
    if len(points) > 0:
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0 and y_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        # Center and map to [0,1] range
        points = points - np.mean(points, axis=0)
        max_extent = np.max(np.abs(points))
        if max_extent > 0:
            points = points / max_extent
        points = (points + 1) / 2
    
    # Ensure exactly 16 points
    points = points[:16]
    
    # Add controlled random perturbations to break symmetry and improve local search
    noise_scale = 0.01
    points += np.random.normal(0, noise_scale, points.shape)
    points = np.clip(points, 0, 1)
    
    # Simulated Annealing parameters with better convergence properties
    current_points = points.copy()
    best_points = current_points.copy()
    best_ratio = compute_min_max_ratio(current_points)
    
    # Advanced cooling schedule inspired by theoretical convergence rates
    temperature = 1.0
    min_temperature = 1e-12
    # Use a more aggressive cooling schedule for faster exploration followed by slower convergence
    cooling_rate = 0.9999  # Slightly faster cooling for better exploration
    max_iterations = 60000  # Maximum iterations allowed
    
    # Optimization loop with adaptive step sizes
    for iteration in range(max_iterations):
        # Adaptive step size based on progress and iteration count
        current_step_size = 0.05 * (1.0 - iteration / max_iterations) + 0.001
        
        # Generate neighbor solution with adaptive step size
        new_points = neighbor_step(current_points, current_step_size)
        
        # Calculate energies
        current_energy = objective_function(current_points)
        new_energy = objective_function(new_points)
        
        # Accept or reject new solution
        if acceptance_probability(current_energy, new_energy, temperature) > random.random():
            current_points = new_points
        
        # Update best solution
        current_ratio = compute_min_max_ratio(current_points)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = current_points.copy()
        
        # Cool down
        temperature *= cooling_rate
        
        # Early stopping if temperature gets too low
        if temperature < min_temperature:
            break
    
    # Multi-stage refinement for better convergence
    # Stage 1: Medium refinement with moderate steps
    if best_ratio > 0.06:
        for _ in range(20000):
            new_points = neighbor_step(best_points, 0.01)  # Medium steps
            new_energy = objective_function(new_points)
            current_energy = objective_function(best_points)
            
            if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / (temperature * 0.01)):
                best_points = new_points
                
            # Update best if we improved
            current_ratio = compute_min_max_ratio(best_points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
    
    # Stage 2: Fine tuning with very small steps
    if best_ratio > 0.06:
        for _ in range(20000):
            new_points = neighbor_step(best_points, 0.002)  # Very small steps
            new_energy = objective_function(new_points)
            current_energy = objective_function(best_points)
            
            if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / (temperature * 0.001)):
                best_points = new_points
                
            # Update best if we improved
            current_ratio = compute_min_max_ratio(best_points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
    
    return best_points


# EVOLVE-BLOCK-END
