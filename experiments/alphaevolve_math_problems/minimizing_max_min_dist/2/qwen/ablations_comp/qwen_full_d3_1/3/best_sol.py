# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses improved simulated annealing optimization approach with better initialization.

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
    
    # Initialize with a better starting configuration - inspired by successful patterns
    np.random.seed(42)
    
    # Create initial configuration using a structured 4x4 grid approach
    points = np.zeros((16, 2))
    
    # Use a grid pattern with small random perturbations to break symmetry
    # This approach tends to work well for this type of optimization
    for i in range(4):
        for j in range(4):
            # Grid positions with slight random variation
            x = i * 0.25 + 0.125 + np.random.normal(0, 0.015)
            y = j * 0.25 + 0.125 + np.random.normal(0, 0.015)
            points[i*4 + j] = [np.clip(x, 0, 1), np.clip(y, 0, 1)]
    
    # Simulated Annealing parameters - optimized for high-quality results
    current_points = points.copy()
    best_points = current_points.copy()
    best_ratio = compute_min_max_ratio(current_points)
    
    # Following the successful approach from inspiration programs with tuned parameters
    temperature = 1.0
    min_temperature = 1e-12
    cooling_rate = 0.99985  # Slightly faster cooling for more thorough exploration
    max_iterations = 50000  # More iterations to allow for better convergence
    
    # Optimization loop with enhanced termination conditions
    for iteration in range(max_iterations):
        # Generate neighbor solution
        new_points = neighbor_step(current_points, 0.05)
        
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
        
        # Stop if temperature gets too low or we've converged
        if temperature < min_temperature:
            break
    
    # Final refinement phase for high-quality solutions
    if best_ratio > 0.05:
        # Do additional fine-tuning with very small steps
        fine_tune_steps = 20000
        for _ in range(fine_tune_steps):
            new_points = neighbor_step(best_points, 0.002)  # Very small steps for fine-tuning
            new_energy = objective_function(new_points)
            current_energy = objective_function(best_points)
            
            # Accept even worse solutions with very low probability for final polish
            if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / (temperature * 0.001)):
                best_points = new_points
                
            # Update best if we improved
            current_ratio = compute_min_max_ratio(best_points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
    
    return best_points


# EVOLVE-BLOCK-END
