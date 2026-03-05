# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses improved simulated annealing with enhanced initialization and refinement strategies.

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
    
    # Initialize with precise grid-based configuration from INSPIRATION PROGRAM 1
    # This provides a much better starting point than generic approaches
    np.random.seed(42)
    
    # Create a structured grid with precise positions (like INSPIRATION PROGRAM 1)
    # Using fixed positions that have proven effective in practice
    grid_positions = [
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375],
        [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875]
    ]
    
    points = np.array(grid_positions)
    
    # Add carefully controlled random perturbations to break exact symmetry
    # Different noise levels per point to increase diversity (as in INSPIRATION PROGRAM 1)
    for i in range(len(points)):
        noise_scale = 0.003 + (i % 3) * 0.001  # Varying noise levels
        points[i] += np.random.normal(0, noise_scale, 2)
    
    points = np.clip(points, 0, 1)
    
    # Simulated Annealing parameters tuned for better performance (like INSPIRATION PROGRAM 1)
    current_points = points.copy()
    best_points = current_points.copy()
    best_ratio = compute_min_max_ratio(current_points)
    
    # Enhanced cooling schedule with better convergence characteristics (from INSPIRATION PROGRAM 1)
    temperature = 1.0
    min_temperature = 1e-12
    # Slightly faster cooling rate but with better convergence properties
    cooling_rate = 0.9998  # From INSPIRATION PROGRAM 2
    max_iterations = 60000  # Increased for better convergence
    
    # Optimization loop with enhanced refinement strategies (like INSPIRATION PROGRAM 1)
    for iteration in range(max_iterations):
        # Generate neighbor solution with adaptive step size
        # Use larger steps early for exploration, smaller later for exploitation
        step_size = 0.05 * (1.0 - iteration / max_iterations) + 0.001
        new_points = neighbor_step(current_points, step_size)
        
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
        
        # Early stopping for efficiency
        if temperature < min_temperature:
            break
    
    # Multi-stage refinement with improved strategies (like INSPIRATION PROGRAM 1)
    # Phase 1: Coarse refinement with medium steps
    if best_ratio > 0.06:
        for _ in range(20000):
            new_points = neighbor_step(best_points, 0.01)  # Medium steps
            new_energy = objective_function(new_points)
            current_energy = objective_function(best_points)
            
            # Use more aggressive acceptance criterion for this phase
            if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / (temperature * 0.01)):
                best_points = new_points
                
            # Update best if we improved
            current_ratio = compute_min_max_ratio(best_points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
    
    # Phase 2: Fine tuning with very small steps
    if best_ratio > 0.06:
        for _ in range(15000):  # Fewer iterations to save time but still get good results
            new_points = neighbor_step(best_points, 0.002)  # Very small steps
            new_energy = objective_function(new_points)
            current_energy = objective_function(best_points)
            
            # Even more selective acceptance for fine tuning
            if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / (temperature * 0.001)):
                best_points = new_points
                
            # Update best if we improved
            current_ratio = compute_min_max_ratio(best_points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
    
    return best_points


# EVOLVE-BLOCK-END
