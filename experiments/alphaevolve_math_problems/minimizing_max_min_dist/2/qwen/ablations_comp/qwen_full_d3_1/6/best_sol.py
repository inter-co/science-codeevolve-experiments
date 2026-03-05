# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematically-inspired initialization with enhanced 
    simulated annealing optimization and progressive refinement.

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
    
    # Mathematically-inspired initialization using a precise grid construction
    # This approach draws from the most successful configurations in literature
    np.random.seed(42)
    
    # Create a high-quality initial configuration using a 4x4 grid with precise coordinates
    # These positions have been empirically shown to work well for dispersion problems
    grid_positions = [
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375],
        [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875]
    ]
    
    # Apply varying noise levels to break symmetry without disrupting good structure
    points = np.zeros((16, 2))
    for i, pos in enumerate(grid_positions):
        # Different noise scales for different points to increase diversity
        noise_scale = 0.004 + (i % 5) * 0.001  # Increasing noise with index
        points[i] = [
            np.clip(pos[0] + np.random.normal(0, noise_scale), 0, 1),
            np.clip(pos[1] + np.random.normal(0, noise_scale), 0, 1)
        ]
    
    # Enhanced Simulated Annealing with adaptive cooling and progressive refinement
    current_points = points.copy()
    best_points = current_points.copy()
    best_ratio = compute_min_max_ratio(current_points)
    
    # Advanced cooling schedule that dynamically adapts to convergence
    temperature = 1.0
    min_temperature = 1e-15
    max_iterations = 60000  # More iterations for better convergence
    
    # Track progress to enable adaptive cooling
    last_improvement_iter = 0
    improvement_count = 0
    
    # Main optimization loop with adaptive behavior
    for iteration in range(max_iterations):
        # Dynamic step size that decreases with temperature
        current_step_size = 0.05 * (temperature / 1.0)
        
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
            last_improvement_iter = iteration
            improvement_count += 1
        
        # Adaptive cooling schedule with smarter logic
        if iteration - last_improvement_iter > 800:
            # Aggressive cooling when progress stalls
            temperature *= 0.99995
        elif iteration - last_improvement_iter > 400:
            # Moderate cooling
            temperature *= 0.9999
        else:
            # Normal cooling
            temperature *= 0.99985
            
        # Ensure temperature doesn't go below minimum
        if temperature < min_temperature:
            temperature = min_temperature
    
    # Multi-phase progressive refinement to squeeze out final improvements
    # Phase 1: Coarse refinement with moderate steps
    for _ in range(25000):
        new_points = neighbor_step(best_points, 0.015)
        new_energy = objective_function(new_points)
        current_energy = objective_function(best_points)
        
        if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / (temperature * 0.01)):
            best_points = new_points
            
        # Update best if we improved
        current_ratio = compute_min_max_ratio(best_points)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
    
    # Phase 2: Fine refinement with small steps
    for _ in range(20000):
        new_points = neighbor_step(best_points, 0.003)
        new_energy = objective_function(new_points)
        current_energy = objective_function(best_points)
        
        if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / (temperature * 0.001)):
            best_points = new_points
            
        # Update best if we improved
        current_ratio = compute_min_max_ratio(best_points)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
    
    # Phase 3: Very fine tuning with smallest steps
    for _ in range(10000):
        new_points = neighbor_step(best_points, 0.001)
        new_energy = objective_function(new_points)
        current_energy = objective_function(best_points)
        
        if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / (temperature * 0.0001)):
            best_points = new_points
            
        # Update best if we improved
        current_ratio = compute_min_max_ratio(best_points)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
    
    return best_points


# EVOLVE-BLOCK-END
