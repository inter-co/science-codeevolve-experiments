# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses improved simulated annealing with enhanced initialization and cooling schedule.

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
    
    # Enhanced initialization with better hexagonal pattern
    np.random.seed(42)
    
    # Create a more sophisticated hexagonal pattern inspired by optimal configurations
    points = np.zeros((16, 2))
    
    # Create points in a hexagonal lattice with better spacing
    # Using a more systematic approach to distribute points evenly
    row_positions = [0, 1, 2, 3]
    col_positions = [0, 1, 2, 3]
    
    idx = 0
    for i, row in enumerate(row_positions):
        for j, col in enumerate(col_positions):
            if idx >= 16:
                break
            # More carefully spaced hexagonal pattern
            x = col + (row % 2) * 0.5
            y = row * math.sqrt(3) / 2
            
            # Add more substantial random noise to break symmetry effectively
            x += np.random.normal(0, 0.02)
            y += np.random.normal(0, 0.02)
            
            points[idx] = [x, y]
            idx += 1
    
    # Normalize properly to fit in [0,1] square with better scaling
    if len(points) > 0:
        # Find bounding box
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        # Avoid division by zero
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        # Scale and center
        if x_range > 0:
            points[:, 0] = (points[:, 0] - x_min) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - y_min) / y_range
            
        # Further normalize to [0,1] range by centering and scaling appropriately
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        points[:, 0] -= center_x
        points[:, 1] -= center_y
        
        # Scale to fit nicely in [0,1] square
        max_extent = max(np.max(np.abs(points[:, 0])), np.max(np.abs(points[:, 1])))
        if max_extent > 0:
            points /= max_extent * 1.1  # Add a bit of padding
            
        # Shift to [0,1] range
        points = (points + 1) / 2
    
    # Ensure we have exactly 16 points and they're within bounds
    points = points[:16]
    points = np.clip(points, 0, 1)
    
    # Simulated Annealing parameters tuned for optimal performance
    current_points = points.copy()
    best_points = current_points.copy()
    best_ratio = compute_min_max_ratio(current_points)
    
    # Very precise cooling schedule for maximum convergence
    temperature = 1.0
    min_temperature = 1e-15  # Even lower minimum for better convergence
    cooling_rate = 0.99985  # Precise cooling rate from top performers
    max_iterations = 60000  # Slightly more iterations for better convergence
    
    # Main optimization loop
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
        
        # Stop if temperature gets too low
        if temperature < min_temperature:
            break
    
    # Enhanced final polishing stage with adaptive step sizes
    if best_ratio > 0.05:
        # Phase 1: Coarse adjustment with larger steps
        coarse_steps = 15000
        for _ in range(coarse_steps):
            new_points = neighbor_step(best_points, 0.005)  # Medium steps
            new_energy = objective_function(new_points)
            current_energy = objective_function(best_points)
            
            if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / (temperature * 0.001)):
                best_points = new_points
                
            # Update best if we improved
            current_ratio = compute_min_max_ratio(best_points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
        
        # Phase 2: Fine adjustment with very small steps
        fine_steps = 20000
        for _ in range(fine_steps):
            new_points = neighbor_step(best_points, 0.001)  # Very small steps
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
