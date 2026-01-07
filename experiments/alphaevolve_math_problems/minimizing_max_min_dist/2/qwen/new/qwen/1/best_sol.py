# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses simulated annealing with geometric initialization and multiple restarts for robust optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    n = 16
    
    def calculate_ratio(points):
        """Calculate the ratio of minimum to maximum distance"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return d_min / d_max
    
    def neighbor_step(points, step_size=0.05):
        """Generate a neighboring solution by perturbing one point"""
        new_points = points.copy()
        # Choose random point to perturb
        idx = random.randint(0, n-1)
        # Add small random perturbation
        new_points[idx] += np.random.normal(0, step_size, 2)
        # Keep within bounds [0,1] x [0,1]
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        return new_points
    
    def improved_initialization():
        """Create a better starting configuration using circular pattern"""
        # Start with points arranged in a circle pattern
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 0.4
        center = np.array([0.5, 0.5])
        
        # Create initial configuration in circular pattern
        points = np.zeros((n, 2))
        for i in range(n):
            points[i] = center + radius * np.array([np.cos(angles[i]), np.sin(angles[i])])
        
        # Add some randomness to break symmetry
        points += np.random.normal(0, 0.05, (n, 2))
        points = np.clip(points, 0, 1)
        
        return points
    
    def simulated_annealing(initial_points=None, max_iter=50000):
        """Run simulated annealing optimization"""
        
        # Initialize with improved configuration if not provided
        if initial_points is None:
            points = improved_initialization()
        else:
            points = initial_points.copy()
        
        current_points = points.copy()
        current_ratio = calculate_ratio(current_points)
        
        # Optimized annealing parameters - inspired by successful approaches
        temperature = 1.0
        min_temperature = 1e-15
        cooling_rate = 0.9992  # Carefully tuned cooling rate
        max_iterations = max_iter
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Main annealing loop
        for iteration in range(max_iterations):
            # Generate neighbor with adaptive step size
            step_size = max(0.005, 0.1 * (temperature / 1.0))  # Decreasing step size
            new_points = neighbor_step(current_points, step_size)
            new_ratio = calculate_ratio(new_points)
            
            # Accept or reject the move
            delta = new_ratio - current_ratio
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_points = new_points
                current_ratio = new_ratio
                
                # Update best solution
                if current_ratio > best_ratio:
                    best_points = current_points.copy()
                    best_ratio = current_ratio
            
            # Cool down
            temperature *= cooling_rate
            
            # Early stopping condition
            if temperature < min_temperature:
                break
                
        return best_points, best_ratio
    
    # Run multiple optimizations with different random seeds to ensure quality
    best_points = None
    best_ratio = -np.inf
    
    # Try many more random restarts to improve chances of finding better solution
    # Using a larger set of seeds for more thorough exploration
    seeds = []
    for i in range(100):  # 100 seeds instead of 20 for better exploration
        seeds.append(i * 100 + 42)
    
    # Also try some specific seeds that might work well
    additional_seeds = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    seeds.extend(additional_seeds)
    
    for seed in seeds:
        np.random.seed(seed)
        points, ratio = simulated_annealing(max_iter=60000)  # Increased iterations for better convergence
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # Final refinement with even more aggressive optimization
    if best_points is not None:
        # Do a final intensive refinement with even more iterations
        np.random.seed(99999)
        # Increase iterations significantly to get better convergence
        for _ in range(30000):  # 30000 iterations for final refinement
            refined_points = neighbor_step(best_points, 0.001)  # Even smaller step size
            refined_ratio = calculate_ratio(refined_points)
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points
    
    return best_points if best_points is not None else np.random.rand(n, 2)


# EVOLVE-BLOCK-END
