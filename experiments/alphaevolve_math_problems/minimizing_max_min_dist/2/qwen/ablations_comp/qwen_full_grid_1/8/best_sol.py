# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and multi-start optimization 
    with emphasis on the most effective strategies from previous experiments.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def objective(points_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = points_flat.reshape(-1, 2)
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    def fibonacci_sphere_points(n: int) -> np.ndarray:
        """
        Generate points using Fibonacci sphere distribution for better uniformity.
        Then project to 2D square [0,1] x [0,1].
        """
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        
        for i in range(n):
            # Distribute points more evenly along the height
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            # Use golden angle with small perturbation for better distribution
            theta = phi * i + (i * 0.1)  # Add small perturbation to avoid regular patterns
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            # Map to 2D square [0,1] x [0,1] - this preserves good distribution properties
            points.append([(x + 1) / 2, (z + 1) / 2])
        
        return np.array(points)
    
    def initialize_hexagonal_lattice():
        """Initialize points in a hexagonal lattice pattern"""
        # Create a 4x4 grid with hexagonal offset for better distribution
        points = []
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Offset every other row for better hexagonal packing
                x = j + 0.5 * (i % 2)
                y = i * math.sqrt(3) / 2
                points.append([x, y])
        
        # Convert to numpy array and normalize
        points = np.array(points[:16])
        
        if len(points) > 0:
            # Normalize to [0,1] range
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0 and y_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
                
                # Scale and center appropriately to avoid boundary effects
                points[:, 0] = points[:, 0] * 0.8 + 0.1
                points[:, 1] = points[:, 1] * 0.8 + 0.1
        
        return points
    
    def enhanced_simulated_annealing(initial_points, max_iterations=60000):
        """Enhanced simulated annealing with better parameters and early stopping"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Parameters inspired by best practices from inspirations
        temperature = 0.5  # Start with higher temperature for better exploration
        cooling_rate = 0.9995  # Moderate cooling rate for balance between exploration and exploitation
        min_temperature = 1e-12
        max_iterations = max_iterations
        
        # Track best solution
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Track recent improvements for early stopping
        recent_improvements = []
        max_recent = 10
        
        for iteration in range(max_iterations):
            # Random neighbor generation with adaptive perturbation
            neighbor_points = current_points.copy()
            
            # Perturb one random point
            idx = random.randint(0, len(neighbor_points) - 1)
            # Use adaptive step size that decreases with temperature
            step_size = max(0.002, 0.02 * (1 - iteration/max_iterations))
            neighbor_points[idx] += np.random.normal(0, step_size, 2)
            
            # Keep within bounds [0,1]
            neighbor_points = np.clip(neighbor_points, 0, 1)
            
            # Compute ratio for neighbor
            neighbor_ratio = compute_min_max_ratio(neighbor_points)
            
            # Accept or reject based on Metropolis criterion
            if neighbor_ratio > current_ratio:
                # Always accept better solutions
                current_points = neighbor_points
                current_ratio = neighbor_ratio
                
                # Update best solution
                if neighbor_ratio > best_ratio:
                    best_ratio = neighbor_ratio
                    best_points = neighbor_points.copy()
                    recent_improvements.append(best_ratio)
                    if len(recent_improvements) > max_recent:
                        recent_improvements.pop(0)
            else:
                # Accept worse solutions with probability based on temperature
                delta = neighbor_ratio - current_ratio
                acceptance_prob = math.exp(delta / temperature)
                if random.random() < acceptance_prob:
                    current_points = neighbor_points
                    current_ratio = neighbor_ratio
            
            # Cooling schedule
            temperature *= cooling_rate
            
            # Early stopping if no significant improvement
            if len(recent_improvements) >= max_recent:
                if abs(recent_improvements[-1] - recent_improvements[0]) < 1e-12:
                    break
        
        return best_points, best_ratio
    
    def optimize_with_l_bfgs(points, max_iter=500):
        """Optimize using L-BFGS-B with proper constraints."""
        # Flatten for optimization
        x0 = points.flatten()
        
        # Define bounds (0,1) for all coordinates
        bounds = [(0, 1) for _ in range(32)]
        
        def objective_func(x_flat):
            points = x_flat.reshape(-1, 2)
            distances = pdist(points)
            if len(distances) == 0 or np.allclose(distances, 0):
                return np.inf
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist < 1e-10:
                return np.inf
            return -min_dist / max_dist  # Negative because we maximize
        
        try:
            result = minimize(
                objective_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                return optimized_points
        except Exception:
            pass
        return points
    
    # Strategy: Multi-start approach with Fibonacci sphere initialization (most effective from inspiration 1)
    np.random.seed(42)
    random.seed(42)
    
    best_points = None
    best_ratio = 0
    
    # Strategy 1: Fibonacci sphere initialization with enhanced SA (from inspiration 1)
    fib_configs = [
        (42, 0.01, 60000),
        (123, 0.015, 60000),
        (456, 0.012, 60000),
        (789, 0.018, 60000),
    ]
    
    for seed, perturbation, max_iter in fib_configs:
        np.random.seed(seed)
        initial_points = fibonacci_sphere_points(16)
        initial_points += np.random.normal(0, perturbation, initial_points.shape)
        initial_points = np.clip(initial_points, 0, 1)
        
        sa_points, sa_ratio = enhanced_simulated_annealing(initial_points, max_iter)
        if sa_ratio > best_ratio:
            best_ratio = sa_ratio
            best_points = sa_points.copy()
    
    # Strategy 2: Hexagonal pattern initialization (from inspiration 2)
    hex_configs = [
        (111, 0.02, 50000),
        (222, 0.015, 50000),
        (333, 0.025, 50000),
    ]
    
    for seed, perturbation, max_iter in hex_configs:
        np.random.seed(seed)
        initial_points = initialize_hexagonal_lattice()
        initial_points += np.random.normal(0, perturbation, initial_points.shape)
        initial_points = np.clip(initial_points, 0, 1)
        
        sa_points, sa_ratio = enhanced_simulated_annealing(initial_points, max_iter)
        if sa_ratio > best_ratio:
            best_ratio = sa_ratio
            best_points = sa_points.copy()
    
    # Strategy 3: Final gradient refinement on best result (from inspiration 2)
    if best_points is not None:
        try:
            polished_points = optimize_with_l_bfgs(best_points, max_iter=500)
            ratio = compute_min_max_ratio(polished_points)
            if ratio > best_ratio:
                best_points = polished_points.copy()
        except Exception:
            pass
    
    # Fallback to Fibonacci sphere if nothing worked
    if best_points is None:
        np.random.seed(42)
        initial_points = fibonacci_sphere_points(16)
        initial_points += np.random.normal(0, 0.01, initial_points.shape)
        initial_points = np.clip(initial_points, 0, 1)
        best_points, _ = enhanced_simulated_annealing(initial_points, 60000)
    
    return best_points


# EVOLVE-BLOCK-END
