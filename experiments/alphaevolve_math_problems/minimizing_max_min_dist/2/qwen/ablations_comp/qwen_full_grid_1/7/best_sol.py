# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
import random
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, multiple restarts, and 
    advanced optimization techniques for robust global optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all pairs"""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 1e-12:
            return 0
            
        return min_dist / max_dist
    
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
    
    def create_hexagonal_pattern():
        """Create a hexagonal lattice pattern with better spacing."""
        points = []
        rows, cols = 4, 4
        
        spacing_x = 1.0
        spacing_y = math.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Offset every other row for better hexagonal packing
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                points.append([x, y])
        
        # Convert to numpy array and normalize to [0,1]x[0,1]
        points = np.array(points[:16])
        
        if len(points) > 0:
            # Normalize coordinates to fit in [0,1] range
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0 and y_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
                
                # Scale and center to [0.1, 0.9] range to avoid boundary issues
                points[:, 0] = points[:, 0] * 0.8 + 0.1
                points[:, 1] = points[:, 1] * 0.8 + 0.1
        
        return points
    
    def create_regular_polygon_pattern():
        """Create points arranged in a regular 16-gon for symmetry."""
        points = []
        for i in range(16):
            angle = 2 * math.pi * i / 16
            x = 0.5 + 0.4 * math.cos(angle)
            y = 0.5 + 0.4 * math.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def create_concentric_ring_pattern():
        """Create points in concentric rings for balanced distribution."""
        points = []
        # Outer ring: 12 points
        for i in range(12):
            angle = 2 * math.pi * i / 12
            x = 0.5 + 0.4 * math.cos(angle)
            y = 0.5 + 0.4 * math.sin(angle)
            points.append([x, y])
        
        # Inner square: 4 points
        inner_radius = 0.15
        points.extend([
            [0.5 - inner_radius, 0.5 - inner_radius],  # bottom-left
            [0.5 + inner_radius, 0.5 - inner_radius],  # bottom-right
            [0.5 - inner_radius, 0.5 + inner_radius],  # top-left
            [0.5 + inner_radius, 0.5 + inner_radius]   # top-right
        ])
        return np.array(points)
    
    def improved_simulated_annealing(initial_points, max_iterations=50000):
        """Improved simulated annealing with better parameters and early stopping"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Parameters inspired by best practices from inspirations
        temperature = 0.3  # Start with moderate temperature for exploration
        cooling_rate = 0.9996  # Slightly more aggressive cooling
        min_temperature = 1e-12
        max_iterations = max_iterations
        
        # Track best solution
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Track recent improvements for early stopping
        recent_improvements = []
        max_recent = 15
        
        # Track stagnation for early stopping
        stagnation_count = 0
        max_stagnation = 10000
        
        for iteration in range(max_iterations):
            # Random neighbor generation with adaptive perturbation
            neighbor_points = current_points.copy()
            
            # Perturb one random point
            idx = random.randint(0, len(neighbor_points) - 1)
            # Use adaptive step size that decreases with temperature
            step_size = max(0.001, 0.02 * (1 - iteration/max_iterations))
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
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                # Accept worse solutions with probability based on temperature
                delta = neighbor_ratio - current_ratio
                acceptance_prob = math.exp(delta / temperature)
                if random.random() < acceptance_prob:
                    current_points = neighbor_points
                    current_ratio = neighbor_ratio
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            
            # Cooling schedule
            temperature *= cooling_rate
            
            # Early stopping if no significant improvement
            if len(recent_improvements) >= max_recent:
                if abs(recent_improvements[-1] - recent_improvements[0]) < 1e-12:
                    break
                    
            # Early stopping for stagnation
            if stagnation_count > max_stagnation:
                break
        
        return best_points, best_ratio
    
    def optimize_with_l_bfgs(points, max_iter=300):
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
    
    # Strategy: Multi-start approach with diverse initialization strategies
    np.random.seed(42)
    random.seed(42)
    
    best_points = None
    best_ratio = 0
    
    # Strategy 1: Fibonacci sphere initialization (most effective from inspiration 1)
    fib_configs = [
        (42, 0.01, 50000),
        (123, 0.015, 50000),
        (456, 0.012, 50000),
        (789, 0.018, 50000),
        (999, 0.01, 60000),  # Extended run for better chance
    ]
    
    for seed, perturbation, max_iter in fib_configs:
        np.random.seed(seed)
        initial_points = fibonacci_sphere_points(16)
        initial_points += np.random.normal(0, perturbation, initial_points.shape)
        initial_points = np.clip(initial_points, 0, 1)
        
        sa_points, sa_ratio = improved_simulated_annealing(initial_points, max_iter)
        if sa_ratio > best_ratio:
            best_ratio = sa_ratio
            best_points = sa_points.copy()
    
    # Strategy 2: Hexagonal pattern initialization (from inspiration 2)
    hex_configs = [
        (111, 0.02, 45000),
        (222, 0.015, 45000),
        (333, 0.025, 45000),
        (444, 0.01, 50000),  # Extra run
    ]
    
    for seed, perturbation, max_iter in hex_configs:
        np.random.seed(seed)
        initial_points = create_hexagonal_pattern()
        initial_points += np.random.normal(0, perturbation, initial_points.shape)
        initial_points = np.clip(initial_points, 0, 1)
        
        sa_points, sa_ratio = improved_simulated_annealing(initial_points, max_iter)
        if sa_ratio > best_ratio:
            best_ratio = sa_ratio
            best_points = sa_points.copy()
    
    # Strategy 3: Regular polygon pattern (from inspiration 1)
    np.random.seed(999)
    initial_points = create_regular_polygon_pattern()
    initial_points += np.random.normal(0, 0.01, initial_points.shape)
    initial_points = np.clip(initial_points, 0, 1)
    
    sa_points, sa_ratio = improved_simulated_annealing(initial_points, 50000)
    if sa_ratio > best_ratio:
        best_ratio = sa_ratio
        best_points = sa_points.copy()
    
    # Strategy 4: Concentric rings pattern (from inspiration 1)
    np.random.seed(555)
    initial_points = create_concentric_ring_pattern()
    initial_points += np.random.normal(0, 0.015, initial_points.shape)
    initial_points = np.clip(initial_points, 0, 1)
    
    sa_points, sa_ratio = improved_simulated_annealing(initial_points, 50000)
    if sa_ratio > best_ratio:
        best_ratio = sa_ratio
        best_points = sa_points.copy()
    
    # Strategy 5: Random initialization with longer SA run (from inspiration 2)
    np.random.seed(888)
    random_points = np.random.rand(16, 2)
    random_points += np.random.normal(0, 0.015, random_points.shape)
    random_points = np.clip(random_points, 0, 1)
    
    sa_points, sa_ratio = improved_simulated_annealing(random_points, 60000)
    if sa_ratio > best_ratio:
        best_ratio = sa_ratio
        best_points = sa_points.copy()
    
    # Strategy 6: Additional restart with carefully selected parameters
    np.random.seed(333)
    # Try a configuration with different spacing
    points_grid = np.zeros((16, 2))
    for i in range(4):
        for j in range(4):
            points_grid[i*4 + j] = [(j+0.5)/4.0, (i+0.5)/4.0]
    # Add small perturbation
    points_grid += np.random.normal(0, 0.02, points_grid.shape)
    points_grid = np.clip(points_grid, 0, 1)
    
    sa_points, sa_ratio = improved_simulated_annealing(points_grid, 50000)
    if sa_ratio > best_ratio:
        best_ratio = sa_ratio
        best_points = sa_points.copy()
    
    # Strategy 7: Final gradient refinement on best result (from inspiration 2)
    if best_points is not None:
        try:
            polished_points = optimize_with_l_bfgs(best_points, max_iter=400)
            ratio = compute_min_max_ratio(polished_points)
            if ratio > best_ratio:
                best_points = polished_points.copy()
        except Exception:
            pass
    
    # Fallback to the best configuration if nothing worked
    if best_points is None:
        # Use Fibonacci sphere as fallback (most geometrically sound)
        np.random.seed(42)
        initial_points = fibonacci_sphere_points(16)
        initial_points += np.random.normal(0, 0.01, initial_points.shape)
        initial_points = np.clip(initial_points, 0, 1)
        best_points, _ = improved_simulated_annealing(initial_points, 60000)
    
    return best_points


# EVOLVE-BLOCK-END
