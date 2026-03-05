# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, multiple restarts, and advanced optimization.

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
    
    def generate_hexagonal_pattern():
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
    
    def improved_simulated_annealing(initial_points, max_iterations=60000):
        """Enhanced simulated annealing with improved cooling schedule and parameters"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Very aggressive cooling schedule inspired by best practices from inspirations
        temp = 0.5  # Higher initial temperature
        cooling_rate = 0.9996  # Slightly more aggressive cooling rate
        min_temp = 1e-16  # Even smaller minimum temperature
        
        # Track best solution
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Adaptive step size with better initial value
        step_size = 0.06  # Slightly larger initial step size
        
        # Track recent improvements for early stopping
        recent_improvements = []
        max_recent = 25
        stagnation_count = 0
        max_stagnation = 2500  # More aggressive early stopping
        
        for iteration in range(max_iterations):
            # Random neighbor generation
            neighbor_points = current_points.copy()
            
            # Perturb one random point
            idx = random.randint(0, len(neighbor_points) - 1)
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
                    best_points = neighbor_points.copy()
                    best_ratio = neighbor_ratio
                    recent_improvements.append(best_ratio)
                    if len(recent_improvements) > max_recent:
                        recent_improvements.pop(0)
                    stagnation_count = 0
            else:
                # Accept worse solutions with probability based on temperature
                delta = neighbor_ratio - current_ratio
                # Prevent numerical overflow in exponential
                if delta < -100:
                    acceptance_prob = 0.0
                else:
                    acceptance_prob = math.exp(delta / temp)
                if random.random() < acceptance_prob:
                    current_points = neighbor_points
                    current_ratio = neighbor_ratio
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            
            # Cooling schedule with adaptive behavior
            temp = max(temp * cooling_rate, min_temp)
            
            # Adaptive step size adjustment - slightly faster decay
            if iteration % 400 == 0 and iteration > 0:
                step_size *= 0.96
            
            # Early stopping condition for stagnation
            if stagnation_count > max_stagnation and iteration > 5000:
                break
            
            # Early stopping based on recent improvement range
            if len(recent_improvements) >= max_recent:
                improvement_range = max(recent_improvements) - min(recent_improvements)
                if improvement_range < 1e-11:
                    break
        
        return best_points, best_ratio
    
    def optimize_with_l_bfgs(points, max_iter=300):
        """Optimize using L-BFGS-B with proper constraints and better parameters."""
        # Flatten for optimization
        x0 = points.flatten()
        
        # Define bounds (0,1) for all coordinates
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            result = minimize(
                lambda x: -compute_min_max_ratio(x.reshape(-1, 2)),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                return optimized_points
        except Exception:
            pass
        return points
    
    def coordinate_descent_refinement(points, max_iter=200):
        """Enhanced coordinate descent refinement to polish the solution"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        for iteration in range(max_iter):
            improved = False
            # Shuffle point indices for better exploration
            point_indices = list(range(len(current_points)))
            np.random.shuffle(point_indices)
            
            for i in point_indices:
                old_point = current_points[i].copy()
                best_point = old_point.copy()
                best_ratio = current_ratio
                
                # Try perturbations with more varied scales for better search
                scales = [0.001, 0.003, 0.005, 0.007, 0.01]
                for scale in scales:
                    for _ in range(40):  # Even more iterations per scale
                        perturbation = np.random.normal(0, scale, 2)
                        new_point = old_point + perturbation
                        new_point = np.clip(new_point, 0, 1)
                        
                        test_points = current_points.copy()
                        test_points[i] = new_point
                        new_ratio = compute_min_max_ratio(test_points)
                        
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_point = new_point
                            improved = True
                            break  # Early exit if improvement found
                
                current_points[i] = best_point
                current_ratio = best_ratio
            
            if not improved:
                break
        
        return current_points
    
    # Strategy: Enhanced multi-start approach with better initialization and optimization
    np.random.seed(42)
    random.seed(42)
    
    best_points = None
    best_ratio = 0
    
    # Strategy 1: Multiple restarts with Fibonacci sphere initialization (most effective)
    fibonacci_configs = [
        (42, 0.01, 60000),
        (123, 0.015, 60000),
        (456, 0.012, 60000),
        (789, 0.018, 60000),
        (999, 0.02, 60000),
        (111, 0.014, 60000),
        (222, 0.016, 60000),
        (333, 0.013, 60000),
    ]
    
    for seed, perturbation, max_iter in fibonacci_configs:
        np.random.seed(seed)
        initial_points = fibonacci_sphere_points(16)
        initial_points += np.random.normal(0, perturbation, initial_points.shape)
        initial_points = np.clip(initial_points, 0, 1)
        
        sa_points, sa_ratio = improved_simulated_annealing(initial_points, max_iter)
        if sa_ratio > best_ratio:
            best_ratio = sa_ratio
            best_points = sa_points.copy()
    
    # Strategy 2: Additional hexagonal pattern restarts
    hex_configs = [
        (333, 0.02, 50000),
        (555, 0.015, 50000),
        (666, 0.025, 50000),
        (777, 0.018, 50000),
        (888, 0.012, 50000),
    ]
    
    for seed, perturbation, max_iter in hex_configs:
        np.random.seed(seed)
        initial_points = generate_hexagonal_pattern()
        initial_points += np.random.normal(0, perturbation, initial_points.shape)
        initial_points = np.clip(initial_points, 0, 1)
        
        sa_points, sa_ratio = improved_simulated_annealing(initial_points, max_iter)
        if sa_ratio > best_ratio:
            best_ratio = sa_ratio
            best_points = sa_points.copy()
    
    # Strategy 3: Regular polygon pattern
    np.random.seed(888)
    regular_points = np.array([[0.5 + 0.4 * math.cos(2 * math.pi * i / 16), 
                               0.5 + 0.4 * math.sin(2 * math.pi * i / 16)] 
                              for i in range(16)])
    regular_points += np.random.normal(0, 0.01, regular_points.shape)
    regular_points = np.clip(regular_points, 0, 1)
    
    sa_points, sa_ratio = improved_simulated_annealing(regular_points, 50000)
    if sa_ratio > best_ratio:
        best_ratio = sa_ratio
        best_points = sa_points.copy()
    
    # Strategy 4: Concentric rings pattern for diversity
    np.random.seed(999)
    # Create points in concentric rings
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
    concentric_points = np.array(points)
    concentric_points += np.random.normal(0, 0.01, concentric_points.shape)
    concentric_points = np.clip(concentric_points, 0, 1)
    
    sa_points, sa_ratio = improved_simulated_annealing(concentric_points, 50000)
    if sa_ratio > best_ratio:
        best_ratio = sa_ratio
        best_points = sa_points.copy()
    
    # Strategy 5: Random initialization with long SA run
    np.random.seed(555)
    random_points = np.random.rand(16, 2)
    random_points += np.random.normal(0, 0.015, random_points.shape)
    random_points = np.clip(random_points, 0, 1)
    
    sa_points, sa_ratio = improved_simulated_annealing(random_points, 60000)
    if sa_ratio > best_ratio:
        best_ratio = sa_ratio
        best_points = sa_points.copy()
    
    # Strategy 6: Final refinement with coordinate descent
    if best_points is not None:
        refined_points = coordinate_descent_refinement(best_points, max_iter=250)
        ratio = compute_min_max_ratio(refined_points)
        if ratio > best_ratio:
            best_points = refined_points.copy()
    
    # Strategy 7: Final gradient refinement on best result
    if best_points is not None:
        try:
            polished_points = optimize_with_l_bfgs(best_points, max_iter=300)
            ratio = compute_min_max_ratio(polished_points)
            if ratio > best_ratio:
                best_points = polished_points.copy()
        except Exception:
            pass
    
    # Strategy 8: Comprehensive SA with multiple configurations if nothing worked well
    if best_points is None:
        # Run more aggressive simulated annealing with different configurations
        best_points = None
        best_ratio = 0
        
        # Different seeds and perturbations for robustness
        sa_configs = [
            (1234, 0.01, 60000),
            (5678, 0.015, 60000),
            (9012, 0.02, 60000),
            (3456, 0.012, 60000),
            (7890, 0.018, 60000),
        ]
        
        for seed, perturbation, max_iter in sa_configs:
            np.random.seed(seed)
            # Try fibonacci again with different parameters
            initial_points = fibonacci_sphere_points(16)
            initial_points += np.random.normal(0, perturbation, initial_points.shape)
            initial_points = np.clip(initial_points, 0, 1)
            
            sa_points, sa_ratio = improved_simulated_annealing(initial_points, max_iter)
            if sa_ratio > best_ratio:
                best_ratio = sa_ratio
                best_points = sa_points.copy()
    
    # Fallback to the best fibonacci configuration if nothing worked
    if best_points is None:
        np.random.seed(42)
        initial_points = fibonacci_sphere_points(16)
        initial_points += np.random.normal(0, 0.01, initial_points.shape)
        initial_points = np.clip(initial_points, 0, 1)
        best_points, _ = improved_simulated_annealing(initial_points, 60000)
    
    return best_points


# EVOLVE-BLOCK-END
