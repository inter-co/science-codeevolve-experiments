# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust hybrid approach inspired by successful optimization strategies.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    
    def calculate_ratio(points):
        """Calculate the min/max distance ratio."""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0
        
        return min_dist / max_dist
    
    def fibonacci_spiral_initialization(n: int) -> np.ndarray:
        """Generate points on a Fibonacci spiral for good initial distribution."""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        for i in range(n):
            theta = i * 2 * math.pi / phi
            r = math.sqrt(i / (n - 1)) if n > 1 else 0
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            points.append([x, y])
        
        return np.array(points)
    
    def normalize_to_unit_square(points: np.ndarray) -> np.ndarray:
        """Normalize points to fit within [0,1] x [0,1]."""
        if len(points) == 0:
            return points
            
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # Handle edge case where all points are the same
        if x_max == x_min and y_max == y_min:
            return points
            
        # Avoid division by zero
        if x_max != x_min and y_max != y_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        
        # Scale to [0,1] range with padding
        points[:, 0] = points[:, 0] * 0.9 + 0.05
        points[:, 1] = points[:, 1] * 0.9 + 0.05
        
        return points
    
    # Generate multiple diverse initial configurations (like Inspo 1)
    def generate_hexagonal_arrangement():
        """Generate points arranged in a hexagonal pattern"""
        points = []
        # Use 4x4 grid with hexagonal offset
        rows = 4
        cols = 4
        spacing = 0.25  # More compact spacing
        offset = spacing * 0.5
        
        for i in range(rows):
            for j in range(cols):
                x = j * spacing
                y = i * spacing * np.sqrt(3) / 2
                if i % 2 == 1:  # Offset every other row
                    x += offset
                points.append([x, y])
        
        # Trim to exactly 16 points if needed
        points = points[:16]
        # Add slight randomization to avoid degenerate cases
        points = np.array(points) + np.random.normal(0, 0.005, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def generate_grid_arrangement():
        """Generate points in a regular grid"""
        points = []
        for i in range(4):
            for j in range(4):
                points.append([i/3.0, j/3.0])
        
        # Add slight randomization
        points = np.array(points) + np.random.normal(0, 0.01, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def generate_circle_plus_inner_initialization(n: int) -> np.ndarray:
        """Initialize points in a circle with inner points for better spread"""
        points = []
        
        # Place points around a circle
        num_outer = 12
        num_inner = 4
        
        # Outer points - distributed around circle
        for i in range(num_outer):
            angle = 2 * np.pi * i / num_outer
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        
        # Inner points - scattered inside
        for i in range(num_inner):
            angle = 2 * np.pi * i / num_inner
            radius = 0.15
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        
        # Fill remaining points with random distribution
        while len(points) < n:
            points.append([np.random.random(), np.random.random()])
        
        points = np.array(points[:n])
        
        # Add small random noise
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    def generate_golden_ratio_grid_initialization(n: int) -> np.ndarray:
        """Initialize points using golden ratio based distribution"""
        # Golden ratio spiral approach
        points = []
        
        # Generate points along a golden spiral pattern
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(n):
            angle = i * 2 * np.pi / phi
            radius = i / (n - 1) * 0.4 + 0.1  # radial distribution
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        
        # If we didn't get enough points, fill with random
        while len(points) < n:
            points.append([np.random.random(), np.random.random()])
        
        points = np.array(points[:n])
        
        # Add small random noise
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    # Generate multiple diverse initial configurations (like Inspo 1)
    initial_configs = [
        fibonacci_spiral_initialization(16),
        generate_hexagonal_arrangement(),
        generate_grid_arrangement(),
        generate_circle_plus_inner_initialization(16),
        generate_golden_ratio_grid_initialization(16),
        np.random.rand(16, 2),
        np.random.uniform(0.1, 0.9, (16, 2))
    ]
    
    # Normalize all initial configurations to unit square
    for i in range(len(initial_configs)):
        initial_configs[i] = normalize_to_unit_square(initial_configs[i])
    
    # Evaluate all initial configurations and find the best one
    best_initial_config = None
    best_initial_ratio = -np.inf
    
    for config in initial_configs:
        ratio = calculate_ratio(config)
        if ratio > best_initial_ratio:
            best_initial_ratio = ratio
            best_initial_config = config.copy()
    
    points = best_initial_config.copy()
    
    # Strategy 1: Differential Evolution with highly aggressive parameters (Inspo 1)
    try:
        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return float('inf')
            
            # Small epsilon to prevent division by zero
            eps = 1e-15
            return -min_dist / (max_dist + eps)
        
        bounds = [(0, 1) for _ in range(2*n)]
        
        # Very aggressive DE parameters like Inspo 1 for better convergence
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=250,  # Increased for better convergence (Inspo 1)
            popsize=30,   # Increased for better diversity (Inspo 1)
            mutation=(0.9, 1.0),  # More aggressive mutation (Inspo 1)
            recombination=0.95,   # Higher recombination rate (Inspo 1)
            atol=1e-15,  # Tighter tolerance (Inspo 1)
            tol=1e-15    # Tighter tolerance (Inspo 1)
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            if calculate_ratio(optimized_points) > calculate_ratio(points):
                points = optimized_points.copy()
                
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
    
    # Strategy 2: Enhanced Simulated Annealing with more iterations (Inspo 1)
    try:
        def simulated_annealing_optimization(initial_points, max_iterations=35000):
            current_points = initial_points.copy()
            current_ratio = calculate_ratio(current_points)
            
            best_points = current_points.copy()
            best_ratio = current_ratio
            
            # More aggressive cooling schedule like Inspo 1
            temperature = 1.0
            cooling_rate = 0.9995  # More aggressive cooling (Inspo 1)
            
            for iteration in range(max_iterations):
                # Generate neighbor solution by perturbing one point
                neighbor_points = current_points.copy()
                
                # Choose random point to perturb
                point_idx = np.random.randint(0, len(neighbor_points))
                
                # Perturb with larger step size for better exploration (Inspo 1)
                delta = np.random.normal(0, 0.007, 2)  # Larger perturbation (Inspo 1)
                neighbor_points[point_idx] += delta
                
                # Keep within bounds [0,1]
                neighbor_points[point_idx] = np.clip(neighbor_points[point_idx], 0, 1)
                
                # Calculate new ratio
                new_ratio = calculate_ratio(neighbor_points)
                
                # Accept or reject based on Metropolis criterion
                if new_ratio > current_ratio:
                    current_points = neighbor_points
                    current_ratio = new_ratio
                    
                    # Update best solution
                    if new_ratio > best_ratio:
                        best_points = neighbor_points.copy()
                        best_ratio = new_ratio
                else:
                    # Accept with probability based on temperature
                    if temperature > 1e-15:  # Prevent numerical issues
                        acceptance_prob = np.exp((new_ratio - current_ratio) / temperature)
                        if np.random.random() < acceptance_prob:
                            current_points = neighbor_points
                            current_ratio = new_ratio
                
                # Cool down
                temperature *= cooling_rate
                
                # Early stopping if temperature gets very low
                if temperature < 1e-15:
                    break
                    
            return best_points
        
        # Run simulated annealing with more iterations than before (Inspo 1)
        sa_points = simulated_annealing_optimization(points, max_iterations=35000)
        sa_ratio = calculate_ratio(sa_points)
        if sa_ratio > calculate_ratio(points):
            points = sa_points.copy()
            
    except Exception as e:
        warnings.warn(f"Simulated annealing failed: {e}")
    
    # Strategy 3: Local optimization with multiple restarts (Inspo 1)
    try:
        # Multiple restarts with different seeds and aggressive optimization (Inspo 1)
        best_ratio = calculate_ratio(points)
        best_points = points.copy()
        
        # Increase restarts to 10 for better chance of finding global optimum (Inspo 1)
        for restart in range(10):
            # Perturb the best points slightly with different seeds
            np.random.seed(42 + restart)
            start_points = points + np.random.normal(0, 0.015, points.shape)  # Larger perturbation (Inspo 1)
            start_points = np.clip(start_points, 0, 1)
            
            def objective(x_flat):
                points = x_flat.reshape(-1, 2)
                distances = pdist(points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist == 0:
                    return float('inf')
                
                # Small epsilon to prevent division by zero
                eps = 1e-15
                return -min_dist / (max_dist + eps)
            
            bounds = [(0, 1) for _ in range(2*n)]
            x0 = start_points.flatten()
            
            # Use L-BFGS-B with even tighter tolerances for local refinement (Inspo 1)
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-15},  # Very tight tolerances (Inspo 1)
                tol=1e-15
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = calculate_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        points = best_points.copy()
                    
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
    
    return points


# EVOLVE-BLOCK-END
