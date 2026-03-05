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
    
    def create_hexagonal_arrangement():
        """Create points arranged in a hexagonal pattern"""
        points = []
        rows = 4
        cols = 4
        
        # Hexagonal grid with alternating rows
        for i in range(rows):
            for j in range(cols):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                
                # Apply slight random perturbation
                x += (np.random.random() - 0.5) * 0.05
                y += (np.random.random() - 0.5) * 0.05
                
                # Keep within bounds
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                
                points.append([x, y])
        
        return np.array(points[:16])
    
    def create_fibonacci_arrangement():
        """Create points using fibonacci-like distribution"""
        points = []
        n = 16
        
        for i in range(n):
            angle = 2 * math.pi * i * (3 - math.sqrt(5)) / 2
            radius = math.sqrt(i / (n - 1)) if n > 1 else 0
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            points.append([x, y])
            
        return np.array(points)
    
    # Try multiple initialization strategies and keep the best
    initial_strategies = [
        lambda: fibonacci_spiral_initialization(n),
        create_hexagonal_arrangement,
        create_fibonacci_arrangement
    ]
    
    best_points = None
    best_ratio = 0
    
    # Try multiple initialization strategies
    for strategy in initial_strategies:
        try:
            points = strategy()
            points = normalize_to_unit_square(points)
            ratio = calculate_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
        except Exception:
            continue
    
    # If no good initialization found, use default
    if best_points is None:
        points = fibonacci_spiral_initialization(n)
        points = normalize_to_unit_square(points)
        best_points = points
    
    points = best_points.copy()
    
    # Strategy 1: Differential Evolution for global optimization
    try:
        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return float('inf')
            
            return -min_dist / max_dist
        
        bounds = [(0, 1) for _ in range(2*n)]
        
        # Use very aggressive parameters for better convergence
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=300,      # More iterations for better convergence
            popsize=35,       # Larger population size
            mutation=(0.95, 1.0),  # Very high mutation rate
            recombination=0.98,   # Extremely high recombination rate
            atol=1e-16,     # Extremely tight absolute tolerance
            tol=1e-16       # Extremely tight relative tolerance
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            if calculate_ratio(optimized_points) > calculate_ratio(points):
                points = optimized_points
                
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
    
    # Strategy 2: Enhanced Simulated Annealing for fine-tuning
    try:
        def simulated_annealing_optimization(initial_points, max_iterations=35000):
            current_points = initial_points.copy()
            current_ratio = calculate_ratio(current_points)
            
            best_points = current_points.copy()
            best_ratio = current_ratio
            
            # Very aggressive cooling schedule
            temperature = 1.0
            cooling_rate = 0.9997  # Even more aggressive cooling
            
            for iteration in range(max_iterations):
                # Generate neighbor solution by perturbing one point
                neighbor_points = current_points.copy()
                
                # Choose random point to perturb
                point_idx = np.random.randint(0, len(neighbor_points))
                
                # Perturb the point with adaptive perturbation
                delta = np.random.normal(0, 0.01, 2)  # Larger perturbation for exploration
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
                    if temperature > 1e-16:  # Prevent numerical issues
                        acceptance_prob = math.exp((new_ratio - current_ratio) / temperature)
                        if np.random.random() < acceptance_prob:
                            current_points = neighbor_points
                            current_ratio = new_ratio
                
                # Cool down
                temperature *= cooling_rate
                
                # Early stopping if temperature gets very low
                if temperature < 1e-16:
                    break
                    
            return best_points
        
        # Run simulated annealing with more iterations and better cooling
        sa_points = simulated_annealing_optimization(points, max_iterations=35000)
        sa_ratio = calculate_ratio(sa_points)
        if sa_ratio > calculate_ratio(points):
            points = sa_points
            
    except Exception as e:
        warnings.warn(f"Simulated annealing failed: {e}")
    
    # Strategy 3: Local optimization with multiple restarts
    try:
        # Multiple restarts with different seeds and aggressive optimization
        best_ratio = calculate_ratio(points)
        best_points = points.copy()
        
        # Try more restarts with better diversity
        for restart in range(12):  # Increase from 10 to 12
            # Perturb the best points with varying amounts
            np.random.seed(42 + restart)
            perturbation_scale = 0.003 + (restart * 0.0005)  # Increasing perturbation
            start_points = points + np.random.normal(0, perturbation_scale, points.shape)
            start_points = np.clip(start_points, 0, 1)
            
            def objective_local(x_flat):
                points = x_flat.reshape(-1, 2)
                distances = pdist(points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist == 0:
                    return float('inf')
                
                return -min_dist / max_dist
            
            bounds = [(0, 1) for _ in range(2*n)]
            x0 = start_points.flatten()
            
            # Use L-BFGS-B with even tighter tolerances for local refinement
            result = minimize(
                objective_local,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16},
                tol=1e-16
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = calculate_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        points = best_points
                    
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
    
    # Final validation and return
    return points


# EVOLVE-BLOCK-END
