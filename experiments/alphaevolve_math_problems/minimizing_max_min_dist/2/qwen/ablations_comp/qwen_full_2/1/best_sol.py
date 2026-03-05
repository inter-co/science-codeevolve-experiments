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
    
    # Initialize with Fibonacci spiral - proven strong starting configuration
    points = fibonacci_spiral_initialization(n)
    points = normalize_to_unit_square(points)
    
    # Strategy 1: Differential Evolution for global optimization (from Inspiration 2)
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
        
        # Use balanced parameters from Inspiration 2 - more stable approach
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=150,      # Reduced iterations for better performance
            popsize=20,       # Moderate population size
            mutation=(0.8, 1.0),  # Standard mutation rate
            recombination=0.9,   # Standard recombination rate
            atol=1e-12,     # Reasonable tolerance
            tol=1e-12       # Reasonable tolerance
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            if calculate_ratio(optimized_points) > calculate_ratio(points):
                points = optimized_points
                
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
    
    # Strategy 2: Enhanced Simulated Annealing with refined approach (from Inspiration 2)
    try:
        def simulated_annealing_optimization(initial_points, max_iterations=35000):
            current_points = initial_points.copy()
            current_ratio = calculate_ratio(current_points)
            
            best_points = current_points.copy()
            best_ratio = current_ratio
            
            # Temperature schedule from successful inspirations
            temperature = 1.0
            cooling_rate = 0.9995  # Standard cooling rate
            
            for iteration in range(max_iterations):
                # Generate neighbor solution by perturbing one point
                neighbor_points = current_points.copy()
                
                # Choose random point to perturb
                point_idx = np.random.randint(0, len(neighbor_points))
                
                # Perturb with slightly larger steps for better exploration (Inspiration 2)
                delta = np.random.normal(0, 0.003, 2)  # Slightly larger than before
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
                    if temperature > 1e-12:  # Prevent numerical issues
                        acceptance_prob = math.exp((new_ratio - current_ratio) / temperature)
                        if np.random.random() < acceptance_prob:
                            current_points = neighbor_points
                            current_ratio = new_ratio
                
                # Cool down
                temperature *= cooling_rate
                
                # Early stopping if temperature gets very low
                if temperature < 1e-12:
                    break
                    
            return best_points
        
        # Run enhanced simulated annealing with more iterations for better convergence
        sa_points = simulated_annealing_optimization(points, max_iterations=35000)
        sa_ratio = calculate_ratio(sa_points)
        if sa_ratio > calculate_ratio(points):
            points = sa_points
            
    except Exception as e:
        warnings.warn(f"Simulated annealing failed: {e}")
    
    # Strategy 3: Local optimization with multiple restarts (from Inspiration 2)
    try:
        # Multiple restarts with different seeds and reasonable optimization
        best_ratio = calculate_ratio(points)
        best_points = points.copy()
        
        # Use fewer restarts but with better perturbation (from Inspiration 2)
        for restart in range(5):  # Reduced from 10 to 5 for better performance
            # Perturb the best points slightly with different seeds
            np.random.seed(42 + restart)
            start_points = points + np.random.normal(0, 0.01, points.shape)  # Moderate perturbation
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
            
            # Use L-BFGS-B with reasonable tolerances for local refinement
            result = minimize(
                objective_local,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12},  # Reasonable tolerances
                tol=1e-12
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
