# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, differential evolution for global search,
    and local optimization for fine-tuning.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_min_max_ratio(points: np.ndarray) -> float:
        """Calculate the min/max distance ratio for given points."""
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0
    
    def objective(x):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x.reshape(-1, 2)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    def generate_fibonacci_spiral():
        """Generate points using Fibonacci spiral for good distribution"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(16):
            # Fibonacci spiral approach
            theta = i * 2 * np.pi / golden_ratio
            r = np.sqrt(i / 15.0)  # Normalize to [0,1] range
            
            x = 0.5 + r * np.cos(theta) * 0.4
            y = 0.5 + r * np.sin(theta) * 0.4
            
            # Add some randomness
            x += (np.random.random() - 0.5) * 0.05
            y += (np.random.random() - 0.5) * 0.05
            
            # Keep within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            points.append([x, y])
        
        return np.array(points)
    
    def generate_clustered_pattern():
        """Generate points in clusters for better exploration"""
        points = []
        # Create 4 clusters of 4 points each
        cluster_centers = [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75]
        ]
        
        for center in cluster_centers:
            for _ in range(4):
                # Add small random perturbation around cluster center
                x = center[0] + (np.random.random() - 0.5) * 0.2
                y = center[1] + (np.random.random() - 0.5) * 0.2
                # Keep within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                points.append([x, y])
        
        return np.array(points)
    
    def generate_hexagonal_grid():
        """Generate a hexagonal-like grid pattern for good initial distribution"""
        points = []
        # Create a 4x4 grid with alternating offsets to create hexagonal packing
        for i in range(4):
            for j in range(4):
                offset_x = 0.5 if j % 2 == 0 else 0.75
                offset_y = 0.5 if i % 2 == 0 else 0.75
                x = (i + offset_x) / 4.0
                y = (j + offset_y) / 4.0
                points.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
        return np.array(points)
    
    def generate_perturbed_circle():
        """Generate points on a circle with perturbations"""
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.ones(16) * 0.4
        
        # Add some irregularity to avoid perfect circle
        radii += (np.random.rand(16) - 0.5) * 0.1
        
        # Convert to Cartesian coordinates and scale to [0,1] box
        x = 0.5 + radii * np.cos(angles) * 0.4
        y = 0.5 + radii * np.sin(angles) * 0.4
        
        # Add some randomness to break perfect circle
        x += (np.random.rand(16) - 0.5) * 0.1
        y += (np.random.rand(16) - 0.5) * 0.1
        
        # Clip to valid range
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        points = np.column_stack([x, y])
        return points
    
    def generate_random_distribution():
        """Generate completely random points"""
        return np.random.rand(16, 2)
    
    def generate_regular_grid():
        """Generate a regular grid pattern"""
        points = []
        for i in range(4):
            for j in range(4):
                x = i / 3.0
                y = j / 3.0
                points.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
        return np.array(points)
    
    # Try multiple initialization strategies and use the best one
    initial_strategies = [
        generate_fibonacci_spiral,
        generate_clustered_pattern,
        generate_hexagonal_grid,
        generate_perturbed_circle,
        generate_random_distribution,
        generate_regular_grid
    ]
    
    best_ratio = -float('inf')
    best_points = None
    
    # Try multiple restarts with different initializations
    for strategy in initial_strategies:
        for restart in range(5):  # More restarts for better exploration
            try:
                # Generate initial points
                points = strategy()
                
                # Add some randomness to avoid degenerate cases
                points += np.random.normal(0, 0.01, points.shape)
                points = np.clip(points, 0, 1)
                
                # First, try differential evolution for global optimization
                bounds = [(0, 1) for _ in range(32)]
                de_result = differential_evolution(
                    objective, 
                    bounds, 
                    seed=42 + restart,
                    maxiter=100,  # Balanced iterations for time constraint
                    popsize=20,   # Larger population for better exploration
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    atol=1e-8,    # Tighter tolerance
                    rtol=1e-8
                )
                
                # Extract optimized points from DE
                optimized_points = de_result.x.reshape(-1, 2)
                # Ensure points stay within bounds
                optimized_points[:, 0] = np.clip(optimized_points[:, 0], 0, 1)
                optimized_points[:, 1] = np.clip(optimized_points[:, 1], 0, 1)
                
                # Evaluate the result
                ratio = calculate_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
            except Exception:
                continue
    
    # If we still don't have a good solution, fall back to a good initialization
    if best_points is None:
        # Use Fibonacci spiral as fallback
        points = generate_fibonacci_spiral()
        # Add some randomness
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        best_points = points
        best_ratio = calculate_min_max_ratio(best_points)
    
    # Perform enhanced local optimization on the best result found so far
    try:
        x0 = best_points.flatten()
        
        # Try multiple local optimization methods with different settings
        optimization_configs = [
            ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}),
            ('SLSQP', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}),
            ('TNC', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        ]
        
        for method, options in optimization_configs:
            try:
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    options=options
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    ratio = calculate_min_max_ratio(optimized_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
            except Exception:
                continue
                
    except Exception:
        pass
    
    # Additional refinement with multiple local searches
    try:
        # Try several random restarts from the current best
        for _ in range(3):
            # Perturb current best points
            perturbed = best_points + (np.random.rand(16, 2) - 0.5) * 0.05
            perturbed = np.clip(perturbed, 0, 1)
            
            x0 = perturbed.flatten()
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = calculate_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
