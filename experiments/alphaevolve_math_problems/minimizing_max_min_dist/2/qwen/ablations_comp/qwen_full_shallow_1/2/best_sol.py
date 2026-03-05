# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import math
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
    
    np.random.seed(42)
    
    def calculate_min_max_ratio(points: np.ndarray) -> float:
        """Calculate the min/max distance ratio for given points."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
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
    
    def generate_regular_grid():
        """Generate a regular 4x4 grid"""
        points = []
        for i in range(4):
            for j in range(4):
                points.append([i/3.0, j/3.0])
        return np.array(points[:16])
    
    def generate_random_config():
        """Generate a random configuration"""
        return np.random.rand(16, 2)
    
    def generate_perturbed_hexagon():
        """Generate a hexagonal pattern with random perturbations"""
        # Create points arranged in a hexagon-like pattern
        points = []
        # Place 16 points in a way that approximates hexagonal packing
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.ones(16) * 0.4
        
        # Perturb some points to avoid degenerate cases
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            x = 0.5 + radius * np.cos(angle) + (np.random.random() - 0.5) * 0.05
            y = 0.5 + radius * np.sin(angle) + (np.random.random() - 0.5) * 0.05
            points.append([np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)])
        
        return np.array(points)
    
    def generate_symmetric_pattern():
        """Generate a symmetric pattern with points near corners and center"""
        points = []
        # Corner points
        corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        for corner in corners:
            points.append(corner)
        
        # Center points in a grid
        for i in range(2):
            for j in range(2):
                points.append([0.5 + (i-0.5)*0.2, 0.5 + (j-0.5)*0.2])
        
        # Additional points to reach 16
        for i in range(8):
            points.append([0.5 + (np.random.random()-0.5)*0.3, 0.5 + (np.random.random()-0.5)*0.3])
        
        return np.array(points[:16])
    
    # Try multiple initialization strategies and use the best one
    initial_strategies = [
        generate_fibonacci_spiral,
        generate_hexagonal_grid,
        generate_regular_grid,
        generate_perturbed_hexagon,
        generate_symmetric_pattern,
        generate_random_config
    ]
    
    best_ratio = -float('inf')
    best_points = None
    
    # Try multiple restarts with different initializations
    for strategy_idx, strategy in enumerate(initial_strategies):
        for restart in range(2):  # Fewer restarts to save time
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
                    seed=42 + strategy_idx * 10 + restart,
                    maxiter=100,  # More iterations for better global search
                    popsize=20,   # Even larger population for better exploration
                    mutation=(0.8, 1.0),  # Higher mutation for more exploration
                    recombination=0.9,    # High recombination rate
                    atol=1e-8,
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
    
    # Perform aggressive local optimization on the best result found so far
    try:
        x0 = best_points.flatten()
        # Try multiple local optimization methods
        methods = ['L-BFGS-B', 'SLSQP']
        for method in methods:
            try:
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
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
    
    # Final refinement with a second round of differential evolution but with more aggressive settings
    try:
        bounds = [(0, 1) for _ in range(32)]
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=999,
            maxiter=50,  # Fewer iterations for speed
            popsize=15,  # Medium population size
            mutation=(0.9, 1.0),  # High mutation
            recombination=0.95,   # Very high recombination
            atol=1e-9,
            rtol=1e-9
        )
        
        if de_result.success:
            refined_points = de_result.x.reshape(-1, 2)
            refined_points[:, 0] = np.clip(refined_points[:, 0], 0, 1)
            refined_points[:, 1] = np.clip(refined_points[:, 1], 0, 1)
            
            ratio = calculate_min_max_ratio(refined_points)
            if ratio > best_ratio:
                best_points = refined_points
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
