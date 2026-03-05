# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import math
import random
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining global and local optimization strategies.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Strategy 1: Try multiple initialization methods and pick the best
    initial_strategies = [
        generate_hexagonal_perturbed,
        generate_random_spherical,
        generate_fibonacci_sphere,
        generate_grid_with_noise
    ]
    
    best_points = None
    best_ratio = -float('inf')
    
    # Try different initialization approaches
    for strategy in initial_strategies:
        try:
            points = strategy()
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
        except Exception:
            continue
    
    if best_points is None:
        best_points = generate_hexagonal_perturbed()
    
    # Strategy 2: Global optimization using differential evolution with multiple restarts
    points = optimize_with_differential_evolution(best_points)
    
    # Strategy 3: Multiple local optimizations with different approaches
    points = multi_strategy_local_optimization(points)
    
    # Strategy 4: Final aggressive refinement
    points = final_refinement(points)
    
    return points


def generate_hexagonal_perturbed() -> np.ndarray:
    """Generate initial points using a hexagonal pattern with random perturbations"""
    # Create a hexagonal lattice pattern
    points = []
    
    # Hexagonal pattern parameters
    spacing = 0.25  # Smaller spacing to better utilize space
    row_spacing = spacing * math.sqrt(3) / 2
    
    # Generate points in a hexagonal arrangement
    for i in range(4):
        for j in range(4):
            x = j * spacing + (i % 2) * spacing / 2
            y = i * row_spacing
            
            # Ensure points stay within [0,1] bounds
            if x <= 1 and y <= 1:
                points.append([x, y])
    
    # Trim to exactly 16 points if needed
    if len(points) > 16:
        points = points[:16]
    elif len(points) < 16:
        # Add additional points using a more strategic approach
        for i in range(16 - len(points)):
            # Place points in a circular pattern to fill gaps
            angle = 2 * math.pi * i / (16 - len(points))
            radius = 0.3 + i * 0.03
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
    
    points = np.array(points[:16])
    
    # Add small random perturbations to break symmetries
    noise = np.random.normal(0, 0.005, points.shape)  # Smaller noise for better stability
    points = points + noise
    points = np.clip(points, 0, 1)
    
    return points


def generate_random_spherical() -> np.ndarray:
    """Generate points with random distribution that tends to spread well"""
    points = np.random.rand(16, 2)
    # Add some structure to avoid overly clustered solutions
    # Use more strategic placement for better distribution
    for i in range(16):
        if i % 4 == 0:
            points[i] = [0.5 + 0.4 * (np.random.rand() - 0.5), 0.5 + 0.4 * (np.random.rand() - 0.5)]
        elif i % 4 == 1:
            points[i] = [0.2 + 0.6 * np.random.rand(), 0.2 + 0.6 * np.random.rand()]
        elif i % 4 == 2:
            points[i] = [0.3 + 0.4 * np.random.rand(), 0.3 + 0.4 * np.random.rand()]
    return np.clip(points, 0, 1)


def generate_fibonacci_sphere() -> np.ndarray:
    """Generate points using Fibonacci spiral (approximation)"""
    points = []
    n = 16
    golden_ratio = (1 + math.sqrt(5)) / 2
    
    for i in range(n):
        # Fibonacci spiral on sphere (projected to 2D)
        phi = math.acos(-1 + (2 * i) / (n - 1))
        theta = math.sqrt(n * math.pi) * phi
        
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        
        # Project to 2D and scale to unit square with better distribution
        points.append([0.5 + 0.45 * x, 0.5 + 0.45 * y])
    
    points = np.array(points)
    return np.clip(points, 0, 1)


def generate_grid_with_noise() -> np.ndarray:
    """Generate a structured grid with noise"""
    points = []
    for i in range(4):
        for j in range(4):
            # Create a more strategic grid layout
            x = i * 0.333 + 0.02 * (np.random.rand() - 0.5) 
            y = j * 0.333 + 0.02 * (np.random.rand() - 0.5)
            points.append([x, y])
    
    points = np.array(points[:16])
    return np.clip(points, 0, 1)


def compute_min_max_ratio(points: np.ndarray) -> float:
    """Compute the ratio of minimum to maximum distance"""
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist == 0:
        return 0.0
    return min_dist / max_dist


def optimize_with_differential_evolution(initial_points: np.ndarray) -> np.ndarray:
    """Use differential evolution for global optimization"""
    n_points = len(initial_points)
    
    def objective(params):
        # Reshape back to 2D array
        points = params.reshape(n_points, 2)
        
        # Compute distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 1e10  # Very large if no distances
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 1e10
            
        # Return negative because we want to maximize the ratio
        # But we return positive value for DE (minimize)
        return -min_dist / max_dist
    
    # Define bounds: [0,1] for both coordinates
    bounds = [(0, 1) for _ in range(2 * n_points)]
    
    # Use differential evolution with more aggressive parameters for better convergence
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=300,  # More iterations for better convergence
            popsize=20,   # More population for better exploration
            mutation=(0.8, 1.0),  # More aggressive mutation
            recombination=0.9,    # Higher recombination rate
            seed=42,
            disp=False,
            tol=1e-8
        )
        
        if result.success:
            optimized_points = result.x.reshape(n_points, 2)
            return np.clip(optimized_points, 0, 1)
    except Exception:
        pass
    
    return initial_points.copy()


def multi_strategy_local_optimization(initial_points: np.ndarray) -> np.ndarray:
    """Apply multiple rounds of local optimization with different strategies"""
    n_points = len(initial_points)
    current_points = initial_points.copy()
    current_ratio = compute_min_max_ratio(current_points)
    
    # Try different optimization approaches with multiple restarts
    strategies = [
        ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}),
        ('TNC', {'maxiter': 500, 'ftol': 1e-10}),
        ('SLSQP', {'maxiter': 500, 'ftol': 1e-10})
    ]
    
    # Multiple restarts with different perturbations
    for restart in range(10):  # More restarts for better exploration
        # Slightly perturb current solution
        perturbed = current_points + np.random.normal(0, 0.005, current_points.shape)  # Larger perturbation
        perturbed = np.clip(perturbed, 0, 1)
        
        # Try different optimization methods
        for method, options in strategies:
            try:
                def objective(params):
                    points = params.reshape(n_points, 2)
                    distances = pdist(points)
                    if len(distances) == 0:
                        return 1e10
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist == 0:
                        return 1e10
                    return -min_dist / max_dist
                
                bounds = [(0, 1) for _ in range(2 * n_points)]
                
                result = minimize(
                    objective,
                    perturbed.flatten(),
                    method=method,
                    bounds=bounds,
                    options=options
                )
                
                if result.success:
                    optimized_points = result.x.reshape(n_points, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = -objective(result.x)
                    
                    if ratio > current_ratio:
                        current_ratio = ratio
                        current_points = optimized_points.copy()
                        
            except Exception:
                continue
    
    return current_points


def final_refinement(points: np.ndarray) -> np.ndarray:
    """Final aggressive refinement using multiple optimization approaches"""
    n_points = len(points)
    
    # Try more aggressive optimization with tight tolerances
    def objective(params):
        points_array = params.reshape(n_points, 2)
        distances = pdist(points_array)
        if len(distances) == 0:
            return 1e10
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 1e10
        return -min_dist / max_dist
    
    bounds = [(0, 1) for _ in range(2 * n_points)]
    
    # Try multiple optimization approaches with very tight tolerances
    approaches = [
        ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}),
        ('TNC', {'maxiter': 1000, 'ftol': 1e-12}),
        ('SLSQP', {'maxiter': 1000, 'ftol': 1e-12})
    ]
    
    best_points = points.copy()
    best_ratio = compute_min_max_ratio(points)
    
    for method, options in approaches:
        try:
            result = minimize(
                objective,
                points.flatten(),
                method=method,
                bounds=bounds,
                options=options
            )
            
            if result.success:
                refined_points = result.x.reshape(n_points, 2)
                refined_points = np.clip(refined_points, 0, 1)
                refined_ratio = compute_min_max_ratio(refined_points)
                
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
        except Exception:
            continue
    
    return best_points


# EVOLVE-BLOCK-END
