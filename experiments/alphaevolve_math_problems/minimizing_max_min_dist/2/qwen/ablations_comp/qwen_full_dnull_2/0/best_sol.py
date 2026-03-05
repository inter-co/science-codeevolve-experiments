# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust hybrid approach combining multiple initialization strategies, global optimization, 
    and comprehensive local refinement.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Strategy 1: Multiple initialization strategies to find good starting points
    initial_strategies = [
        generate_hexagonal_grid,
        generate_fibonacci_spiral,
        generate_regular_polygon,
        generate_perturbed_grid,
        generate_improved_hexagonal
    ]
    
    best_points = None
    best_ratio = -float('inf')
    
    # Try multiple initialization approaches and pick the best
    for strategy in initial_strategies:
        try:
            points = strategy()
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
        except Exception:
            continue
    
    # If no good initial configuration found, fall back to hexagonal grid
    if best_points is None:
        best_points = generate_hexagonal_grid()
    
    # Strategy 2: Robust global optimization using differential evolution
    # This is particularly effective for this type of problem
    points = optimize_with_differential_evolution(best_points)
    
    # Strategy 3: Aggressive local optimization with multiple restarts
    points = aggressive_local_refinement(points)
    
    # Strategy 4: Final verification and fine-tuning
    points = final_verification(points)
    
    return points


def generate_hexagonal_grid() -> np.ndarray:
    """Generate initial points using a hexagonal grid pattern"""
    points = []
    
    # Hexagonal pattern parameters
    spacing = 1.0
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
        # Add additional points using a spiral pattern
        for i in range(16 - len(points)):
            angle = i * 0.5
            radius = 0.3 + i * 0.05
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
    
    points = np.array(points[:16])
    
    # Add small random perturbations to break symmetries
    noise = np.random.normal(0, 0.01, points.shape)
    points = points + noise
    points = np.clip(points, 0, 1)
    
    return points


def generate_improved_hexagonal() -> np.ndarray:
    """Generate an improved hexagonal grid with better spacing for 16 points"""
    points = []
    
    # More careful hexagonal spacing for exactly 16 points
    # Arrange in 4 rows with 4 columns, adjusting for optimal spacing
    spacing_x = 0.8 / 3  # Space between points horizontally  
    spacing_y = 0.8 * math.sqrt(3) / 6  # Space vertically (hexagonal spacing)
    
    for i in range(4):
        for j in range(4):
            x = 0.1 + j * spacing_x
            y = 0.1 + i * spacing_y
            
            # Adjust for hexagonal pattern - alternate rows
            if i % 2 == 1:
                x += spacing_x / 2
            
            points.append([x, y])
    
    points = np.array(points[:16])
    
    # Add small random perturbations to break symmetries
    noise = np.random.normal(0, 0.01, points.shape)
    points = points + noise
    points = np.clip(points, 0, 1)
    
    return points


def generate_fibonacci_spiral() -> np.ndarray:
    """Generate points using Fibonacci spiral pattern"""
    points = []
    n = 16
    golden_ratio = (1 + math.sqrt(5)) / 2
    
    for i in range(n):
        # Fibonacci spiral on sphere (projected to 2D)
        phi = math.acos(-1 + (2 * i) / (n - 1))
        theta = math.sqrt(n * math.pi) * phi
        
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        
        # Project to 2D and scale to unit square
        points.append([0.5 + 0.4 * x, 0.5 + 0.4 * y])
    
    points = np.array(points)
    return np.clip(points, 0, 1)


def generate_regular_polygon() -> np.ndarray:
    """Generate points in a regular polygon pattern"""
    points = []
    
    # Create points around a circle
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    radius = 0.4
    
    for i in range(16):
        angle = angles[i]
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        points.append([x, y])
    
    points = np.array(points)
    
    # Add small perturbations to avoid degenerate cases
    noise = np.random.normal(0, 0.02, points.shape)
    points = points + noise
    points = np.clip(points, 0, 1)
    
    return points


def generate_perturbed_grid() -> np.ndarray:
    """Generate a structured grid with noise"""
    points = []
    for i in range(4):
        for j in range(4):
            x = (j + 0.5) / 4.0 + (np.random.rand() - 0.5) * 0.1
            y = (i + 0.5) / 4.0 + (np.random.rand() - 0.5) * 0.1
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
    
    # Use differential evolution with tuned parameters for better convergence
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=250,
            popsize=30,
            mutation=(0.5, 1),
            recombination=0.8,
            seed=42,
            disp=False,
            tol=1e-9
        )
        
        if result.success:
            optimized_points = result.x.reshape(n_points, 2)
            return np.clip(optimized_points, 0, 1)
    except Exception:
        pass
    
    return initial_points.copy()


def aggressive_local_refinement(initial_points: np.ndarray) -> np.ndarray:
    """Apply aggressive local optimization with multiple restarts and methods"""
    n_points = len(initial_points)
    current_points = initial_points.copy()
    current_ratio = compute_min_max_ratio(current_points)
    
    # Multiple aggressive optimization attempts
    for attempt in range(15):
        # Slightly perturb current solution
        perturbed = current_points + np.random.normal(0, 0.005, current_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        
        # Use multiple optimization methods with aggressive settings
        methods_and_options = [
            ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}),
            ('TNC', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}),
            ('SLSQP', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        ]
        
        for method, options in methods_and_options:
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


def final_verification(initial_points: np.ndarray) -> np.ndarray:
    """Final verification and fine-tuning"""
    n_points = len(initial_points)
    current_points = initial_points.copy()
    current_ratio = compute_min_max_ratio(current_points)
    
    # Try a few more specialized optimizations
    try:
        def final_objective(params):
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
        
        # Try L-BFGS-B with very tight tolerance
        result = minimize(
            final_objective,
            current_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result.success:
            refined_points = result.x.reshape(n_points, 2)
            refined_points = np.clip(refined_points, 0, 1)
            # Only accept if it improves the ratio significantly
            if compute_min_max_ratio(refined_points) > current_ratio * 1.0001:
                current_points = refined_points
                
    except Exception:
        pass
    
    return current_points


# EVOLVE-BLOCK-END
