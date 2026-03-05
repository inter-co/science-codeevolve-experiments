# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
from scipy.optimize import differential_evolution
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a multi-start optimization approach with enhanced initialization and robust optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    
    best_ratio = -np.inf
    best_points = None
    
    # Use a more focused set of initial configurations that have shown success
    # We'll use the most promising ones from previous analysis
    initial_configs = [
        _initialize_hexagonal_grid(n),           # Hexagonal tiling - excellent packing
        _initialize_fibonacci_spiral(n),         # Fibonacci - uniform distribution  
        _initialize_golden_ratio(n),             # Golden ratio - good distribution
        _initialize_regular_polygon(n),          # Regular polygon - balanced
        _initialize_optimized_initial(n),        # Enhanced hexagonal with perturbations
        _initialize_voronoi_like(n),             # Voronoi-inspired - good spread
        _initialize_polar_grid(n),               # Polar arrangement - even coverage
    ]
    
    # Add fewer random configurations to save time but maintain diversity
    for i in range(2):
        initial_configs.append(_initialize_random_points(n))
    
    # Run optimization from each initial configuration with multiple methods
    # More aggressive optimization with higher iteration counts
    max_attempts = 20  # Reduced to balance time vs quality
    attempts_made = 0
    
    for i, initial_points in enumerate(initial_configs):
        if attempts_made >= max_attempts:
            break
            
        # Optimize using scipy's minimize with bounds
        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            distances = pdist(points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max == 0:
                return -1e10  # Avoid division by zero
                
            ratio = d_min / d_max
            return -ratio  # Negative because we minimize
        
        # Define bounds: [0,1] x [0,1] for each coordinate
        bounds = [(0, 1) for _ in range(2*n)]
        
        # Use multiple optimization methods with high iteration limits
        methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
        
        for method in methods_to_try:
            if attempts_made >= max_attempts:
                break
            attempts_made += 1
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        objective,
                        initial_points.flatten(),
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10}
                    )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    # Ensure points are within [0,1] bounds
                    optimized_points = np.clip(optimized_points, 0, 1)
                    
                    # Calculate final ratio
                    distances = pdist(optimized_points)
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    
                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
            except Exception:
                continue
    
    # If no good solution found, try global optimization with moderate parameters
    if best_points is None:
        def global_objective(x_flat):
            points = x_flat.reshape(-1, 2)
            distances = pdist(points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max == 0:
                return -1e10
                
            ratio = d_min / d_max
            return -ratio
        
        bounds = [(0, 1) for _ in range(2*n)]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = differential_evolution(
                    global_objective,
                    bounds,
                    maxiter=200,  # Moderate iterations to save time
                    popsize=25,   # Moderate population size
                    tol=1e-10,    # Tight tolerance
                    mutation=(0.5, 1),
                    recombination=0.7,
                    seed=42
                )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                
                distances = pdist(optimized_points)
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 0:
                    ratio = d_min / d_max
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
    
    # If no good solution found, return the best we have
    if best_points is None:
        # Fallback to a good hexagonal arrangement with refined optimization
        fallback_points = _initialize_hexagonal_grid(n)
        # Do one final optimization on the fallback with high iteration count
        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            distances = pdist(points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max == 0:
                return -1e10
                
            ratio = d_min / d_max
            return -ratio
        
        bounds = [(0, 1) for _ in range(2*n)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    fallback_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10}
                )
            
            if result.success:
                best_points = result.x.reshape(-1, 2)
                best_points = np.clip(best_points, 0, 1)
            else:
                best_points = fallback_points
        except Exception:
            best_points = fallback_points
    
    return best_points


def _initialize_fibonacci_spiral(n: int) -> np.ndarray:
    """Initialize points using Fibonacci spiral for good distribution"""
    points = []
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    for i in range(n):
        theta = i * 2 * np.pi / phi
        r = np.sqrt(i) / np.sqrt(n-1) if n > 1 else 0.5
        x = 0.5 + r * np.cos(theta) * 0.4
        y = 0.5 + r * np.sin(theta) * 0.4
        points.append([x, y])
    return np.array(points)


def _initialize_hexagonal_grid(n: int) -> np.ndarray:
    """Initialize points in a hexagonal grid pattern"""
    # Create a hexagonal grid that fits nicely in [0,1] x [0,1]
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    points = []
    spacing_x = 1.0 / max(1, cols - 1)
    spacing_y = 1.0 / max(1, rows - 1)
    
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            # Offset every other row
            x = j * spacing_x
            y = i * spacing_y * np.sqrt(3) / 2
            # Apply offset
            if i % 2 == 1:
                x += spacing_x * 0.5
            points.append([x, y])
    
    # If we have too many points, take first n
    if len(points) > n:
        points = points[:n]
    
    # If we have too few points, fill with random points
    while len(points) < n:
        points.append([np.random.rand(), np.random.rand()])
    
    return np.array(points)


def _initialize_golden_ratio(n: int) -> np.ndarray:
    """Initialize points using golden ratio based approach"""
    points = []
    phi = (1 + np.sqrt(5)) / 2
    
    for i in range(n):
        # Golden ratio based positioning
        angle = i * 2 * np.pi / phi
        radius = np.sqrt(i / (n - 1)) if n > 1 else 0.5
        x = 0.5 + radius * np.cos(angle) * 0.4
        y = 0.5 + radius * np.sin(angle) * 0.4
        points.append([x, y])
    
    return np.array(points)


def _initialize_polar_grid(n: int) -> np.ndarray:
    """Initialize points in a polar grid arrangement"""
    points = []
    # Distribute points evenly in polar coordinates
    for i in range(n):
        angle = 2 * np.pi * i / n
        # Add some variation to avoid degeneracy
        radius = 0.4 * (1.0 - 0.2 * np.random.rand())
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)


def _initialize_regular_polygon(n: int) -> np.ndarray:
    """Initialize points forming a regular polygon with center"""
    points = []
    if n == 1:
        points.append([0.5, 0.5])
    elif n == 2:
        points.extend([[0.3, 0.5], [0.7, 0.5]])
    else:
        # Regular polygon centered at (0.5, 0.5)
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
    return np.array(points)


def _initialize_voronoi_like(n: int) -> np.ndarray:
    """Initialize points using a Voronoi-inspired approach"""
    # Start with a regular grid then perturb
    points = []
    side = int(np.ceil(np.sqrt(n)))
    
    for i in range(side):
        for j in range(side):
            if len(points) >= n:
                break
            x = j / (side - 1) if side > 1 else 0.5
            y = i / (side - 1) if side > 1 else 0.5
            # Add larger perturbation to create better distribution
            x += (np.random.rand() - 0.5) * 0.2
            y += (np.random.rand() - 0.5) * 0.2
            points.append([x, y])
    
    # If we have too many points, take first n
    if len(points) > n:
        points = points[:n]
    
    # If we have too few points, fill with random points
    while len(points) < n:
        points.append([np.random.rand(), np.random.rand()])
    
    # Clip to bounds
    points = np.array(points)
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    points[:, 1] = np.clip(points[:, 1], 0, 1)
    
    return points


def _initialize_concentric_rings(n: int) -> np.ndarray:
    """Initialize points in concentric rings for better coverage"""
    points = []
    
    # Place points in concentric rings
    # Ring 1: 4 points
    for i in range(4):
        angle = i * np.pi/2
        points.append([0.5 + 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle)])
    
    # Ring 2: 8 points  
    for i in range(8):
        angle = i * np.pi/4
        points.append([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
        
    # Ring 3: 4 points
    for i in range(4):
        angle = i * np.pi/2
        points.append([0.5 + 0.6 * np.cos(angle), 0.5 + 0.6 * np.sin(angle)])
    
    # If we have too many points, take first n
    if len(points) > n:
        points = points[:n]
    
    # If we have too few points, fill with random points
    while len(points) < n:
        points.append([np.random.rand(), np.random.rand()])
    
    return np.array(points)


def _initialize_equilateral_triangle_grid(n: int) -> np.ndarray:
    """Initialize points in an equilateral triangle grid pattern"""
    points = []
    
    # Create a triangular lattice pattern
    rows = int(np.ceil(np.sqrt(n * 2)))
    cols = int(np.ceil(n / rows))
    
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            # Offset every other row
            x = j * 0.6 / (cols - 1) if cols > 1 else 0.3
            y = i * 0.5 * np.sqrt(3) / (rows - 1) if rows > 1 else 0.25
            # Apply offset
            if i % 2 == 1:
                x += 0.3 / (cols - 1) if cols > 1 else 0.15
            points.append([x, y])
    
    # If we have too many points, take first n
    if len(points) > n:
        points = points[:n]
    
    # If we have too few points, fill with random points
    while len(points) < n:
        points.append([np.random.rand(), np.random.rand()])
    
    # Clip to bounds
    points = np.array(points)
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    points[:, 1] = np.clip(points[:, 1], 0, 1)
    
    return points


def _initialize_optimized_initial(n: int) -> np.ndarray:
    """Initialize points using a more optimized approach based on known good configurations"""
    # Start with a hexagonal grid and then refine with moderate perturbations
    points = _initialize_hexagonal_grid(n)
    
    # Apply multiple rounds of moderate perturbations
    for _ in range(15):  # Fewer perturbations to save time
        # Add moderate random perturbations
        perturbation_magnitude = 0.03  # Smaller perturbation
        points += (np.random.rand(n, 2) - 0.5) * perturbation_magnitude
        # Keep within bounds
        points = np.clip(points, 0, 1)
    
    return points


def _initialize_random_points(n: int) -> np.ndarray:
    """Initialize points randomly"""
    return np.random.rand(n, 2)


# EVOLVE-BLOCK-END
