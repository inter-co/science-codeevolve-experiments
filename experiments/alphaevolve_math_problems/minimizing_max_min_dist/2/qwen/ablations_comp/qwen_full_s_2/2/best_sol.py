# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
import math
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical constructions, energy-based initialization,
    and advanced optimization techniques with time management.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set time limit to ensure we don't exceed 60 seconds
    start_time = time.time()
    timeout = 55  # Leave 5 seconds for final processing
    
    # Strategy 1: Golden spiral initialization (inspired by inspiration 2)
    points1 = _initialize_golden_spiral()
    
    # Strategy 2: Hexagonal lattice with perturbations (from inspiration 1)
    points2 = _initialize_hexagonal_arrangement()
    
    # Strategy 3: Regular polygon (circle) with perturbations (from inspiration 1)
    points3 = _initialize_regular_polygon()
    
    # Strategy 4: Grid with noise (from inspiration 1)
    points4 = _initialize_grid_with_noise()
    
    # Strategy 5: Energy-based initialization using potential field model (novel approach)
    points5 = _initialize_energy_based()
    
    # Strategy 6: Spherical code with stereographic projection (from inspiration 2)
    points6 = _initialize_spherical_code()
    
    # Strategy 7: Modified Fibonacci spiral (highly uniform distribution)
    points7 = _initialize_fibonacci_spiral()
    
    # Strategy 8: Random initialization with better spread
    points8 = _initialize_random_better_spread()
    
    # Test all initializations with optimization
    initial_strategies = [points1, points2, points3, points4, points5, points6, points7, points8]
    
    best_points = None
    best_ratio = -float('inf')
    
    for i, initial_points in enumerate(initial_strategies):
        if time.time() - start_time > timeout:
            break
            
        try:
            # Use more aggressive optimization with time-aware parameters
            optimized_points = _optimize_aggressive(initial_points, timeout - (time.time() - start_time))
            ratio = _compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
        except Exception:
            continue
    
    # If no optimization worked, return the best initialization directly
    if best_points is None:
        return initial_strategies[0]
    
    # Ensure points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


def _initialize_golden_spiral() -> np.ndarray:
    """Initialize points using golden spiral for good distribution"""
    n = 16
    points = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    for i in range(n):
        theta = 2 * np.pi * i / phi
        r = np.sqrt(i / (n - 1)) if n > 1 else 0  # Radial distribution
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    
    # Normalize to [0,1] x [0,1] 
    points = np.array(points)
    if len(points) > 0:
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        if x_max > x_min and y_max > y_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        
        # Shift to center and scale appropriately
        points[:, 0] = 0.5 + 0.4 * (points[:, 0] - 0.5)
        points[:, 1] = 0.5 + 0.4 * (points[:, 1] - 0.5)
    
    # Add slight random perturbations to escape local minima
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    
    return points


def _initialize_hexagonal_arrangement() -> np.ndarray:
    """Initialize points in a hexagonal lattice pattern"""
    # Create a hexagonal arrangement
    rows = 4
    cols = 4
    
    points = []
    for i in range(rows):
        for j in range(cols):
            x = j + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            points.append([x, y])
    
    # Take first 16 points
    points = np.array(points[:16])
    
    # Normalize to reasonable scale
    if len(points) > 0:
        # Scale to fit well in unit square
        ranges = np.max(points, axis=0) - np.min(points, axis=0)
        if np.any(ranges > 0):
            points = points / np.max(ranges) * 0.8
        
        # Center around origin
        points = points - np.mean(points, axis=0)
        
        # Shift to [0,1] range
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        if np.any(maxs - mins > 0):
            points = (points - mins) / (maxs - mins) * 0.9 + 0.05
    
    # Add slight random perturbations to escape local minima
    np.random.seed(42)
    points += np.random.normal(0, 0.02, points.shape)
    
    return points


def _initialize_regular_polygon() -> np.ndarray:
    """Initialize points on a regular polygon (circle) with 16 vertices"""
    n = 16
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    # Scale to fit nicely in unit square
    points = points * 0.4 + 0.5  # Center at (0.5, 0.5) with radius 0.4
    
    # Add slight random perturbations to escape local minima
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    
    return points


def _initialize_grid_with_noise() -> np.ndarray:
    """Initialize points in a regular grid pattern with random noise"""
    points = []
    for i in range(4):
        for j in range(4):
            points.append([i/3.0, j/3.0])
    
    points = np.array(points)
    
    # Add small random perturbations
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    
    # Normalize to [0,1] range
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    ranges = maxs - mins
    if np.any(ranges > 0):
        points = (points - mins) / ranges * 0.9 + 0.05
    
    return points


def _initialize_energy_based() -> np.ndarray:
    """Initialize points using an energy-based approach mimicking repulsive forces"""
    # Start with a regular configuration and apply energy minimization concept
    n = 16
    
    # Create initial configuration using a combination of hexagonal and grid
    points = []
    
    # Hexagonal grid with slight randomness
    for i in range(4):
        for j in range(4):
            x = j + (i % 2) * 0.5 + np.random.normal(0, 0.02)
            y = i * np.sqrt(3) / 2 + np.random.normal(0, 0.02)
            points.append([x, y])
    
    points = np.array(points[:16])
    
    # Normalize to fit well in unit square
    if len(points) > 0:
        ranges = np.max(points, axis=0) - np.min(points, axis=0)
        if np.any(ranges > 0):
            points = points / np.max(ranges) * 0.8
        
        # Center around origin
        points = points - np.mean(points, axis=0)
        
        # Shift to [0,1] range
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        if np.any(maxs - mins > 0):
            points = (points - mins) / (maxs - mins) * 0.9 + 0.05
    
    return points


def _initialize_spherical_code() -> np.ndarray:
    """Initialize points using spherical code construction with stereographic projection"""
    # Generate vertices of icosahedron and normalize to unit sphere
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    
    # Vertices of regular icosahedron (normalized)
    vertices = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1],
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1]
    ])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    
    # Apply rotation to create better distribution
    # Use a rotation matrix that maximizes uniformity
    angle = 0.1
    rot_matrix = np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])
    rotated_vertices = vertices @ rot_matrix.T
    
    # Stereographic projection from south pole (0,0,-1)
    points_2d = []
    for pt in rotated_vertices:
        x, y, z = pt
        # Projection formula from south pole (-1,0,0)
        if abs(z + 1) < 1e-10:  # Handle singularity
            continue
        factor = 2 / (1 + z)
        proj_x = x * factor
        proj_y = y * factor
        points_2d.append([proj_x, proj_y])
    
    points_2d = np.array(points_2d[:16])
    
    # Normalize to fit in [0,1] x [0,1]
    if len(points_2d) > 0:
        mins = np.min(points_2d, axis=0)
        maxs = np.max(points_2d, axis=0)
        ranges = maxs - mins
        if np.any(ranges > 0):
            points_2d = (points_2d - mins) / ranges * 0.8 + 0.1
    
    # Add slight random perturbations to escape local minima
    np.random.seed(42)
    points_2d += np.random.normal(0, 0.01, points_2d.shape)
    
    return points_2d


def _initialize_fibonacci_spiral() -> np.ndarray:
    """Initialize points using Fibonacci spiral for highly uniform distribution"""
    n = 16
    points = []
    
    # Fibonacci spiral approach
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        # Angle in radians
        theta = i * 2 * np.pi / golden_ratio
        # Radius (spiral pattern)
        r = np.sqrt(i / (n - 1)) if n > 1 else 0
        # Position in Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    
    # Normalize to [0,1] x [0,1]
    points = np.array(points)
    if len(points) > 0:
        # Normalize to fit in [0,1] range
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        ranges = maxs - mins
        if np.any(ranges > 0):
            points = (points - mins) / ranges * 0.8 + 0.1
    
    # Add slight random perturbations
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    
    return points


def _initialize_random_better_spread() -> np.ndarray:
    """Initialize points with better spread than simple random"""
    np.random.seed(42)
    points = np.random.rand(16, 2)
    
    # Improve spread using a simple clustering avoidance approach
    # This helps avoid very clustered initial configurations
    for i in range(16):
        # Move away from nearby points slightly
        for j in range(i):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < 0.1:  # If too close, move apart
                direction = points[i] - points[j]
                if np.linalg.norm(direction) > 0:
                    points[i] += 0.05 * direction / np.linalg.norm(direction)
    
    # Clip to ensure bounds
    points = np.clip(points, 0, 1)
    
    return points


def _compute_min_max_ratio(points: np.ndarray) -> float:
    """Compute the ratio of minimum to maximum pairwise distances"""
    if len(points) < 2:
        return 0
    
    # Compute pairwise distances
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Handle edge case where all points are coincident
    if max_dist == 0:
        return 0
    
    return min_dist / max_dist


def _optimize_aggressive(initial_points: np.ndarray, remaining_time: float) -> np.ndarray:
    """Apply aggressive optimization with multiple restarts and better strategies"""
    best_points = initial_points.copy()
    best_ratio = _compute_min_max_ratio(best_points)
    
    # Use more aggressive optimization approach
    restarts = min(15, max(5, int(remaining_time / 2)))  # More restarts for better chance
    
    for restart in range(restarts):
        if restart >= 10:  # Limit to avoid excessive time usage
            break
            
        try:
            # Different optimization methods in sequence
            methods = ['L-BFGS-B', 'TNC', 'SLSQP', 'Nelder-Mead']
            if restart < len(methods):
                method = methods[restart]
            else:
                # For later restarts, use a random method or just L-BFGS-B
                method = 'L-BFGS-B'
                
            # Apply optimization with adjusted parameters for time
            max_time_per_run = remaining_time / restarts if restarts > 0 else 1.0
            max_iter = min(3000, max(500, int(max_time_per_run * 300)))
            
            optimized_points = _optimize_single_run(initial_points, method, max_iter=max_iter)
            ratio = _compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    return best_points


def _optimize_single_run(initial_points: np.ndarray, method='L-BFGS-B', max_iter=1000) -> np.ndarray:
    """Apply a single, robust optimization run to find the best configuration"""
    points = initial_points.copy()
    
    # Define bounds for optimization (points must stay in [0,1] x [0,1])
    bounds = [(0, 1), (0, 1)] * len(points)
    
    # Convert to flattened array for scipy optimization
    flat_points = points.flatten()
    
    # Define objective function to maximize min/max ratio
    def objective(flat):
        # Reshape back to 2D array
        reshaped = flat.reshape(-1, 2)
        # Minimize negative of ratio (since scipy minimizes)
        return -_compute_min_max_ratio(reshaped)
    
    # Use specified optimizer with reasonable parameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective, 
            flat_points, 
            method=method, 
            bounds=bounds, 
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
        )
    
    if result.success:
        optimized_points = result.x.reshape(-1, 2)
        optimized_points = np.clip(optimized_points, 0, 1)
        return optimized_points
    
    # If optimization fails, return the original points
    return points


# EVOLVE-BLOCK-END
