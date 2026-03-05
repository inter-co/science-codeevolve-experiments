# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical constructions, energy-based initialization,
    and advanced optimization techniques with focus on finding high-quality solutions.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set time limit to ensure we don't exceed 60 seconds
    start_time = time.time()
    timeout = 55  # Leave 5 seconds for final processing
    
    # Strategy 1: Known good configuration - vertices of a regular 16-gon (inspiration 1)
    points1 = _initialize_regular_polygon()
    
    # Strategy 2: Hexagonal arrangement (very good for packing) (inspiration 1)
    points2 = _initialize_hexagonal_arrangement()
    
    # Strategy 3: Spherical code approach (very promising for uniform distribution) (inspiration 2)
    points3 = _initialize_spherical_code()
    
    # Strategy 4: Improved grid with better spacing (inspiration 1)
    points4 = _initialize_improved_grid()
    
    # Strategy 5: Random but structured initialization (inspiration 1)
    points5 = _initialize_structured_random()
    
    # Strategy 6: Energy-based approach with force simulation (inspiration 2)
    points6 = _initialize_energy_simulation()
    
    # Strategy 7: Fibonacci spiral approach (highly structured) (inspiration 2)
    points7 = _initialize_fibonacci_spiral()
    
    # Strategy 8: Modified golden spiral (inspiration 2)
    points8 = _initialize_modified_golden_spiral()
    
    # Test all initializations with optimization
    initial_strategies = [points1, points2, points3, points4, points5, points6, points7, points8]
    
    best_points = None
    best_ratio = -float('inf')
    
    for i, initial_points in enumerate(initial_strategies):
        if time.time() - start_time > timeout:
            break
            
        try:
            # Use more aggressive optimization with more restarts for better results
            optimized_points = _optimize_aggressive(initial_points, timeout - (time.time() - start_time))
            ratio = _compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
        except Exception as e:
            warnings.warn(f"Strategy {i} failed: {str(e)}")
            continue
    
    # If no optimization worked, return the best initialization directly
    if best_points is None:
        return initial_strategies[0]
    
    # Ensure points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


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


def _initialize_spherical_code() -> np.ndarray:
    """Initialize points using spherical code construction with stereographic projection"""
    # Generate vertices of icosahedron and normalize to unit sphere
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
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
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
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


def _initialize_improved_grid() -> np.ndarray:
    """Initialize points in a better grid arrangement"""
    # Create a more evenly spaced grid with better distribution
    points = []
    for i in range(4):
        for j in range(4):
            # Add jitter to create better distribution
            x = j + np.random.normal(0, 0.05)
            y = i + np.random.normal(0, 0.05)
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


def _initialize_structured_random() -> np.ndarray:
    """Initialize points with structured randomness"""
    # Use a more sophisticated approach that maintains some structure
    points = []
    
    # Distribute points in a way that avoids clustering
    for i in range(4):
        for j in range(4):
            # Add structured randomness
            x = j + (i % 2) * 0.25 + np.random.uniform(-0.1, 0.1)
            y = i * 0.5 + np.random.uniform(-0.1, 0.1)
            points.append([x, y])
    
    points = np.array(points[:16])
    
    # Normalize and adjust
    if len(points) > 0:
        ranges = np.max(points, axis=0) - np.min(points, axis=0)
        if np.any(ranges > 0):
            points = points / np.max(ranges) * 0.8
        
        # Center and shift
        points = points - np.mean(points, axis=0)
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        if np.any(maxs - mins > 0):
            points = (points - mins) / (maxs - mins) * 0.9 + 0.05
    
    return points


def _initialize_energy_simulation() -> np.ndarray:
    """Initialize points using energy-based simulation approach"""
    # Start with a hexagonal arrangement and simulate repulsion
    points = []
    for i in range(4):
        for j in range(4):
            x = j + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
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
    
    # Add small random perturbations
    np.random.seed(42)
    points += np.random.normal(0, 0.02, points.shape)
    
    return points


def _initialize_fibonacci_spiral() -> np.ndarray:
    """Initialize points using Fibonacci spiral for excellent distribution"""
    n = 16
    points = []
    
    # Fibonacci spiral approach with improved scaling
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in range(n):
        # Angular position
        theta = i * 2 * np.pi / golden_ratio
        
        # Radial position (spiral)
        r = np.sqrt(i / (n - 1)) if n > 1 else 0
        
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
        
        # Scale and shift to center
        points[:, 0] = 0.5 + 0.4 * (points[:, 0] - 0.5)
        points[:, 1] = 0.5 + 0.4 * (points[:, 1] - 0.5)
    
    # Add slight random perturbations
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    
    return points


def _initialize_modified_golden_spiral() -> np.ndarray:
    """Initialize points using modified golden spiral with better normalization"""
    n = 16
    points = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    for i in range(n):
        theta = 2 * np.pi * i / phi
        r = np.sqrt(i / (n - 1)) if n > 1 else 0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    
    # Normalize to [0,1] x [0,1] 
    points = np.array(points)
    if len(points) > 0:
        # Find bounding box
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        # Avoid division by zero
        if x_max > x_min and y_max > y_min:
            # Normalize to [0,1] range
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        
        # Scale to fit nicely in unit square
        points[:, 0] = 0.8 * points[:, 0] + 0.1
        points[:, 1] = 0.8 * points[:, 1] + 0.1
    
    # Add slight random perturbations to escape local minima
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    
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
    restarts = min(10, max(3, int(remaining_time / 3)))  # More restarts for better chance
    
    for restart in range(restarts):
        try:
            # Different optimization methods in sequence
            methods = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']
            if restart < len(methods):
                method = methods[restart]
            else:
                # For later restarts, use a random method or just L-BFGS-B
                method = 'L-BFGS-B'
                
            # Apply optimization with adjusted parameters for time
            max_time_per_run = remaining_time / restarts if restarts > 0 else 1.0
            max_iter = min(2000, max(500, int(max_time_per_run * 200)))
            
            optimized_points = _optimize_single_run(initial_points, method, max_iter=max_iter)
            ratio = _compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            warnings.warn(f"Optimization restart {restart} failed: {str(e)}")
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
