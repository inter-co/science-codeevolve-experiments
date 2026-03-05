# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
import time
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust hybrid approach combining geometric constructions, multi-start optimization,
    and careful handling of edge cases to consistently beat the benchmark.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set time limit to ensure we don't exceed 60 seconds
    start_time = time.time()
    timeout = 55  # Leave 5 seconds for final processing
    
    def objective(x_flat):
        """Objective function to maximize min/max distance ratio"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return -1.0
            
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return -1.0
            
        # Return negative ratio (since we want to maximize)
        return -min_dist / max_dist
    
    def _initialize_regular_polygon():
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
    
    def _initialize_hexagonal_arrangement():
        """Initialize points in a hexagonal lattice pattern"""
        # Create a hexagonal arrangement with better spacing
        rows = 4
        cols = 4
        
        points = []
        for i in range(rows):
            for j in range(cols):
                # Offset every other row for better hexagonal packing
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
        points += np.random.normal(0, 0.015, points.shape)
        
        return points
    
    def _initialize_spherical_code():
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
        
        # Stereographic projection from south pole (0,0,-1) - better approach
        points_2d = []
        for pt in rotated_vertices:
            x, y, z = pt
            # Projection formula from south pole (0,0,-1)
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
    
    def _initialize_golden_spiral():
        """Initialize points using golden spiral for good distribution"""
        # Golden spiral arrangement
        points = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(16):
            angle = 2 * math.pi * i / golden_ratio
            radius = 0.4 * math.sqrt(i / 15.0) if i > 0 else 0
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        
        # Add small random perturbations to escape local minima
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        
        return points
    
    def _initialize_improved_grid():
        """Initialize points in a better grid arrangement with more uniform spacing"""
        # Create a grid with better distribution
        points = []
        # Use a 4x4 grid with some strategic perturbations
        for i in range(4):
            for j in range(4):
                # Add more structured perturbations to avoid regular patterns
                x = j + np.random.normal(0, 0.03)
                y = i + np.random.normal(0, 0.03)
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
    
    # Try multiple initialization strategies
    initial_strategies = [
        _initialize_regular_polygon(),      # Regular polygon
        _initialize_hexagonal_arrangement(), # Hexagonal arrangement
        _initialize_spherical_code(),       # Spherical code (best for uniformity)
        _initialize_golden_spiral(),        # Golden spiral
        _initialize_improved_grid(),        # Improved grid
        np.random.uniform(0, 1, (16, 2))    # Random initialization
    ]
    
    best_points = None
    best_ratio = float('-inf')
    
    # Test all initializations with optimization
    for i, initial_points in enumerate(initial_strategies):
        if time.time() - start_time > timeout:
            break
            
        try:
            # Use more aggressive optimization approach
            optimized_points = _optimize_aggressive(initial_points, timeout - (time.time() - start_time))
            ratio = _compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
        except Exception as e:
            continue
    
    # If no optimization worked, return the best initialization directly
    if best_points is None:
        return initial_strategies[0]
    
    # Ensure points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


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
    
    # Use more aggressive optimization approach with multiple methods and restarts
    # More restarts for better exploration
    restarts = min(15, max(5, int(remaining_time / 4)))  # More restarts for better chance
    
    # Keep track of all successful optimizations
    successful_optimizations = []
    
    for restart in range(restarts):
        try:
            # Different optimization methods in sequence with decreasing tolerance
            methods = ['L-BFGS-B', 'TNC', 'SLSQP']
            method_index = restart % len(methods)
            method = methods[method_index]
            
            # Adjust parameters based on restart number for more thorough search
            max_iter = min(2000, max(500, int(remaining_time * 100)))
            
            # Apply optimization with different tolerances for better convergence
            optimized_points = _optimize_single_run(initial_points, method, max_iter=max_iter)
            ratio = _compute_min_max_ratio(optimized_points)
            
            # Store successful optimizations for later evaluation
            if ratio > 0:
                successful_optimizations.append((ratio, optimized_points.copy()))
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    # If we have successful optimizations, pick the best one
    if successful_optimizations:
        best_successful = max(successful_optimizations, key=lambda x: x[0])
        if best_successful[0] > best_ratio:
            best_ratio = best_successful[0]
            best_points = best_successful[1]
    
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
        try:
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
        except:
            pass
    
    # If optimization fails, return the original points
    return points


# EVOLVE-BLOCK-END
