# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import random
from numba import jit
import time
from sklearn.cluster import KMeans
from scipy.spatial import SphericalVoronoi
import itertools

@jit(nopython=True)
def compute_distances_jit(points):
    """Compute pairwise distances efficiently using numba"""
    n = points.shape[0]
    distances = np.zeros((n * (n - 1) // 2,))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            distances[idx] = np.sqrt(dx*dx + dy*dy + dz*dz)
            idx += 1
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    def objective(points_flat):
        """Objective function to minimize negative of min/max ratio"""
        points = points_flat.reshape(-1, 3)
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return -d_min / d_max
    
    def objective_fast(points_flat):
        """Faster objective using numba-compiled distance computation"""
        points = points_flat.reshape(-1, 3)
        distances = compute_distances_jit(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return -d_min / d_max
    
    def initialize_points_spherical():
        """Initialize points using spherical arrangement with enhanced distribution"""
        # Use a more sophisticated approach based on known optimal configurations
        # Inspired by research on optimal point distributions on spheres
        
        # Start with a good known configuration - use vertices of polyhedra or 
        # optimized arrangements for 14 points
        
        # Generate points on a sphere using Fibonacci-like method with improved spacing
        n = 14
        points = []
        
        # Golden ratio and related constants
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        golden_angle = 2 * np.pi * (1 - 1/phi)  # golden angle
        
        # Create points with better distribution using Fibonacci spiral with adjustments
        for i in range(n):
            # Adjusted Fibonacci spiral for better distribution
            y = 1 - (i / (n - 1)) * 2  # y from -1 to 1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Apply a more sophisticated angle calculation
            theta = golden_angle * i + np.random.uniform(-0.1, 0.1)  # Add slight randomness
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        points = np.array(points)
        
        # Normalize to unit sphere and scale appropriately
        norms = np.linalg.norm(points, axis=1)
        if np.max(norms) > 0:
            points = points / np.max(norms) * 0.9
            
        return points
    
    def initialize_points_polyhedral():
        """Initialize using polyhedral structures for better symmetry"""
        # Use a construction based on the icosahedron and other polyhedra
        # 14 points can be constructed by taking vertices of an icosahedron and adding points
        
        # Vertices of a regular icosahedron scaled appropriately
        phi = (1 + np.sqrt(5)) / 2
        # Standard icosahedron vertices (normalized)
        vertices = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        vertices = np.array(vertices)
        norms = np.linalg.norm(vertices, axis=1)
        vertices = vertices / norms[:, np.newaxis] * 0.9
        
        # Select 12 vertices and add 2 more points
        selected_vertices = vertices[:12]
        
        # Add two additional points to get 14 total
        # Place them along a diameter to improve the ratio
        additional_points = [[0, 0, 0.8], [0, 0, -0.8]]
        
        points = np.vstack([selected_vertices, additional_points])
        
        # Add small random perturbations to break perfect symmetry
        noise = np.random.normal(0, 0.03, points.shape)
        points = points + noise
        
        return points
    
    def initialize_points_grid():
        """Initialize using grid-based approach for good coverage"""
        # Create a 3D grid pattern with adjustments for better distribution
        points = []
        # Create roughly uniform distribution
        grid_size = 3
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(points) < 14:
                        x = (i / (grid_size - 1)) * 1.8 - 0.9
                        y = (j / (grid_size - 1)) * 1.8 - 0.9
                        z = (k / (grid_size - 1)) * 1.8 - 0.9
                        points.append([x, y, z])
        
        # Trim to 14 points
        points = points[:14]
        points = np.array(points)
        
        # Add small random perturbations
        noise = np.random.normal(0, 0.05, points.shape)
        points = points + noise
        
        return points
    
    def advanced_optimization(initial_points, max_iter=1000):
        """Advanced optimization with multiple strategies and adaptive parameters"""
        best_ratio = -np.inf
        best_points = None
        
        # Strategy 1: Multiple restarts with different optimization approaches
        bounds = [(-0.9, 0.9)] * 42  # 14 points * 3 coordinates
        
        # Try different optimization methods with varying parameters
        strategies = [
            {'method': 'L-BFGS-B', 'options': {'maxiter': max_iter // 3, 'ftol': 1e-8}},
            {'method': 'TNC', 'options': {'maxiter': max_iter // 4, 'ftol': 1e-8}},
            {'method': 'SLSQP', 'options': {'maxiter': max_iter // 3, 'ftol': 1e-8}}
        ]
        
        # Multi-start optimization with adaptive strategy
        for restart in range(15):  # Fewer restarts but better quality
            # Create perturbed starting point
            if restart == 0:
                current_points = initial_points.copy()
            else:
                # Add larger perturbations for later restarts
                perturbation_scale = 0.1 if restart < 8 else 0.15
                current_points = initial_points + np.random.normal(0, perturbation_scale, initial_points.shape)
            
            # Keep points within bounds
            current_points = np.clip(current_points, -0.9, 0.9)
            
            # Try different optimization strategies
            for strategy in strategies:
                try:
                    result = minimize(
                        objective_fast,
                        current_points.flatten(),
                        method=strategy['method'],
                        bounds=bounds,
                        options=strategy['options'],
                        tol=1e-8
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        distances = pdist(optimized_points)
                        if len(distances) > 0:
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            if d_max > 0:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = optimized_points.copy()
                                    # Early stopping if we're getting close to good solutions
                                    if ratio > 0.47:
                                        break
                except Exception:
                    continue
            
            # If we've found a very good solution, stop early
            if best_ratio > 0.485:
                break
        
        # Return best found or initial points
        return best_points if best_points is not None else initial_points
    
    def hybrid_refinement(final_points, max_iter=500):
        """Hybrid refinement approach combining different techniques"""
        # First try to improve with gradient-based optimization
        bounds = [(-0.9, 0.9)] * 42
        
        try:
            # Try L-BFGS-B with full iterations
            result = minimize(
                objective_fast,
                final_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10}
            )
            
            if result.success:
                candidate_points = result.x.reshape(-1, 3)
                distances = pdist(candidate_points)
                if len(distances) > 0:
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    if d_max > 0:
                        ratio = d_min / d_max
                        current_ratio = np.min(pdist(final_points)) / np.max(pdist(final_points))
                        if ratio > current_ratio:
                            return candidate_points
        except Exception:
            pass
            
        return final_points
    
    # Try multiple initialization strategies with better error handling
    initial_points = None
    
    # Strategy 1: Polyhedral initialization (often best for 14 points)
    try:
        initial_points = initialize_points_polyhedral()
    except Exception:
        pass
    
    # Strategy 2: Spherical initialization if first fails
    if initial_points is None:
        try:
            initial_points = initialize_points_spherical()
        except Exception:
            pass
    
    # Strategy 3: Grid initialization if previous fails
    if initial_points is None:
        try:
            initial_points = initialize_points_grid()
        except Exception:
            pass
    
    # Strategy 4: Fallback to simple random initialization
    if initial_points is None:
        initial_points = np.random.uniform(-0.9, 0.9, (14, 3))
    
    # Optimize using advanced method
    start_time = time.time()
    final_points = advanced_optimization(initial_points, max_iter=800)
    
    # Additional refinement if we have time left
    if time.time() - start_time < 50:  # Leave some time for final refinement
        final_points = hybrid_refinement(final_points, max_iter=300)
    
    # Ensure final points are properly bounded
    final_points = np.clip(final_points, -0.9, 0.9)
    
    return final_points


# EVOLVE-BLOCK-END
