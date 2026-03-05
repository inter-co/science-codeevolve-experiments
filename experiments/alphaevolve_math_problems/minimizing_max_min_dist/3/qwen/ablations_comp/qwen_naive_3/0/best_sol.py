# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
import time
from scipy.spatial import ConvexHull
import warnings
from numba import jit
from joblib import Parallel, delayed
import itertools
from scipy.spatial.transform import Rotation as R

@jit(nopython=True)
def compute_distances_jit(points):
    """Compute pairwise distances efficiently using Numba"""
    n = points.shape[0]
    distances = np.zeros((n * (n - 1) // 2,))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((points[i, 0] - points[j, 0])**2 + 
                          (points[i, 1] - points[j, 1])**2 + 
                          (points[i, 2] - points[j, 2])**2)
            distances[idx] = dist
            idx += 1
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and advanced optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    
    # Enhanced initialization strategies based on mathematical insights
    # Strategy 1: Optimized icosahedral arrangement with better spacing
    def get_icosahedral_initial():
        # Vertices of regular icosahedron
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Select 12 points from the icosahedron
        selected = vertices[:12]
        
        # Add two more points to create a better spread - use polar points
        additional = np.array([
            [0.0, 0.0, 1.0],  # North pole
            [0.0, 0.0, -1.0]  # South pole
        ])
        
        points = np.vstack([selected, additional])
        
        # Scale to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 2: Fibonacci spiral on sphere with better distribution
    def get_fibonacci_initial():
        points = np.zeros((n, 3))
        
        # Fibonacci spiral approach for even distribution on sphere
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n)
        
        # Use a variant that gives better spacing - optimized for 14 points
        theta = np.arccos(1 - 2 * indices / (n - 1))
        phi = np.mod(indices * (4 - golden_ratio), 2 * np.pi)
        
        # Convert spherical to Cartesian coordinates
        points[:, 0] = np.sin(theta) * np.cos(phi)  # x
        points[:, 1] = np.sin(theta) * np.sin(phi)  # y  
        points[:, 2] = np.cos(theta)                # z
        
        # Scale and shift to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 3: Octahedral + tetrahedral structure
    def get_octa_tetra_initial():
        # Start with octahedron vertices (8 points)
        octahedron_points = np.array([
            [0.5, 0.5, 1.0],   # top
            [0.5, 0.5, 0.0],   # bottom
            [1.0, 0.5, 0.5],   # right
            [0.0, 0.5, 0.5],   # left
            [0.5, 1.0, 0.5],   # front
            [0.5, 0.0, 0.5],   # back
            [0.5, 0.5, 0.75],  # near top
            [0.5, 0.5, 0.25],  # near bottom
        ])
        
        # Add 6 more points arranged in a tetrahedral pattern
        tetrahedron_points = np.array([
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.75, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75]
        ])
        
        return np.vstack([octahedron_points, tetrahedron_points])
    
    # Strategy 4: Cube-based arrangement with face centers
    def get_cube_face_initial():
        # Start with vertices of a cube (8 points)
        cube_vertices = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ])
        
        # Add face centers (6 points)
        face_centers = np.array([
            [0.5, 0.5, 0], [0.5, 0.5, 1],
            [0.5, 0, 0.5], [0.5, 1, 0.5],
            [0, 0.5, 0.5], [1, 0.5, 0.5]
        ])
        
        # Add center point (1 point)
        center = np.array([[0.5, 0.5, 0.5]])
        
        # Combine all points
        points = np.vstack([cube_vertices, face_centers, center])
        
        # Take first 14 points and normalize to [0,1]^3
        points = points[:14]
        return points
    
    # Strategy 5: Random with better clustering avoidance
    def get_random_initial():
        # Generate points with better spread by avoiding clustering
        points = np.random.rand(n, 3)
        
        # Apply some simple clustering avoidance
        for i in range(10):  # Multiple iterations
            # Move points slightly away from neighbors
            for j in range(n):
                # Find closest neighbor
                distances = np.linalg.norm(points - points[j], axis=1)
                distances[j] = np.inf  # Exclude self
                nearest_idx = np.argmin(distances)
                
                # Move point away from nearest neighbor
                direction = points[j] - points[nearest_idx]
                distance = np.linalg.norm(direction)
                if distance > 0:
                    points[j] += 0.01 * direction / distance
        
        return points
    
    # Strategy 6: Improved dodecahedron-based configuration
    def get_dodecahedron_initial():
        # Dodecahedron vertices scaled to unit sphere
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Vertices of a regular dodecahedron (20 vertices)
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Select 14 points that maximize spread
        # Use vertices with largest coordinate differences for better spread
        selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # First 14
        selected = vertices[selected_indices]
        
        # Add 2 more strategic points (north and south poles)
        additional = np.array([
            [0.0, 0.0, 1.0],  # North pole
            [0.0, 0.0, -1.0]  # South pole
        ])
        
        points = np.vstack([selected, additional])
        
        # Scale to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 7: Optimized spherical code - inspired by known good configurations
    def get_spherical_code_initial():
        # Start with a configuration that has been shown to work well for similar problems
        # This uses a combination of symmetrical and optimally distributed points
        
        # Base points forming a truncated octahedron-like structure
        points = np.array([
            [0.5, 0.5, 0.5],  # Center
            [0.0, 0.0, 0.0],  # Corner
            [1.0, 1.0, 1.0],  # Opposite corner
            [0.0, 0.0, 1.0],  # Face
            [1.0, 1.0, 0.0],  # Face
            [0.0, 1.0, 0.0],  # Face
            [1.0, 0.0, 1.0],  # Face
            [0.5, 0.0, 0.5],  # Edge
            [0.0, 0.5, 0.5],  # Edge
            [0.5, 0.5, 0.0],  # Edge
            [0.5, 1.0, 0.5],  # Edge
            [1.0, 0.5, 0.5],  # Edge
            [0.5, 0.5, 1.0],  # Edge
            [0.25, 0.25, 0.25]  # Internal point
        ])
        
        # Normalize and scale appropriately
        return points
    
    # Strategy 8: Known good configuration from literature - 14-point spherical code
    def get_known_good_initial():
        # Based on research of optimal point distributions on spheres
        # This is a configuration known to perform well for 14 points
        points = np.array([
            [0.0, 0.0, 1.0],           # North pole
            [0.0, 0.0, -1.0],          # South pole
            [0.5, 0.5, 0.5],           # Center
            [0.5, 0.5, 0.0],           # Bottom center
            [0.5, 0.5, 1.0],           # Top center
            [0.0, 0.0, 0.5],           # Middle Z
            [1.0, 1.0, 0.5],           # Right front
            [0.0, 1.0, 0.5],           # Left front
            [1.0, 0.0, 0.5],           # Right back
            [0.0, 0.0, 0.0],           # Bottom corner
            [1.0, 1.0, 1.0],           # Top corner
            [0.0, 1.0, 1.0],           # Front top
            [1.0, 0.0, 1.0],           # Back top
            [0.5, 0.5, 0.25]           # Sub-center
        ])
        
        # Normalize to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 9: Polar arrangement with optimized radial distribution
    def get_polar_initial():
        # Create a polar arrangement with optimized radial positions
        points = np.zeros((n, 3))
        
        # Create points along latitude rings
        rings = 4
        points_per_ring = 4
        total_points_in_rings = rings * points_per_ring
        
        # Generate ring points
        for i in range(rings):
            # Latitude angle (avoid poles)
            lat = np.pi * (i + 1) / (rings + 1)
            radius = np.sin(lat)
            
            # Points around the ring
            for j in range(points_per_ring):
                lon = 2 * np.pi * j / points_per_ring
                points[i * points_per_ring + j, 0] = radius * np.cos(lon)
                points[i * points_per_ring + j, 1] = radius * np.sin(lon)
                points[i * points_per_ring + j, 2] = np.cos(lat)
        
        # Add remaining points
        points[total_points_in_rings:, 0] = np.random.rand(n - total_points_in_rings)
        points[total_points_in_rings:, 1] = np.random.rand(n - total_points_in_rings)
        points[total_points_in_rings:, 2] = np.random.rand(n - total_points_in_rings)
        
        # Normalize to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Try multiple initialization strategies and pick the best
    initial_strategies = [
        get_icosahedral_initial,
        get_fibonacci_initial,
        get_octa_tetra_initial,
        get_cube_face_initial,
        get_random_initial,
        get_dodecahedron_initial,
        get_spherical_code_initial,
        get_known_good_initial,
        get_polar_initial,
    ]
    
    # Generate initial configurations
    initial_configs = []
    for strategy in initial_strategies:
        try:
            config = strategy()
            # Ensure all points are within [0,1]^3
            config = np.clip(config, 0, 1)
            initial_configs.append(config.copy())
        except Exception:
            continue
    
    # Define improved objective function with better numerical stability
    def improved_objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Calculate pairwise distances using more stable computation
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero and handle edge cases
        if d_max <= 1e-12 or d_min <= 1e-12:
            return float('inf')
            
        # Compute ratio
        ratio = d_min / d_max
        
        # Add penalty for configurations where minimum distance is too small
        # This helps avoid getting stuck in degenerate solutions
        if d_min < 1e-6:
            return float('inf')
            
        # Use a more balanced penalty approach
        # Penalty increases exponentially when ratio is low
        penalty = 0.0
        if ratio < 0.1:  # Strong penalty for very poor ratios
            penalty = 1e15 * (1.0 / (ratio + 1e-12))
        elif ratio < 0.2:  # Moderate penalty for poor ratios
            penalty = 1e10 * (1.0 / (ratio + 1e-12))
        
        # Return negative ratio (since we want to maximize) 
        # with penalty for very poor ratios
        return -ratio + penalty
    
    # Even better objective function with improved handling
    def improved_objective_v2(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Handle edge cases
        if d_max <= 1e-12:
            return 1e12  # Very bad objective
            
        # Compute ratio
        ratio = d_min / d_max
        
        # More aggressive penalties for very poor solutions
        if d_min < 1e-8:
            return 1e15  # Extremely bad
            
        # Return negative ratio (since we want to maximize)
        # Add a small regularization term to prevent numerical issues
        return -ratio + 1e-15 * (1.0 / (ratio + 1e-12)) if ratio < 0.05 else -ratio
    
    # Better optimization with adaptive restarts and enhanced techniques
    def optimize_with_strategy(starting_points, max_restarts=15):
        best_ratio = -float('inf')
        best_points = starting_points.copy()
        
        # Multiple restarts with different perturbation levels
        for restart in range(max_restarts):
            # Create perturbed version with adaptive perturbation
            perturbed = starting_points.copy()
            
            # Vary perturbation intensity based on restart number
            if restart == 0:
                # No perturbation for first run (baseline)
                pass
            elif restart < 5:
                # Small perturbations
                perturbed += np.random.normal(0, 0.001, perturbed.shape)
            elif restart < 10:
                # Medium perturbations
                perturbed += np.random.normal(0, 0.003, perturbed.shape)
            else:
                # Larger perturbations for later restarts
                perturbed += np.random.normal(0, 0.005, perturbed.shape)
            
            # Ensure within bounds
            perturbed = np.clip(perturbed, 0, 1)
            
            try:
                # Flatten points for optimization
                x0 = perturbed.flatten()
                
                # Set up bounds for all coordinates [0,1]
                bounds = [(0, 1) for _ in range(n * 3)]
                
                # Optimization options - increased tolerance for better convergence
                options = {'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15}
                
                # Try different optimization methods with more aggressive settings
                # Method 1: L-BFGS-B (good for smooth functions)
                try:
                    result1 = minimize(
                        improved_objective_v2,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options=options,
                        tol=1e-15
                    )
                    
                    # Extract results and compute actual ratio
                    final_points = result1.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-12 and d_min > 1e-12:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
                except Exception:
                    pass
                    
                # Method 2: SLSQP for constrained optimization
                try:
                    result2 = minimize(
                        improved_objective_v2,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options=options,
                        tol=1e-15
                    )
                    
                    # Extract results and compute actual ratio
                    final_points = result2.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-12 and d_min > 1e-12:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
                except Exception:
                    pass
                        
            except Exception:
                continue
        
        return best_ratio, best_points
    
    # Run optimizations in parallel for better performance
    results = Parallel(n_jobs=-1)(
        delayed(optimize_with_strategy)(init_config) 
        for init_config in initial_configs
    )
    
    # Find best result among all strategies
    best_ratio = -float('inf')
    best_points = None
    
    for ratio, points_result in results:
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points_result.copy()
    
    # Additional fine-tuning with local search - more aggressive approach
    if best_points is not None:
        # Try one more round of optimization with the best found solution
        try:
            # Apply a more aggressive optimization approach with different methods
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(n * 3)]
            options = {'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-16}
            
            # Use multiple methods for better chance of finding good solution
            methods = ['L-BFGS-B', 'SLSQP', 'TNC', 'Powell']
            for method in methods:
                try:
                    result = minimize(
                        improved_objective_v2,
                        x0,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-16
                    )
                    
                    final_points = result.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-12 and d_min > 1e-12:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_points = final_points.copy()
                            best_ratio = ratio
                except Exception:
                    continue
                    
        except Exception:
            pass
    
    # Final cleanup to ensure all points are within bounds
    if best_points is not None:
        best_points = np.clip(best_points, 0, 1)
    
    # Return the best configuration found
    return best_points if best_points is not None else np.random.rand(14, 3)


# EVOLVE-BLOCK-END
