# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
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
    
    # Strategy 1: Known good configuration based on sphere packing principles
    def get_sphere_packing_initial():
        # Start with vertices of a regular icosahedron
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
        
        # Add 2 more points for better distribution (north and south poles)
        additional = np.array([
            [0.0, 0.0, 1.0],  # North pole
            [0.0, 0.0, -1.0]  # South pole
        ])
        
        points = np.vstack([selected, additional])
        
        # Scale to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 2: Improved fibonacci spiral on sphere
    def get_improved_fibonacci_initial():
        points = np.zeros((n, 3))
        
        # Use a more uniform distribution for 14 points
        indices = np.arange(n)
        phi = np.arccos(1 - 2 * indices / (n - 1))
        golden_ratio = (1 + np.sqrt(5)) / 2
        theta = np.mod(indices * 2 * np.pi / golden_ratio, 2 * np.pi)
        
        # Convert spherical to Cartesian coordinates
        points[:, 0] = np.sin(phi) * np.cos(theta)  # x
        points[:, 1] = np.sin(phi) * np.sin(theta)  # y  
        points[:, 2] = np.cos(phi)                # z
        
        # Scale and shift to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 3: Octahedral plus strategic points
    def get_octahedral_plus_initial():
        # Octahedron vertices
        octahedron_points = np.array([
            [0.5, 0.5, 1.0],   # top
            [0.5, 0.5, 0.0],   # bottom
            [1.0, 0.5, 0.5],   # right
            [0.0, 0.5, 0.5],   # left
            [0.5, 1.0, 0.5],   # front
            [0.5, 0.0, 0.5],   # back
        ])
        
        # Add 8 more points arranged to maximize spread
        additional_points = np.array([
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.75, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.5, 0.5, 0.5]
        ])
        
        points = np.vstack([octahedron_points, additional_points])
        return points
    
    # Strategy 4: Cube with corner adjustments
    def get_cube_adjusted_initial():
        # Start with vertices of a cube
        cube_vertices = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ])
        
        # Add face centers
        face_centers = np.array([
            [0.5, 0.5, 0], [0.5, 0.5, 1],
            [0.5, 0, 0.5], [0.5, 1, 0.5],
            [0, 0.5, 0.5], [1, 0.5, 0.5]
        ])
        
        # Add center point
        center = np.array([[0.5, 0.5, 0.5]])
        
        # Combine all points
        points = np.vstack([cube_vertices, face_centers, center])
        
        # Take first 14 points and normalize to [0,1]^3
        points = points[:14]
        return points
    
    # Strategy 5: Modified icosahedral approach
    def get_modified_icosahedral_initial():
        # Start with icosahedral points
        points = get_sphere_packing_initial()
        
        # Perturb points slightly to improve spread
        perturbation_magnitude = 0.01
        for i in range(len(points)):
            # Apply small random perturbations
            perturbation = np.random.normal(0, perturbation_magnitude, 3)
            points[i] += perturbation
            
        # Ensure all points are within [0,1]^3
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 6: Concentric spheres approach with better spacing
    def get_concentric_initial():
        # Place points on two concentric spheres
        points = np.zeros((n, 3))
        
        # First 8 points on outer sphere (like octahedron with more spread)
        outer_points = np.array([
            [0.5, 0.5, 1.0], [0.5, 0.5, 0.0],
            [1.0, 0.5, 0.5], [0.0, 0.5, 0.5],
            [0.5, 1.0, 0.5], [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.8], [0.5, 0.5, 0.2]
        ])
        
        # Remaining 6 points on inner sphere with better distribution
        inner_points = np.array([
            [0.3, 0.3, 0.7], [0.3, 0.7, 0.3],
            [0.7, 0.3, 0.3], [0.7, 0.7, 0.7],
            [0.3, 0.7, 0.7], [0.7, 0.3, 0.7]
        ])
        
        points[:8] = outer_points
        points[8:] = inner_points
        
        # Apply small perturbations to improve distribution
        for i in range(n):
            if i < 8:
                # Perturb outer points more
                points[i] += np.random.normal(0, 0.005, 3)
            else:
                # Perturb inner points less
                points[i] += np.random.normal(0, 0.002, 3)
        
        # Ensure all points are within [0,1]^3
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 7: Golden ratio-based arrangement
    def get_golden_ratio_initial():
        # Create points along golden ratio spiral in 3D
        points = np.zeros((n, 3))
        
        # Generate points using golden angle spiral in 3D
        for i in range(n):
            # Golden angle in radians
            golden_angle = 2.399963229728653  # ~4π/(1+√5)
            
            # Spherical coordinates
            theta = golden_angle * i
            phi = np.arccos(1 - 2 * i / (n - 1))
            
            # Convert to Cartesian
            points[i, 0] = np.sin(phi) * np.cos(theta)
            points[i, 1] = np.sin(phi) * np.sin(theta)
            points[i, 2] = np.cos(phi)
        
        # Normalize and scale to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 8: Simple hexagonal close-packed arrangement
    def get_hcp_initial():
        # Create a simple HCP-like arrangement
        points = np.zeros((n, 3))
        
        # Place points in layers
        layer_count = 3
        points_per_layer = n // layer_count
        
        # Layer 1: center point
        points[0] = [0.5, 0.5, 0.5]
        
        # Layer 2: 6 points around the center
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles
        for i in range(6):
            points[i+1] = [0.5 + 0.3*np.cos(angles[i]), 0.5 + 0.3*np.sin(angles[i]), 0.3]
        
        # Layer 3: 7 points in a hexagon pattern
        for i in range(7):
            points[i+7] = [0.5 + 0.3*np.cos(angles[i]), 0.5 + 0.3*np.sin(angles[i]), 0.7]
        
        # Fill remaining positions randomly but keep some structure
        for i in range(14):
            if i >= len(points):
                points[i] = np.random.rand(3)
        
        # Ensure all points are within [0,1]^3
        points = np.clip(points, 0, 1)
        return points
    
    # Try multiple initialization strategies and pick the best
    initial_strategies = [
        get_sphere_packing_initial,
        get_improved_fibonacci_initial,
        get_octahedral_plus_initial,
        get_cube_adjusted_initial,
        get_modified_icosahedral_initial,
        get_concentric_initial,
        get_golden_ratio_initial,
        get_hcp_initial
    ]
    
    # Generate initial configurations
    initial_configs = []
    for strategy in initial_strategies:
        try:
            config = strategy()
            # Ensure all points are within [0,1]^3
            config = np.clip(config, 0, 1)
            initial_configs.append(config.copy())
        except Exception as e:
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
        if d_max <= 1e-15 or d_min <= 1e-15:
            return float('inf')
            
        # Compute ratio - use log scale to prevent numerical issues
        ratio = d_min / d_max
        
        # Add penalty for configurations with very small distances to avoid local minima
        penalty = 0.0
        if d_min < 0.001:  # Penalty for very small distances
            penalty = 1e10 * (1.0 / (d_min + 1e-12))  # Stronger penalty for small distances
        
        # Return negative ratio (since we want to maximize) 
        # with penalty for very small distances
        return -ratio + penalty
    
    # Better optimization with adaptive restarts and improved methods
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
                
                # Optimization options - increased tolerance for speed
                options = {'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
                
                # Try different optimization methods with better parameters
                # Method 1: L-BFGS-B (good for smooth functions)
                try:
                    result1 = minimize(
                        improved_objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options=options,
                        tol=1e-14
                    )
                    
                    # Extract results and compute actual ratio
                    final_points = result1.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-15 and d_min > 1e-15:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
                except Exception:
                    pass
                    
                # Method 2: SLSQP for constrained optimization
                try:
                    result2 = minimize(
                        improved_objective,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options=options,
                        tol=1e-14
                    )
                    
                    # Extract results and compute actual ratio
                    final_points = result2.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-15 and d_min > 1e-15:
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
    results = Parallel(n_jobs=-1, timeout=55)(
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
    
    # Additional fine-tuning with local search
    if best_points is not None:
        # Try one more round of optimization with the best found solution
        try:
            # Apply a more aggressive optimization approach
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(n * 3)]
            options = {'maxiter': 350, 'ftol': 1e-15, 'gtol': 1e-15}
            
            # Use multiple methods for better chance of finding good solution
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            for method in methods:
                try:
                    result = minimize(
                        improved_objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-15
                    )
                    
                    final_points = result.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-15 and d_min > 1e-15:
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
