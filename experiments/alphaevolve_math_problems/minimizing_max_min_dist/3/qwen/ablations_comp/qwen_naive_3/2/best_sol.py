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
from scipy.spatial import SphericalVoronoi
import math

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
    
    # Strategy 1: Optimized spherical code based on known constructions
    def get_spherical_code_initial():
        # Generate points on sphere using a variant of the icosahedral construction
        # This uses a method inspired by spherical codes and equiangular lines
        
        # First, create a good starting configuration
        # Based on known constructions for 14 points
        points = np.array([
            [0.0, 0.0, 1.0],           # North pole
            [0.0, 0.0, -1.0],          # South pole
            
            # Create a set of points distributed around the equator
            [0.70710678, 0.0, 0.70710678],  # Equatorial points
            [0.0, 0.70710678, 0.70710678],
            [-0.70710678, 0.0, 0.70710678],
            [0.0, -0.70710678, 0.70710678],
            [0.70710678, 0.0, -0.70710678],
            [0.0, 0.70710678, -0.70710678],
            [-0.70710678, 0.0, -0.70710678],
            [0.0, -0.70710678, -0.70710678],
            
            # Additional points for better coverage
            [0.5, 0.5, 0.5],           # Octant point
            [0.5, -0.5, 0.5],          # Octant point
            [-0.5, 0.5, 0.5],          # Octant point
            [-0.5, -0.5, 0.5]          # Octant point
        ])
        
        # Normalize to unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Scale to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 2: Fibonacci-based spherical distribution with refinement
    def get_fibonacci_sphere_initial():
        points = np.zeros((n, 3))
        
        # Use a more sophisticated Fibonacci-like approach for better distribution
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n)
        
        # Modified Fibonacci approach that works well for odd numbers
        phi = np.arccos(1 - 2 * indices / (n - 1))  # Polar angle
        theta = np.mod(indices * (4 * np.pi) / golden_ratio, 2 * np.pi)  # Azimuthal angle
        
        # Convert spherical to Cartesian coordinates
        points[:, 0] = np.sin(phi) * np.cos(theta)  # x
        points[:, 1] = np.sin(phi) * np.sin(theta)  # y  
        points[:, 2] = np.cos(phi)                # z
        
        # Scale and shift to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 3: Optimized icosahedral plus refinement
    def get_icosahedral_refined_initial():
        # Start with icosahedral vertices
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # 12 vertices of icosahedron (normalized)
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Add some additional points that help with spacing
        # Add midpoints of edges to improve distribution
        additional_points = []
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                # Check if these vertices form an edge of icosahedron
                # (distance should be sqrt(2*phi^2 + 2*phi) for regular icosahedron)
                dist = np.linalg.norm(vertices[i] - vertices[j])
                if abs(dist - np.sqrt(2 * (phi**2 + phi))) < 0.1:
                    # Add midpoint
                    midpoint = (vertices[i] + vertices[j]) / 2
                    additional_points.append(midpoint)
        
        # Fill remaining points with random but distributed selection
        if len(additional_points) < 14 - len(vertices):
            extra_points = np.random.rand(14 - len(vertices) - len(additional_points), 3)
            additional_points.extend(extra_points)
        
        # Combine and normalize to [0,1]^3
        points = np.vstack([vertices, additional_points[:14-len(vertices)]])
        points = (points + 1) / 2
        return points
    
    # Strategy 4: Octahedral + Tetrahedral structure with better spacing
    def get_octa_tetra_initial():
        # Start with octahedron vertices
        octahedron_points = np.array([
            [0.5, 0.5, 1.0],   # top
            [0.5, 0.5, 0.0],   # bottom
            [1.0, 0.5, 0.5],   # right
            [0.0, 0.5, 0.5],   # left
            [0.5, 1.0, 0.5],   # front
            [0.5, 0.0, 0.5],   # back
        ])
        
        # Add points in tetrahedral positions
        tetrahedron_points = np.array([
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.75, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.5, 0.5, 0.5]
        ])
        
        # Combine and add more points for better distribution
        points = np.vstack([octahedron_points, tetrahedron_points])
        
        # Add remaining points with more careful spacing
        remaining = 14 - len(points)
        if remaining > 0:
            # Add points along coordinate axes
            axis_points = np.array([
                [0.5, 0.5, 0.25],
                [0.5, 0.5, 0.75],
                [0.5, 0.25, 0.5],
                [0.5, 0.75, 0.5],
                [0.25, 0.5, 0.5],
                [0.75, 0.5, 0.5]
            ])
            points = np.vstack([points, axis_points[:remaining]])
        
        return points
    
    # Strategy 5: Random with intelligent distribution
    def get_intelligent_random_initial():
        # Start with a better distributed set of random points
        points = np.random.rand(n, 3)
        
        # Apply a simple clustering avoidance technique
        # Move points away from each other iteratively
        for _ in range(100):
            for i in range(n):
                # Find closest neighbor
                dists = np.linalg.norm(points - points[i], axis=1)
                dists[i] = np.inf  # Ignore self-distance
                nearest_idx = np.argmin(dists)
                
                # Move point slightly away from nearest neighbor
                if dists[nearest_idx] < 0.1:  # If too close
                    direction = points[i] - points[nearest_idx]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                        points[i] += 0.01 * direction
        
        return points
    
    # Try multiple initialization strategies and pick the best
    initial_strategies = [
        get_spherical_code_initial,
        get_fibonacci_sphere_initial,
        get_icosahedral_refined_initial,
        get_octa_tetra_initial,
        get_intelligent_random_initial,
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
        if d_max <= 1e-15 or d_min <= 1e-15:
            return float('inf')
            
        # Compute ratio
        ratio = d_min / d_max
        
        # Add penalty for very small distances to encourage better spread
        # Also penalize when there are many very small distances
        penalty = 0.0
        small_count = np.sum(distances < 0.01)
        if d_min < 1e-8:
            return float('inf')
        elif d_min < 0.05:
            penalty = 1e10 * (1.0 / (d_min + 1e-12)) + 1e6 * small_count
        
        # Return negative ratio (since we want to maximize) 
        # with penalty for very small distances
        return -ratio + penalty
    
    # Better optimization with adaptive restarts and smarter methods
    def optimize_with_strategy(starting_points, max_restarts=8):
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
            elif restart < 3:
                # Small perturbations
                perturbed += np.random.normal(0, 0.003, perturbed.shape)
            elif restart < 6:
                # Medium perturbations
                perturbed += np.random.normal(0, 0.008, perturbed.shape)
            else:
                # Larger perturbations for later restarts
                perturbed += np.random.normal(0, 0.015, perturbed.shape)
            
            # Ensure within bounds
            perturbed = np.clip(perturbed, 0, 1)
            
            try:
                # Flatten points for optimization
                x0 = perturbed.flatten()
                
                # Set up bounds for all coordinates [0,1]
                bounds = [(0, 1) for _ in range(n * 3)]
                
                # Optimization options
                options = {'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                
                # Try different optimization methods
                methods = ['L-BFGS-B', 'SLSQP']  # More reliable than others
                
                for method in methods:
                    try:
                        result = minimize(
                            improved_objective,
                            x0,
                            method=method,
                            bounds=bounds,
                            options=options,
                            tol=1e-12
                        )
                        
                        # Extract results and compute actual ratio
                        final_points = result.x.reshape(-1, 3)
                        final_distances = pdist(final_points)
                        d_min = np.min(final_distances)
                        d_max = np.max(final_distances)
                        
                        if d_max > 1e-15 and d_min > 1e-15:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points.copy()
                                
                    except Exception:
                        continue
                        
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
    
    # Additional fine-tuning with local search
    if best_points is not None:
        # Try one more round of optimization with the best found solution
        try:
            # Apply a more aggressive optimization approach
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(n * 3)]
            options = {'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            
            # Use multiple methods for better chance of finding good solution
            methods = ['L-BFGS-B', 'SLSQP']
            for method in methods:
                try:
                    result = minimize(
                        improved_objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-14
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
