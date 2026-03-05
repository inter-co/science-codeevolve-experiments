# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import random
from typing import Tuple
import time
from numba import jit
from scipy.spatial import SphericalVoronoi
import itertools


@jit(nopython=True)
def compute_min_max_ratio_fast(points):
    """Fast computation of min/max ratio for 14 points"""
    n = points.shape[0]
    min_dist_sq = float('inf')
    max_dist_sq = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist_sq += diff * diff
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
            if dist_sq < min_dist_sq and dist_sq > 0:
                min_dist_sq = dist_sq
    
    if max_dist_sq == 0:
        return 0.0
    return np.sqrt(min_dist_sq) / np.sqrt(max_dist_sq)


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining geometric insights, evolutionary initialization,
    and advanced optimization techniques.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x_flat.reshape(-1, 3)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0  # Avoid division by zero
            
        return -min_dist / max_dist
    
    def constraint_bounds(x_flat):
        """Constraint ensuring all points are within [-1,1]^3 bounds"""
        points = x_flat.reshape(-1, 3)
        # Return negative values for points outside bounds
        return np.minimum(1.0 - np.abs(points), 0.0).flatten()
    
    def compute_min_max_ratio(points):
        """Helper function to compute min/max ratio for given points"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist
    
    # Generate a much better initial configuration using a more principled approach
    np.random.seed(42)
    
    # Strategy 1: Start with a configuration derived from known good arrangements
    # Based on research, a good starting point uses combinations of symmetric arrangements
    
    # Create a set of points inspired by icosahedral symmetry and sphere packing
    # Use a combination of icosahedron vertices plus strategic additional points
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (normalized)
    icosahedron_vertices = []
    # 12 vertices of regular icosahedron
    coords = [
        (0, 1, phi), (0, -1, phi), (0, -1, -phi), (0, 1, -phi),
        (1, phi, 0), (-1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (-phi, 0, -1), (phi, 0, -1)
    ]
    
    # Normalize to unit sphere
    for x, y, z in coords:
        norm = np.sqrt(x*x + y*y + z*z)
        icosahedron_vertices.append([x/norm, y/norm, z/norm])
    
    icosahedron_vertices = np.array(icosahedron_vertices)
    
    # Strategy 2: Add 2 more points strategically placed
    # Use a method that distributes points evenly on sphere
    # Add points that maximize minimum distance to existing points
    
    # Start with the icosahedron vertices
    points = icosahedron_vertices.copy()
    
    # Add two more points - place them along z-axis but with careful positioning
    # This helps balance the configuration and improve the min/max ratio
    points = np.vstack([points, [0, 0, 0.95], [0, 0, -0.95]])
    
    # Strategy 3: Improve initial configuration with a more systematic approach
    # Use a better perturbation that respects symmetry and improves spread
    
    # Better perturbation - apply a more structured approach
    # Add small perturbations in a way that maintains good spread
    perturbation_magnitude = 0.08
    
    # Apply structured perturbations
    for i in range(len(points)):
        # Add a small random perturbation
        perturbation = np.random.normal(0, perturbation_magnitude, 3)
        points[i] += perturbation
        
        # Normalize to keep on or near unit sphere
        norm = np.linalg.norm(points[i])
        if norm > 0:
            points[i] = points[i] / norm * 0.97
    
    # Strategy 4: Use a more intelligent optimization approach
    # Implement a multi-stage optimization with better convergence
    x0 = points.flatten()
    
    # Define constraints - only bounds
    cons = [
        {'type': 'ineq', 'fun': constraint_bounds}
    ]
    
    best_points = points.copy()
    best_ratio = compute_min_max_ratio(best_points)
    
    # Method 1: Multi-start optimization with better parameters
    methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    
    # Try multiple restarts with different strategies
    for restart in range(12):  # More restarts for better exploration
        try:
            # Strategy A: Small perturbation for first few restarts
            # Strategy B: Larger perturbation for later restarts to escape local optima
            if restart < 6:
                current_x0 = x0 + np.random.normal(0, 0.1, x0.shape)
            else:
                current_x0 = x0 + np.random.normal(0, 0.2, x0.shape)
            
            # Try different methods with varying parameters
            for method in methods:
                try:
                    # Different optimization settings for different phases
                    if restart < 4:
                        options = {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9}
                    elif restart < 8:
                        options = {'maxiter': 800, 'ftol': 1e-11, 'gtol': 1e-8}
                    else:
                        options = {'maxiter': 600, 'ftol': 1e-10, 'gtol': 1e-7}
                    
                    result = minimize(
                        objective,
                        current_x0,
                        method=method,
                        constraints=cons,
                        options=options
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 3)
                        # Ensure points stay within bounds
                        final_points = np.clip(final_points, -1, 1)
                        ratio = compute_min_max_ratio(final_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
                        # Early stopping if we're approaching target
                        if best_ratio > 0.48:  # Stop earlier if very close to target
                            break
                except Exception as e:
                    continue
                    
            # Early stopping if we're getting good results
            if best_ratio > 0.485:
                break
        except Exception as e:
            continue
    
    # Method 2: Global optimization approach with better initial seeding
    if best_ratio < 0.46:  # If still not good enough, try more sophisticated approach
        try:
            # Try a more diverse set of initial configurations
            configs_to_try = []
            
            # Configuration 1: Vertices of a cube with added points
            cube_points = np.array([
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                [0, 0, 0.9], [0, 0, -0.9]
            ])
            configs_to_try.append(cube_points)
            
            # Configuration 2: Tetrahedral arrangement with additional points
            tetra_points = np.array([
                [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
                [0, 0, 0.9], [0, 0, -0.9]
            ])
            configs_to_try.append(tetra_points)
            
            # Configuration 3: Mix of icosahedral and cube points
            mixed_points = np.vstack([
                icosahedron_vertices[:8],  # First 8 icosahedron vertices
                [[0, 0, 0.9], [0, 0, -0.9]]  # Two additional points
            ])
            configs_to_try.append(mixed_points)
            
            # Try each configuration
            for config in configs_to_try:
                # Add noise to the configuration
                config_noisy = config + np.random.normal(0, 0.05, config.shape)
                
                # Normalize to unit sphere
                for i in range(len(config_noisy)):
                    norm = np.linalg.norm(config_noisy[i])
                    if norm > 0:
                        config_noisy[i] = config_noisy[i] / norm * 0.97
                
                # Optimize this configuration
                result = minimize(
                    objective,
                    config_noisy.flatten(),
                    method='L-BFGS-B',
                    constraints=cons,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    final_points = np.clip(final_points, -1, 1)
                    ratio = compute_min_max_ratio(final_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
        except Exception as e:
            pass
    
    # Method 3: Local refinement with adaptive learning rates
    if best_ratio < 0.48:  # Only refine if needed
        points_flat = best_points.flatten()
        learning_rate = 0.003  # Slightly higher learning rate
        iterations = 2000  # More iterations
        
        # Track improvement
        prev_ratio = 0
        patience_counter = 0
        
        for iteration in range(iterations):
            # Coordinate descent - update one coordinate at a time
            for coord_idx in range(len(points_flat)):
                # Save original value
                original_value = points_flat[coord_idx]
                
                # Try small positive and negative steps
                best_value = original_value
                best_ratio_current = compute_min_max_ratio(points_flat.reshape(-1, 3))
                
                step_sizes = [learning_rate, -learning_rate]
                for step_size in step_sizes:
                    test_value = original_value + step_size
                    # Clamp to bounds
                    test_value = np.clip(test_value, -1, 1)
                    
                    # Test this change
                    points_flat[coord_idx] = test_value
                    current_points = points_flat.reshape(-1, 3)
                    current_points = np.clip(current_points, -1, 1)
                    ratio = compute_min_max_ratio(current_points)
                    
                    if ratio > best_ratio_current:
                        best_ratio_current = ratio
                        best_value = test_value
                        
                    # Restore original
                    points_flat[coord_idx] = original_value
                
                # Apply best value found
                points_flat[coord_idx] = best_value
            
            # Adaptive learning rate
            if iteration > 500:
                learning_rate *= 0.995  # Slower decay
                
            # Early stopping if improvement is minimal
            current_ratio = compute_min_max_ratio(points_flat.reshape(-1, 3))
            if abs(current_ratio - prev_ratio) < 1e-8:
                patience_counter += 1
                if patience_counter > 50:
                    break
            else:
                patience_counter = 0
            prev_ratio = current_ratio
        
        best_points = points_flat.reshape(-1, 3)
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, -1, 1)
    
    return best_points


# EVOLVE-BLOCK-END
