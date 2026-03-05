# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import random
from typing import Tuple
import time
from numba import jit


@jit(nopython=True)
def compute_distances_numba(points):
    """Compute pairwise distances using numba for speed"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i,j] = dist
            distances[j,i] = dist
    return distances


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
    
    def objective_fast(x_flat):
        """Faster objective using numba compiled computation"""
        points = x_flat.reshape(-1, 3)
        # Use a simpler approach that avoids expensive pdist calls
        n = points.shape[0]
        min_dist_sq = float('inf')
        max_dist_sq = 0.0
        
        for i in range(n):
            for j in range(i+1, n):
                dist_sq = sum((points[i,k] - points[j,k])**2 for k in range(3))
                if dist_sq > max_dist_sq:
                    max_dist_sq = dist_sq
                if dist_sq < min_dist_sq and dist_sq > 0:
                    min_dist_sq = dist_sq
        
        if max_dist_sq == 0:
            return 0
            
        return -(np.sqrt(min_dist_sq) / np.sqrt(max_dist_sq))
    
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
    
    def compute_min_max_ratio_fast(points):
        """Fast version to compute min/max ratio"""
        n = points.shape[0]
        min_dist_sq = float('inf')
        max_dist_sq = 0.0
        
        for i in range(n):
            for j in range(i+1, n):
                dist_sq = sum((points[i,k] - points[j,k])**2 for k in range(3))
                if dist_sq > max_dist_sq:
                    max_dist_sq = dist_sq
                if dist_sq < min_dist_sq and dist_sq > 0:
                    min_dist_sq = dist_sq
        
        if max_dist_sq == 0:
            return 0.0
            
        return np.sqrt(min_dist_sq) / np.sqrt(max_dist_sq)
    
    # Generate a much better initial configuration using known good patterns
    np.random.seed(42)
    
    # Strategy: Start with a configuration inspired by optimal sphere packings
    # Use a construction that places points to avoid clustering and maximize uniformity
    
    # First, create a more structured configuration
    # We'll use a combination of regular polyhedra and carefully placed additional points
    
    # Start with icosahedral symmetry - 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = []
    
    # Generate icosahedron vertices properly
    coords = [
        (0, 1, phi), (0, -1, phi), (0, -1, -phi), (0, 1, -phi),
        (1, phi, 0), (-1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (-phi, 0, -1), (phi, 0, -1)
    ]
    
    # Normalize to unit sphere
    for x, y, z in coords:
        norm = np.sqrt(x*x + y*y + z*z)
        vertices.append([x/norm, y/norm, z/norm])
    
    vertices = np.array(vertices)
    
    # Add two more points to make 14 points total - strategic placement
    # Place at poles and ensure good distribution
    points = np.vstack([vertices, [0, 0, 0.95], [0, 0, -0.95]])
    
    # Better perturbation strategy - use more structured approach
    # Apply small random perturbations with controlled magnitude
    points += np.random.normal(0, 0.05, points.shape)
    
    # Normalize to ensure they're on or near the unit sphere
    for i in range(len(points)):
        norm = np.linalg.norm(points[i])
        if norm > 0:
            points[i] = points[i] / norm * 0.98  # Slightly inside unit sphere
    
    # Refine using a more robust optimization approach with better strategies
    x0 = points.flatten()
    
    # Define constraints - only bounds
    cons = [
        {'type': 'ineq', 'fun': constraint_bounds}
    ]
    
    best_points = points.copy()
    best_ratio = compute_min_max_ratio(best_points)
    
    # Method 1: Multiple restarts with different optimization methods
    methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    
    # Use a more intelligent restart strategy
    restarts = 30  # More restarts for better exploration
    
    for restart in range(restarts):
        try:
            # Randomly perturb the starting point for this restart
            # Use different perturbation sizes based on restart number
            perturbation_scale = 0.15 * (1.0 - restart/float(restarts)) + 0.05
            current_x0 = x0 + np.random.normal(0, perturbation_scale, x0.shape)
            
            # Try different methods with different parameters
            method_params = [
                ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}),
                ('TNC', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}),
                ('SLSQP', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            ]
            
            # Alternate between fast and accurate versions
            use_fast = restart > restarts // 2
            
            for method, options in method_params:
                try:
                    result = minimize(
                        objective_fast if use_fast else objective,
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
                            
                        # Early stopping if we're close to target
                        if best_ratio > 0.48:  # Tighter early stopping
                            break
                except Exception as e:
                    continue
                    
            # If we've made significant progress, don't waste time on more restarts
            if best_ratio > 0.48:
                break
        except Exception as e:
            continue
    
    # Method 2: Enhanced global optimization with better initialization
    if best_ratio < 0.46:  # If still not good enough, try more sophisticated approach
        # Try a hybrid approach: start with a known good configuration and optimize
        try:
            # Use a better known configuration pattern - based on research on optimal point distributions
            # This uses a configuration inspired by mathematical constructions for 14 points
            initial_config = np.array([
                # North and South poles
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                
                # Around equator - 12 points arranged in a circle
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.7071067811865476, 0.0, 0.7071067811865476],
                [-0.7071067811865476, 0.0, 0.7071067811865476],
                [0.0, 0.7071067811865476, 0.7071067811865476],
                [0.0, -0.7071067811865476, 0.7071067811865476],
                [0.7071067811865476, 0.0, -0.7071067811865476],
                [-0.7071067811865476, 0.0, -0.7071067811865476],
                [0.0, 0.7071067811865476, -0.7071067811865476],
                [0.0, -0.7071067811865476, -0.7071067811865476]
            ])
            
            # Add some noise
            initial_config += np.random.normal(0, 0.08, initial_config.shape)
            
            # Normalize to unit sphere
            for i in range(len(initial_config)):
                norm = np.linalg.norm(initial_config[i])
                if norm > 0:
                    initial_config[i] = initial_config[i] / norm * 0.97
            
            # Optimize this configuration
            result = minimize(
                objective,
                initial_config.flatten(),
                method='L-BFGS-B',
                constraints=cons,
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10}
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
    
    # Method 3: Add a more sophisticated local search approach with improved gradient estimation
    if best_ratio < 0.485:  # Only if needed
        # Use a more aggressive gradient-based approach with adaptive learning rates
        points_flat = best_points.flatten()
        learning_rate = 0.01  # Higher learning rate
        iterations = 3000  # More iterations
        momentum = 0.95  # Stronger momentum factor
        velocity = np.zeros_like(points_flat)
        
        # Track improvements to enable early stopping
        last_improvement = 0
        last_ratio = best_ratio
        
        for iteration in range(iterations):
            # Compute current ratio
            old_ratio = compute_min_max_ratio_fast(points_flat.reshape(-1, 3))
            
            # Small perturbation for numerical gradient estimation
            epsilon = 1e-6
            gradients = np.zeros_like(points_flat)
            
            for i in range(len(points_flat)):
                # Perturb coordinate
                points_plus = points_flat.copy()
                points_minus = points_flat.copy()
                points_plus[i] += epsilon
                points_minus[i] -= epsilon
                
                # Clamp to bounds
                points_plus = np.clip(points_plus, -1, 1)
                points_minus = np.clip(points_minus, -1, 1)
                
                ratio_plus = compute_min_max_ratio_fast(points_plus.reshape(-1, 3))
                ratio_minus = compute_min_max_ratio_fast(points_minus.reshape(-1, 3))
                
                gradients[i] = (ratio_plus - ratio_minus) / (2 * epsilon)
            
            # Update with momentum
            velocity = momentum * velocity - learning_rate * gradients
            points_flat += velocity
            
            # Clamp to bounds
            points_flat = np.clip(points_flat, -1, 1)
            
            # Adaptive learning rate
            if iteration > 1000:
                learning_rate *= 0.99  # Gradual decay
                
            # Early stopping if improvement is minimal
            new_ratio = compute_min_max_ratio_fast(points_flat.reshape(-1, 3))
            if abs(new_ratio - old_ratio) < 1e-9:
                last_improvement += 1
                if last_improvement > 100:  # Stop if no improvement for 100 iterations
                    break
            else:
                last_improvement = 0
                last_ratio = new_ratio
        
        best_points = points_flat.reshape(-1, 3)
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, -1, 1)
    
    # Final optimization step with highest precision
    try:
        result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            constraints=cons,
            options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result.success:
            final_points = result.x.reshape(-1, 3)
            final_points = np.clip(final_points, -1, 1)
            ratio = compute_min_max_ratio(final_points)
            if ratio > best_ratio:
                best_points = final_points
    except:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
