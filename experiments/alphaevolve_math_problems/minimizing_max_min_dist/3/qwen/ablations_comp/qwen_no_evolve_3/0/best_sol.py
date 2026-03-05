# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math
from scipy.spatial import SphericalVoronoi
import random
from scipy.spatial import ConvexHull
import time
from numba import jit
import warnings
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix

@jit(nopython=True)
def fast_min_max_ratio(points):
    """Fast computation of min/max distance ratio using Numba JIT"""
    n = points.shape[0]
    min_dist_sq = np.inf
    max_dist_sq = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist_sq += diff * diff
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
    
    min_dist = np.sqrt(min_dist_sq) if min_dist_sq != np.inf else 0.0
    max_dist = np.sqrt(max_dist_sq) if max_dist_sq > 0 else 1.0
    
    return min_dist / max_dist if max_dist > 0 else 0.0

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses an advanced evolutionary approach combining multiple initialization strategies 
    with gradient-based optimization and improved simulated annealing.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x_flat.reshape(-1, 3)
        # Use faster computation
        ratio = fast_min_max_ratio(points)
        return -ratio
    
    def constraint_func(x_flat):
        """Constraint to keep points within unit sphere"""
        points = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return negative values for points outside unit sphere (constraint violation)
        return 1.0 - norms
    
    def initialize_points():
        """Initialize points using a focused approach on known good configurations"""
        # Focus on proven geometric arrangements that work well for point dispersion
        # Strategy 1: Optimized icosahedral arrangement with additional points
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        ico_vertices = [
            [0, 1, phi], [0, -1, phi], [0, -1, -phi], [0, 1, -phi],
            [1, phi, 0], [-1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, -1], [-phi, 0, 1]
        ]
        
        points1 = np.array(ico_vertices[:12])  # Take first 12 vertices
        # Add two more points at poles for better coverage
        points1 = np.vstack([points1, [[0, 0, 0.9], [0, 0, -0.9]]])
        
        # Normalize to unit sphere
        norms1 = np.linalg.norm(points1, axis=1)
        if np.max(norms1) > 0:
            points1 = points1 / np.max(norms1)
        
        # Strategy 2: Fibonacci-based approach with better spacing
        points2 = []
        for i in range(14):
            # Golden spiral approach for better uniformity
            y = 1 - (i / (14 - 1)) * 2  # y from 1 to -1
            radius = math.sqrt(1 - y * y)
            
            # Use golden angle for better distribution
            golden_angle = 2.399963229728653  # ~4π/(1+√5) 
            phi = (i * golden_angle) % (2 * np.pi)
            
            x = radius * np.cos(phi)
            z = radius * np.sin(phi)
            points2.append([x, y, z])
        
        points2 = np.array(points2)
        
        # Strategy 3: Cube-based arrangement with extra points
        points3 = []
        # Place points on the faces of a cube
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == 1 and j == 1 and k == 1:  # Skip center
                        continue
                    points3.append([(i-1)*0.8, (j-1)*0.8, (k-1)*0.8])
        
        # Add 11 more points to make 14 total
        points3.extend([[0, 0, 0.95], [0, 0, -0.95], [0.95, 0, 0], [-0.95, 0, 0], [0, 0.95, 0], [0, -0.95, 0],
                       [0.707, 0.707, 0], [-0.707, 0.707, 0], [0.707, -0.707, 0], [-0.707, -0.707, 0],
                       [0, 0, 0.8]])
        points3 = np.array(points3)
        
        # Strategy 4: Simple but effective approach - regular icosahedron with adjustments
        # Vertices of regular icosahedron
        vertices = np.array([
            [0, 1, 1.618], [0, -1, 1.618], [0, 1, -1.618], [0, -1, -1.618],
            [1.618, 0, 1], [-1.618, 0, 1], [1.618, 0, -1], [-1.618, 0, -1],
            [1, 1.618, 0], [-1, 1.618, 0], [1, -1.618, 0], [-1, -1.618, 0]
        ])
        
        # Add 2 more points at top and bottom
        points4 = np.vstack([vertices, [[0, 0, 1.5], [0, 0, -1.5]]])
        
        # Normalize to unit sphere
        norms4 = np.linalg.norm(points4, axis=1)
        if np.max(norms4) > 0:
            points4 = points4 / np.max(norms4)
        
        # Evaluate strategies
        strategies = [
            ("Icosahedral", points1),
            ("Fibonacci", points2),
            ("Cube-based", points3),
            ("Icosahedron+", points4)
        ]
        
        best_points = points1
        best_ratio = fast_min_max_ratio(points1)
        
        for name, pts in strategies:
            try:
                ratio = fast_min_max_ratio(pts)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = pts.copy()
            except Exception as e:
                continue
        
        return best_points
    
    def improved_simulated_annealing(initial_points, max_iter=2500, temp_start=1.0, temp_decay=0.995):
        """Improved simulated annealing with better convergence properties"""
        current_points = initial_points.copy()
        current_ratio = fast_min_max_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        temperature = temp_start
        
        # Track recent improvements for early stopping
        recent_improvements = []
        stagnation_count = 0
        
        for iteration in range(max_iter):
            if temperature < 1e-12:
                break
                
            # Adaptive perturbation scaling
            perturbation_magnitude = max(0.001, temperature * 0.03)
            
            # Perturb several points at once for better exploration
            num_perturbations = max(1, min(8, int(1 + np.random.exponential(1.5))))
            new_points = current_points.copy()
            
            for _ in range(num_perturbations):
                idx = np.random.randint(0, len(current_points))
                # Use adaptive scale based on temperature
                scale = perturbation_magnitude * (0.3 + 0.7 * np.random.rand())
                perturbation = np.random.normal(0, scale, 3)
                
                # Apply perturbation
                new_points[idx] += perturbation
                
                # Project back to unit sphere
                norm = np.linalg.norm(new_points[idx])
                if norm > 1.0:
                    new_points[idx] = new_points[idx] / norm
            
            # Calculate new ratio
            new_ratio = fast_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                current_points = new_points
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
                    recent_improvements = []
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                # Accept with probability based on temperature
                delta = new_ratio - current_ratio
                if np.random.rand() < np.exp(delta / temperature):
                    current_points = new_points
                    current_ratio = new_ratio
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            
            # Decay temperature
            temperature *= temp_decay
            
            # Early stopping if no improvement for a while
            if stagnation_count > 100:
                break
            
            # Track recent improvements
            recent_improvements.append(current_ratio)
            if len(recent_improvements) > 50:
                recent_improvements.pop(0)
                # Stop if no significant improvement recently
                if len(recent_improvements) >= 10:
                    recent_change = abs(recent_improvements[-1] - recent_improvements[0])
                    if recent_change < 1e-12:
                        break
        
        return best_points, best_ratio
    
    def local_optimization(initial_points, max_time=30):
        """Focused local optimization approach"""
        start_time = time.time()
        best_points = initial_points.copy()
        best_ratio = fast_min_max_ratio(best_points)
        
        # Try multiple optimization approaches
        strategies = ['L-BFGS-B', 'TNC', 'SLSQP']
        n = 14
        
        for strategy in strategies:
            if time.time() - start_time > max_time:
                break
                
            try:
                # Start with current best points
                x0 = best_points.flatten()
                bounds = [(-1, 1) for _ in range(3 * n)]
                cons = [{'type': 'ineq', 'fun': constraint_func}]
                
                result = minimize(
                    objective,
                    x0,
                    method=strategy,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-16}
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 3)
                    # Ensure all points are within unit sphere
                    norms = np.linalg.norm(optimized_points, axis=1)
                    max_norm = np.max(norms)
                    if max_norm > 1.0:
                        optimized_points = optimized_points / max_norm
                    
                    ratio = fast_min_max_ratio(optimized_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
            except Exception as e:
                continue
        
        return best_points, best_ratio
    
    # Initialize with focused geometric strategies
    points = initialize_points()
    
    # First, apply improved simulated annealing
    try:
        sa_points, sa_ratio = improved_simulated_annealing(points, max_iter=2000)
        if sa_ratio > fast_min_max_ratio(points):
            points = sa_points
    except Exception as e:
        pass
    
    # Then apply local optimization
    try:
        optimized_points, ratio = local_optimization(points, max_time=25)
        if ratio > fast_min_max_ratio(points):
            points = optimized_points
    except Exception as e:
        pass
    
    # Final refinement with multiple restarts
    for i in range(8):
        try:
            # Slightly perturb current points
            perturbed_points = points + np.random.normal(0, 0.005, points.shape)
            # Normalize to unit sphere
            norms = np.linalg.norm(perturbed_points, axis=1)
            if np.max(norms) > 0:
                perturbed_points = perturbed_points / np.max(norms)
            
            # Optimize from this perturbed state
            x0 = perturbed_points.flatten()
            bounds = [(-1, 1) for _ in range(3 * 14)]
            cons = [{'type': 'ineq', 'fun': constraint_func}]
            
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(optimized_points, axis=1)
                max_norm = np.max(norms)
                if max_norm > 1.0:
                    optimized_points = optimized_points / max_norm
                
                ratio = fast_min_max_ratio(optimized_points)
                if ratio > fast_min_max_ratio(points):
                    points = optimized_points
                    
        except Exception as e:
            continue
    
    # Final validation
    # Ensure all points are within unit sphere
    norms = np.linalg.norm(points, axis=1)
    max_norm = np.max(norms)
    if max_norm > 1.0:
        points = points / max_norm
    
    return points


# EVOLVE-BLOCK-END
