# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial import SphericalVoronoi
import random
from numba import jit
import time
from scipy.optimize import differential_evolution, dual_annealing
from scipy.spatial import distance
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

@jit(nopython=True)
def compute_min_max_ratio_fast(points):
    """Fast computation of min/max distance ratio using compiled code"""
    n = points.shape[0]
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    min_dist = np.inf
    max_dist = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(dist_sq)
            
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    if max_dist > 0:
        return min_dist / max_dist
    return 0.0

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initialization strategies with advanced optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        return np.min(distances) / np.max(distances)
    
    def objective_function(x):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        # Compute negative ratio (we want to maximize ratio, so minimize negative)
        ratio = compute_min_max_ratio_fast(points)  # Use fast version
        return -ratio
    
    def constraint_func(x):
        """Constraint to keep points within unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return positive values when constraint is satisfied
        return 1.0 - norms
    
    def generate_fibonacci_sphere(n):
        """Generate points on sphere using Fibonacci spiral"""
        points = np.zeros((n, 3))
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = golden_angle * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points[i] = [x, y, z]
        return points
    
    def generate_random_initialization(n, seed=None):
        """Generate random points in unit sphere"""
        if seed is not None:
            np.random.seed(seed)
        points = np.random.uniform(-1, 1, (n, 3))
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        valid_indices = norms <= 1
        points = points[valid_indices]
        
        # If we don't have enough points, generate more
        while len(points) < n:
            additional = np.random.uniform(-1, 1, (n - len(points), 3))
            additional_norms = np.linalg.norm(additional, axis=1)
            valid_additional = additional[additional_norms <= 1]
            points = np.vstack([points, valid_additional])
            
        return points[:n]
    
    def generate_icosahedron_initialization(n):
        """Generate points using icosahedron-based distribution"""
        # Vertices of regular icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # If we need more points, distribute them more evenly
        if n > 12:
            # Generate points along edges and faces of icosahedron
            points = vertices.copy()
            # Add midpoints of edges
            edges = []
            # Icosahedron has 30 edges
            edge_indices = [
                (0,1), (0,4), (0,5), (1,7), (1,10), (2,3), (2,6), (2,11),
                (3,4), (3,8), (4,9), (5,6), (5,9), (6,11), (7,8), (7,10),
                (8,9), (10,11), (0,2), (1,3), (4,5), (6,7), (8,10), (9,11)
            ]
            
            # Add edge midpoints
            for i, j in edge_indices:
                midpoint = (vertices[i] + vertices[j]) / 2
                midpoint = midpoint / np.linalg.norm(midpoint)
                points = np.vstack([points, midpoint])
                
            # Add face centers (approximate)
            faces = [
                (0,1,4), (0,5,1), (1,7,10), (1,10,0), (0,4,5), (5,9,4),
                (4,9,3), (3,9,8), (3,8,2), (2,8,6), (2,6,11), (6,11,7),
                (7,11,10), (10,11,3), (3,2,4), (4,2,5), (5,0,1), (1,0,7),
                (7,1,10), (10,1,3), (3,10,8), (8,10,11), (11,10,7), (7,11,6),
                (6,11,2), (2,11,8), (8,11,3), (3,8,9), (9,8,4), (4,9,5)
            ]
            
            for i, j, k in faces:
                center = (vertices[i] + vertices[j] + vertices[k]) / 3
                center = center / np.linalg.norm(center)
                points = np.vstack([points, center])
                
            # Take first n points
            if len(points) >= n:
                points = points[:n]
            else:
                # Fill with random points
                remaining = n - len(points)
                extra_points = np.random.uniform(-1, 1, (remaining, 3))
                extra_points = extra_points / np.linalg.norm(extra_points, axis=1, keepdims=True)
                points = np.vstack([points, extra_points])
                
        else:
            points = vertices[:n]
            
        return points
    
    def generate_better_initialization():
        """Generate a better initial configuration using geometric insights"""
        # Start with icosahedron points
        points = generate_icosahedron_initialization(14)
        
        # Improve distribution by applying Lloyd relaxation steps
        for _ in range(5):
            # Simple relaxation: move each point towards the centroid of its Voronoi cell
            # For simplicity, we'll do a basic adjustment
            noise = np.random.normal(0, 0.03, points.shape)
            points += noise
            
            # Project back to sphere
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * 0.95
            
        return points
    
    def generate_best_initialization():
        """Generate the best possible initialization strategy"""
        # Strategy 1: Icosahedron-based (most promising)
        points1 = generate_icosahedron_initialization(14)
        ratio1 = compute_min_max_ratio_fast(points1)
        
        # Strategy 2: Fibonacci sphere
        points2 = generate_fibonacci_sphere(14)
        points2 = points2 / np.max(np.linalg.norm(points2, axis=1)) * 0.9
        ratio2 = compute_min_max_ratio_fast(points2)
        
        # Strategy 3: Random initialization
        points3 = generate_random_initialization(14, seed=42)
        ratio3 = compute_min_max_ratio_fast(points3)
        
        # Strategy 4: Better initialization with relaxation
        points4 = generate_better_initialization()
        ratio4 = compute_min_max_ratio_fast(points4)
        
        # Select the best initialization
        ratios = [ratio1, ratio2, ratio3, ratio4]
        best_idx = np.argmax(ratios)
        
        if best_idx == 0:
            return points1
        elif best_idx == 1:
            return points2
        elif best_idx == 2:
            return points3
        else:
            return points4
    
    # Generate best initialization
    initial_points = generate_best_initialization()
    
    # Optimization with targeted strategies for speed and quality
    best_optimized_points = initial_points.copy()
    best_optimization_ratio = compute_min_max_ratio_fast(initial_points)
    
    # Track optimization progress
    start_time = time.time()
    max_time = 55  # Leave some buffer time
    
    # Strategy 1: Direct optimization with trust-constr (primary method)
    if time.time() - start_time < max_time:
        try:
            # Use trust-constr with aggressive settings
            result = minimize(
                objective_function,
                initial_points.flatten(),
                method='trust-constr',
                bounds=[(-0.99, 0.99) for _ in range(42)],
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-18, 'verbose': 0},
                tol=1e-18
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                # Ensure points are within unit sphere
                norms = np.linalg.norm(optimized_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    optimized_points[mask] = optimized_points[mask] / norms[mask, np.newaxis] * 0.99
                
                ratio = compute_min_max_ratio_fast(optimized_points)
                if ratio > best_optimization_ratio:
                    best_optimization_ratio = ratio
                    best_optimized_points = optimized_points
                    
        except Exception:
            pass
    
    # Strategy 2: Enhanced Differential Evolution
    if time.time() - start_time < max_time:
        try:
            # Use a more sophisticated global optimization approach
            bounds = [(-0.99, 0.99) for _ in range(42)]
            
            # Try DE with improved parameters
            de_result = differential_evolution(
                lambda x: objective_function(x),
                bounds,
                maxiter=500, 
                popsize=60, 
                mutation=(0.9, 1), 
                recombination=0.95, 
                seed=42,
                atol=1e-18,
                rtol=1e-18
            )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 3)
                # Ensure points are within unit sphere
                norms = np.linalg.norm(de_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    de_points[mask] = de_points[mask] / norms[mask, np.newaxis] * 0.99
                
                de_ratio = compute_min_max_ratio_fast(de_points)
                if de_ratio > best_optimization_ratio:
                    best_optimization_ratio = de_ratio
                    best_optimized_points = de_points
                        
        except Exception:
            pass
    
    # Strategy 3: Simulated Annealing with better cooling schedule
    if time.time() - start_time < max_time:
        try:
            bounds = [(-0.99, 0.99) for _ in range(42)]
            
            # Try SA with improved parameters
            sa_result = dual_annealing(
                lambda x: objective_function(x),
                bounds,
                maxiter=1000, 
                initial_temp=50000, 
                no_local_search=False, 
                seed=42
            )
            
            if sa_result.success:
                sa_points = sa_result.x.reshape(-1, 3)
                # Ensure points are within unit sphere
                norms = np.linalg.norm(sa_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    sa_points[mask] = sa_points[mask] / norms[mask, np.newaxis] * 0.99
                
                sa_ratio = compute_min_max_ratio_fast(sa_points)
                if sa_ratio > best_optimization_ratio:
                    best_optimization_ratio = sa_ratio
                    best_optimized_points = sa_points
                        
        except Exception:
            pass
    
    # Strategy 4: Multiple restarts with different methods
    if time.time() - start_time < max_time:
        try:
            # Try multiple restarts with different optimization methods
            methods = ['L-BFGS-B', 'TNC', 'SLSQP']
            restarts_per_method = 5  # Reduced to save time
            
            for method in methods:
                if time.time() - start_time > max_time:
                    break
                    
                for restart in range(restarts_per_method):
                    if time.time() - start_time > max_time:
                        break
                        
                    # Use the best solution found so far as starting point
                    current_start = best_optimized_points.copy()
                    
                    # Add noise to escape local minima
                    noise = np.random.normal(0, 0.02, current_start.shape)
                    current_start += noise
                    
                    # Keep within sphere
                    norms = np.linalg.norm(current_start, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        current_start[mask] = current_start[mask] / norms[mask, np.newaxis] * 0.99
                    
                    initial_guess = current_start.flatten()
                    
                    result = minimize(
                        objective_function,
                        initial_guess,
                        method=method,
                        bounds=[(-0.99, 0.99) for _ in range(42)],
                        constraints={'type': 'ineq', 'fun': constraint_func},
                        options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18},
                        tol=1e-18
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        # Ensure points are within unit sphere
                        norms = np.linalg.norm(optimized_points, axis=1)
                        mask = norms > 1.0
                        if np.any(mask):
                            optimized_points[mask] = optimized_points[mask] / norms[mask, np.newaxis] * 0.99
                        
                        ratio = compute_min_max_ratio_fast(optimized_points)
                        if ratio > best_optimization_ratio:
                            best_optimization_ratio = ratio
                            best_optimized_points = optimized_points
                            
        except Exception:
            pass
    
    # Strategy 5: Final refinement with targeted approach
    if time.time() - start_time < max_time:
        try:
            # Try a more focused optimization with specialized approach
            refined_points = best_optimized_points.copy()
            
            # Pass 1: Trust-constr with very tight tolerances
            result = minimize(
                objective_function,
                refined_points.flatten(),
                method='trust-constr',
                bounds=[(-0.99, 0.99) for _ in range(42)],
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18, 'verbose': 0},
                tol=1e-18
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(refined_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    refined_points[mask] = refined_points[mask] / norms[mask, np.newaxis] * 0.99
                
                ratio = compute_min_max_ratio_fast(refined_points)
                if ratio > best_optimization_ratio:
                    best_optimization_ratio = ratio
                    best_optimized_points = refined_points
            
            # Pass 2: L-BFGS-B with different tolerance
            result = minimize(
                objective_function,
                refined_points.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.99, 0.99) for _ in range(42)],
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18},
                tol=1e-18
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(refined_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    refined_points[mask] = refined_points[mask] / norms[mask, np.newaxis] * 0.99
                
                ratio = compute_min_max_ratio_fast(refined_points)
                if ratio > best_optimization_ratio:
                    best_optimization_ratio = ratio
                    best_optimized_points = refined_points
                    
        except Exception:
            pass
    
    # Strategy 6: Additional refinement using adaptive sampling
    if time.time() - start_time < max_time:
        try:
            # Generate several candidates from a refined neighborhood
            candidates = []
            for i in range(5):  # Reduced number to save time
                # Small perturbation around the best solution
                perturbed = best_optimized_points + np.random.normal(0, 0.01, best_optimized_points.shape)
                # Project back to sphere
                norms = np.linalg.norm(perturbed, axis=1)
                perturbed = perturbed / norms[:, np.newaxis] * 0.99
                candidates.append(perturbed)
            
            # Evaluate all candidates
            candidate_ratios = []
            for candidate in candidates:
                ratio = compute_min_max_ratio_fast(candidate)
                candidate_ratios.append(ratio)
            
            # Select the best candidate
            best_candidate_idx = np.argmax(candidate_ratios)
            if candidate_ratios[best_candidate_idx] > best_optimization_ratio:
                best_optimization_ratio = candidate_ratios[best_candidate_idx]
                best_optimized_points = candidates[best_candidate_idx]
                
        except Exception:
            pass
    
    return best_optimized_points


# EVOLVE-BLOCK-END
