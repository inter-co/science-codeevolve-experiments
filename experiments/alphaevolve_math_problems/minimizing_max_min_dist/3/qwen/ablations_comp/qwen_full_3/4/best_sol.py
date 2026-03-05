# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction and multi-start optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x_flat.reshape(-1, 3)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0
            
        # Return negative ratio (we want to maximize ratio, so minimize negative)
        return -min_dist / max_dist
    
    def fibonacci_sphere(n_points):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)
    
    # Try multiple initialization strategies with enhanced multi-start approach
    best_points = None
    best_ratio = -np.inf
    
    # Strategy 1: Fibonacci sphere (good general distribution)
    try:
        initial_points = fibonacci_sphere(14)
        norms = np.linalg.norm(initial_points, axis=1)
        if np.any(norms > 0):
            initial_points = initial_points / np.max(norms) * 0.9
        initial_flat = initial_points.flatten()
        
        # Multi-start optimization with different random perturbations (like INSPIRATION 1)
        for _ in range(10):  # Increased number of random starts for better exploration
            # Add small random perturbation to avoid local minima
            perturbed = initial_flat + np.random.normal(0, 0.05, len(initial_flat))
            
            # Optimize using SLSQP method which handles constraints better
            try:
                result = minimize(
                    objective,
                    perturbed,
                    method='SLSQP',
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10},
                    tol=1e-10
                )
                
                if result.success:
                    current_points = result.x.reshape(-1, 3)
                    distances = pdist(current_points)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = current_points.copy()
                            
            except Exception:
                continue
                        
    except Exception:
        pass
    
    # Strategy 2: Icosahedral-based initialization (from inspiration 3)
    try:
        # Use icosahedral initialization with better point distribution
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1)
        vertices = vertices / norms[:, np.newaxis]
        
        # Add two more points at poles for 14 total
        additional = np.array([[0, 0, 1], [0, 0, -1]])
        initial_points = np.vstack([vertices, additional])
        
        initial_flat = initial_points.flatten()
        
        # Multi-start optimization with more random starts
        for _ in range(10):  # Increased number of random starts for better exploration
            # Add small random perturbation
            perturbed = initial_flat + np.random.normal(0, 0.05, len(initial_flat))
            
            # Optimize using SLSQP method which handles constraints better
            try:
                result = minimize(
                    objective,
                    perturbed,
                    method='SLSQP',
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10},
                    tol=1e-10
                )
                
                if result.success:
                    current_points = result.x.reshape(-1, 3)
                    distances = pdist(current_points)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = current_points.copy()
                            
            except Exception:
                continue
                        
    except Exception:
        pass
    
    # Strategy 3: Random initialization with better distribution
    try:
        # Generate random points on sphere
        random_points = np.random.randn(14, 3)
        norms = np.linalg.norm(random_points, axis=1)
        random_points = random_points / norms[:, np.newaxis] * 0.9
        
        random_flat = random_points.flatten()
        
        # Multi-start optimization
        for _ in range(5):  # Keep reasonable number of starts
            # Add small random perturbation
            perturbed = random_flat + np.random.normal(0, 0.05, len(random_flat))
            
            # Optimize using SLSQP method which handles constraints better
            try:
                result = minimize(
                    objective,
                    perturbed,
                    method='SLSQP',
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10},
                    tol=1e-10
                )
                
                if result.success:
                    current_points = result.x.reshape(-1, 3)
                    distances = pdist(current_points)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = current_points.copy()
                            
            except Exception:
                continue
                        
    except Exception:
        pass
    
    # Final refinement step - try one additional global optimization with best configuration
    if best_points is not None and best_ratio < 0.4898:  # If not already very good
        # Refine the best solution found with more aggressive optimization
        try:
            # Create a more refined optimization starting from best solution
            refined_flat = best_points.flatten()
            result = minimize(
                objective,
                refined_flat,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                distances = pdist(refined_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    refined_ratio = min_dist / max_dist
                    if refined_ratio > best_ratio:
                        best_points = refined_points
        except Exception:
            pass
    
    # Ensure we always return a valid result
    if best_points is None:
        # Use Fibonacci sphere as final fallback
        best_points = fibonacci_sphere(14)
        norms = np.linalg.norm(best_points, axis=1)
        if np.any(norms > 0):
            best_points = best_points / np.max(norms) * 0.9
    
    return best_points


# EVOLVE-BLOCK-END
