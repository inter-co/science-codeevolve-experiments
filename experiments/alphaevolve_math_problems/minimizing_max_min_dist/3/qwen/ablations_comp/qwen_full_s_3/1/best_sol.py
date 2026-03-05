# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import math
from typing import Tuple
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction, physics-based refinement, and optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return 0.0
            
        return min_dist / max_dist
    
    def objective_function(x):
        """Objective function to minimize (negative of ratio to maximize ratio)."""
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        ratio = compute_min_max_ratio(points)
        # Return negative because we want to maximize
        return -ratio
    
    def sphere_constraint(x):
        """Constraint to keep points on unit sphere."""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return difference from unit radius (should be close to 0)
        return norms - 1.0
    
    def energy_minimization(points, max_iter=500, learning_rate=0.02):
        """Physics-based energy minimization approach with repulsive forces."""
        points = points.copy()
        for _ in range(max_iter):
            # Compute pairwise forces (repulsive)
            n = len(points)
            forces = np.zeros_like(points)
            
            for i in range(n):
                for j in range(i+1, n):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq > 1e-10:  # Avoid division by zero
                        dist = np.sqrt(dist_sq)
                        force_magnitude = 1.0 / (dist_sq * dist)  # Repulsive force
                        force = force_magnitude * diff / dist
                        forces[i] += force
                        forces[j] -= force
            
            # Update positions
            points += learning_rate * forces
            
            # Project back to sphere
            norms = np.linalg.norm(points, axis=1)
            norms = np.where(norms == 0, 1, norms)
            points = points / norms[:, np.newaxis]
        
        return points
    
    # Multiple initialization strategies (inspired by INSPIRATION 1)
    best_ratio = 0.0
    best_solution = None
    
    # Strategy 1: Improved icosahedral construction with better pole placement (like INSPIRATION 1)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    # Vertices of regular icosahedron scaled to unit sphere
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0])
    
    # Add two more points - better positioned for symmetry and distribution
    # Place them closer to poles but with better spacing (like INSPIRATION 1)
    additional_points = np.array([[0, 0, 0.9], [0, 0, -0.9]])
    
    # Combine all points
    initial_points_1 = np.vstack([ico_vertices, additional_points])
    
    # Strategy 2: Fibonacci spiral with improved distribution (like INSPIRATION 1)
    fib_points = np.zeros((14, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(14):
        y = 1 - (i / (14 - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = np.arccos(y)
        phi_val = (i * golden_ratio) % (2 * np.pi)
        fib_points[i] = [radius * np.sin(theta) * np.cos(phi_val),
                        radius * np.sin(theta) * np.sin(phi_val),
                        radius * np.cos(theta)]
    
    # Strategy 3: Random initialization with better spread (like INSPIRATION 2)
    np.random.seed(42)  # Fixed seed for reproducibility
    random_points = np.random.randn(14, 3)
    norms = np.linalg.norm(random_points, axis=1)
    norms = np.where(norms == 0, 1, norms)
    random_points = random_points / norms[:, np.newaxis]
    
    strategies = [initial_points_1, fib_points, random_points]
    
    # Try multiple optimization restarts with different strategies (like INSPIRATION 1)
    for strategy_idx, initial_points in enumerate(strategies):
        # Multiple restarts for each strategy (more restarts than INSPIRATION 2 for better exploration)
        for restart in range(5):  # Like INSPIRATION 1 with more restarts
            # Start with base configuration
            current_points = initial_points.copy()
            
            # Add small random perturbations for each restart (larger perturbations like INSPIRATION 1)
            np.random.seed(strategy_idx * 100 + restart)
            # Use moderate perturbations for better exploration (like INSPIRATION 1)
            perturbation = np.random.normal(0, 0.02, (14, 3))  # Slightly larger perturbation
            current_points += perturbation
            
            # Normalize to unit sphere again to maintain constraint
            norms = np.linalg.norm(current_points, axis=1)
            current_points = current_points / norms[:, np.newaxis]
            
            # Apply physics-based energy minimization first to get a better starting point
            # More iterations like INSPIRATION 1 for better refinement
            current_points = energy_minimization(current_points, max_iter=500, learning_rate=0.02)
            
            # Flatten for optimization
            x0 = current_points.flatten()
            
            # Define constraint for unit sphere
            constraint = {'type': 'eq', 'fun': sphere_constraint}
            
            try:
                # Use multiple optimization methods for robustness (like INSPIRATION 1)
                optimizers = ['trust-constr', 'SLSQP']
                
                for method in optimizers:
                    try:
                        # Settings optimized to balance quality and speed (like INSPIRATION 1)
                        result = minimize(
                            objective_function,
                            x0,
                            method=method,
                            constraints=constraint,
                            options={
                                'maxiter': 500,   # Like INSPIRATION 2 - reasonable number of iterations
                                'ftol': 1e-9,     # Tight tolerance for quality (like INSPIRATION 1)
                                'gtol': 1e-9,     # Tight tolerance for quality (like INSPIRATION 1)
                                'disp': False
                            },
                            tol=1e-9
                        )
                        
                        if result.success:
                            optimized_points = result.x.reshape(-1, 3)
                            ratio = compute_min_max_ratio(optimized_points)
                            
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_solution = optimized_points.copy()
                        break  # Success, move to next strategy
                    except Exception:
                        # Continue with other methods if one fails
                        continue
                        
            except Exception:
                continue
    
    # If no good optimization found, return the best initial configuration
    if best_solution is None:
        # Return the first strategy as fallback
        return initial_points_1
    
    return best_solution


# EVOLVE-BLOCK-END
