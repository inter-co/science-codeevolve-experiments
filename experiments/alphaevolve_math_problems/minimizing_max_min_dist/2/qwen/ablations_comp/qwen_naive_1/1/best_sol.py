# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
from numba import jit


@jit(nopython=True)
def compute_distances_jit(points):
    """Compute pairwise distances using numba for speed"""
    n = points.shape[0]
    distances = np.zeros((n * (n - 1) // 2,))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            distances[idx] = np.sqrt(dx * dx + dy * dy)
            idx += 1
    return distances


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses advanced optimization techniques with evolutionary algorithms and improved constraint handling.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    d = 2
    
    # Define bounds for all coordinates (0 to 1 for both x and y)
    bounds = [(0, 1) for _ in range(2*n)]
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x_flat.reshape((n, d))
        
        # Use faster distance computation
        distances = compute_distances_jit(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 1e-12:
            return -1e12  # Very bad solution
            
        # Return negative ratio (since we want to maximize)
        return -min_dist / max_dist
    
    # Better initialization strategy - start with a good configuration
    def initialize_points():
        # Start with a regular grid pattern, then slightly perturb
        # Create a 4x4 grid with some perturbation
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        
        points = np.array([[x, y] for y in grid_y for x in grid_x])
        
        # Add small random perturbations to avoid degenerate cases
        noise = np.random.normal(0, 0.02, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        
        return points
    
    # Improved initialization using circle packing approach
    def initialize_points_circle():
        # Place points around a circle with some randomness
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        # Add some randomness to avoid perfect symmetry
        angles += np.random.normal(0, 0.1, n)
        radii = np.ones(n) * 0.4  # Keep points within a reasonable radius
        # Add some variation in radii to prevent perfect circular arrangement
        radii *= (0.8 + np.random.rand(n) * 0.4)
        
        points = np.column_stack([
            0.5 + radii * np.cos(angles),
            0.5 + radii * np.sin(angles)
        ])
        
        # Clip to bounds
        points = np.clip(points, 0, 1)
        return points
    
    # Try multiple optimization approaches with different starting points
    best_points = None
    best_ratio = -np.inf
    
    # Strategy 1: Differential Evolution with better parameters
    try:
        # Run with multiple random seeds for better exploration
        seeds = [42, 123, 456, 789, 999]
        for seed in seeds:
            np.random.seed(seed)
            de_result = differential_evolution(
                objective,
                bounds,
                seed=seed,
                maxiter=100,
                popsize=30,
                mutation=(0.5, 1),
                recombination=0.7,
                atol=1e-12,
                rtol=1e-12
            )
            
            if de_result.success:
                current_points = de_result.x.reshape((n, d))
                current_ratio = -objective(de_result.x)
                if current_ratio > best_ratio:
                    best_points = current_points
                    best_ratio = current_ratio
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
    
    # Strategy 2: If no good DE solution, use our custom initialization
    if best_points is None:
        best_points = initialize_points()
        best_ratio = -objective(best_points.flatten())
    
    # Strategy 3: Local refinement with multiple methods
    methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
    
    for method in methods_to_try:
        try:
            result = minimize(
                objective,
                best_points.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success:
                current_points = result.x.reshape((n, d))
                current_ratio = -objective(result.x)
                if current_ratio > best_ratio:
                    best_points = current_points
                    best_ratio = current_ratio
        except Exception as e:
            warnings.warn(f"Method {method} failed: {e}")
            continue
    
    # Strategy 4: Multiple random restarts with local optimization
    for restart in range(5):  # More restarts for better exploration
        # Start from a different initialization
        if restart == 0:
            # Regular grid initialization
            start_points = initialize_points()
        elif restart == 1:
            # Circle initialization
            start_points = initialize_points_circle()
        elif restart == 2:
            # Random initialization with some clustering avoidance
            start_points = np.random.uniform(0.1, 0.9, (n, d))
        else:
            # Perturbed version of best solution so far
            start_points = best_points + np.random.normal(0, 0.01, (n, d))
            start_points = np.clip(start_points, 0, 1)
        
        try:
            result = minimize(
                objective,
                start_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success:
                current_points = result.x.reshape((n, d))
                current_ratio = -objective(result.x)
                if current_ratio > best_ratio:
                    best_points = current_points
                    best_ratio = current_ratio
        except Exception as e:
            warnings.warn(f"Restart {restart} failed: {e}")
            continue
    
    # Strategy 5: Final aggressive refinement with higher precision
    try:
        result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-16}
        )
        
        if result.success:
            final_points = result.x.reshape((n, d))
            final_ratio = -objective(result.x)
            if final_ratio > best_ratio:
                best_points = final_points
                best_ratio = final_ratio
    except Exception as e:
        warnings.warn(f"Final refinement failed: {e}")
    
    # Strategy 6: Additional global search with simulated annealing approach
    # Try a few more random configurations
    for _ in range(3):
        try:
            # Generate completely random configuration
            random_points = np.random.uniform(0, 1, (n, d))
            result = minimize(
                objective,
                random_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success:
                current_points = result.x.reshape((n, d))
                current_ratio = -objective(result.x)
                if current_ratio > best_ratio:
                    best_points = current_points
                    best_ratio = current_ratio
        except Exception as e:
            warnings.warn(f"Random search failed: {e}")
            continue
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
