# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a refined approach with better initialization and optimization strategy inspired by 
    successful implementations.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distances."""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def objective_function(x_flat):
        """Objective function to minimize (negative of ratio to maximize ratio)."""
        points = x_flat.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        return -ratio
    
    # Better geometric initialization - start with regular 16-gon then perturb
    n = 16
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 0.4  # Scaled down to fit in unit square
    
    # Initial configuration: regular polygon with some perturbation
    initial_points = np.zeros((n, 2))
    initial_points[:, 0] = 0.5 + radius * np.cos(angles)  # x coordinates
    initial_points[:, 1] = 0.5 + radius * np.sin(angles)  # y coordinates
    
    # Add some random perturbation to break symmetry (using fixed seed for reproducibility)
    np.random.seed(42)
    perturbation = 0.07 * np.random.randn(n, 2)  # Slightly larger than before
    initial_points += perturbation
    
    # Clip to unit square bounds
    initial_points = np.clip(initial_points, 0, 1)
    
    best_ratio = -float('inf')
    best_points = initial_points.copy()
    
    # Try multiple optimization attempts with different random seeds for robustness
    # Using the same restart strategy as the best performing inspirations
    restart_seeds = [42, 123, 456, 789, 999]  # Same as Inspiration 2 & 3
    
    for seed in restart_seeds:
        np.random.seed(seed)
        
        # Create a slightly different perturbed version for each attempt
        # Using larger perturbations for better exploration (as in Inspiration 1/2/3)
        perturbation = 0.1 * np.random.randn(n, 2)
        perturbed_points = initial_points + perturbation
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        # Single optimization run with better parameters (same as Inspiration 3)
        x0 = perturbed_points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}  # Tighter tolerances
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                # Ensure all points are within bounds
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    return best_points


# EVOLVE-BLOCK-END
