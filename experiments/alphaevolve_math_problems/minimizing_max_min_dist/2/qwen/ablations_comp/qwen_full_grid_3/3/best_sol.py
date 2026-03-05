# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust hybrid approach combining geometric initialization with multiple optimization restarts.

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
    
    # Create a high-quality initial configuration based on regular 16-gon
    n = 16
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 0.4  # Scaled down to fit in unit square
    
    # Initial configuration: regular polygon with some perturbation
    initial_points = np.zeros((n, 2))
    initial_points[:, 0] = 0.5 + radius * np.cos(angles)  # x coordinates
    initial_points[:, 1] = 0.5 + radius * np.sin(angles)  # y coordinates
    
    # Add moderate perturbation to break symmetry and escape local optima
    np.random.seed(42)
    perturbation = 0.07 * np.random.randn(n, 2)
    initial_points += perturbation
    initial_points = np.clip(initial_points, 0, 1)
    
    # Try multiple random restarts with different seeds to find better solutions
    best_points = None
    best_ratio = -float('inf')
    
    # Try 5 different random restarts with different seeds (balanced approach)
    # Based on the performance of inspirations, 5 seems to be optimal balance
    for restart in range(5):
        np.random.seed(42 + restart * 100)
        
        # Create perturbed version with larger perturbation for better exploration
        perturbed_points = initial_points.copy()
        # Use moderate perturbations for good exploration without over-perturbing
        perturbation = 0.1 * np.random.randn(n, 2)
        perturbed_points += perturbation
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        # Flatten for optimization
        x0 = perturbed_points.flatten()
        
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(2*n)]
        
        # Use SLSQP optimization method with increased robustness
        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                
                # Calculate ratio for this solution
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            continue
    
    # If no optimization worked, return the initial configuration
    if best_points is None:
        return initial_points
    
    return best_points


# EVOLVE-BLOCK-END
