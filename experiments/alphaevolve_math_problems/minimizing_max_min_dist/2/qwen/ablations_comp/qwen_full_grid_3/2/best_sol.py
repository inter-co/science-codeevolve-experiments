# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a regular 16-gon initialization with enhanced optimization strategy.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x_flat):
        """Objective function to maximize min/max distance ratio"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio since we want to maximize
        if d_max == 0:
            return -1.0
        return -d_min / d_max
    
    # Start with a good geometric configuration - regular 16-gon inscribed in circle
    n = 16
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 0.4  # Scaled down to fit in unit square
    
    # Initial configuration: regular polygon with some perturbation
    initial_points = np.zeros((n, 2))
    initial_points[:, 0] = 0.5 + radius * np.cos(angles)  # x coordinates
    initial_points[:, 1] = 0.5 + radius * np.sin(angles)  # y coordinates
    
    # Add moderate perturbation to break symmetry (following INSPIRATION 1/3)
    np.random.seed(42)
    perturbation = 0.07 * np.random.randn(n, 2)
    initial_points += perturbation
    initial_points = np.clip(initial_points, 0, 1)
    
    # Try multiple random restarts with different seeds to find better solutions
    best_points = None
    best_ratio = -float('inf')
    
    # Try 5 different random restarts (like INSPIRATION 1/3) with improved parameters
    for restart in range(5):
        np.random.seed(42 + restart * 100)
        
        # Create perturbed version with larger perturbation for better exploration (as in INSPIRATION 1/3)
        perturbed_points = initial_points.copy()
        perturbation = 0.1 * np.random.randn(n, 2)
        perturbed_points += perturbation
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        # Flatten for optimization
        x0 = perturbed_points.flatten()
        
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(2*n)]
        
        # Perform optimization with SLSQP using tighter tolerances (as in INSPIRATION 1/3)
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                
                # Calculate ratio for this solution
                distances = pdist(optimized_points)
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 0:
                    ratio = d_min / d_max
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
