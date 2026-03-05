# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Combines the best elements from both inspiration programs with a focus on robust optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective_function(points_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape flat array back to points
        points = points_flat.reshape(-1, 2)
        
        # Ensure points are within [0,1] x [0,1]
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Handle edge case where there are no distances
        if len(distances) == 0:
            return float('inf')
            
        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return float('inf')
            
        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        return -min_dist / max_dist
    
    # Generate initial points using a golden spiral pattern (Inspiration Program 2 style)
    # This provides good even distribution for 16 points
    n = 16
    np.random.seed(42)
    
    initial_points = np.zeros((n, 2))
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    for i in range(n):
        # Golden spiral approach for even distribution
        radius = 0.4 * np.sqrt(i / (n - 1)) if n > 1 else 0.4
        angle = i * 2 * np.pi / (phi - 1)  # Golden angle increment
        
        # Add some randomness to break symmetry (more aggressive than inspiration)
        noise_magnitude = 0.05
        radius += np.random.normal(0, noise_magnitude)
        angle += np.random.normal(0, noise_magnitude * 0.5)
        
        # Ensure radius stays reasonable
        radius = np.clip(radius, 0.05, 0.45)
        
        initial_points[i, 0] = 0.5 + radius * np.cos(angle)
        initial_points[i, 1] = 0.5 + radius * np.sin(angle)
    
    # Ensure all points are within the unit square
    initial_points = np.clip(initial_points, 0, 1)
    
    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(2*n)]
    
    # Apply simulated annealing optimization with multiple restarts for better results
    # Using 7 restarts as in Inspiration Program 2 for better global exploration
    best_result = None
    best_ratio = float('-inf')
    
    num_restarts = 7  # Match Inspiration Program 2 approach
    for restart in range(num_restarts):
        # Set different random seed for each restart
        np.random.seed(42 + restart)
        
        # Create perturbed initial points for each restart with larger perturbation
        # (More aggressive perturbation than Inspiration Program 2 to escape local optima)
        perturbed_points = initial_points.copy()
        perturbed_points += np.random.uniform(-0.1, 0.1, (n, 2))  # Larger perturbation
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = dual_annealing(
                objective_function,
                bounds,
                maxiter=1000,  # Match Inspiration Program 2 iterations
                seed=42 + restart,
                no_local_search=True
            )
        
        # Check if this result is better
        points = result.x.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        distances = pdist(points)
        
        # Additional safety check to avoid degenerate cases
        if len(distances) > 0 and np.max(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = result
    
    # Use the best result found
    if best_result is not None:
        optimized_points = best_result.x.reshape(-1, 2)
        optimized_points = np.clip(optimized_points, 0, 1)
    else:
        # Fallback to single optimization with original initial points
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = dual_annealing(
                objective_function,
                bounds,
                maxiter=1000,
                seed=42,
                no_local_search=True
            )
        optimized_points = result.x.reshape(-1, 2)
        optimized_points = np.clip(optimized_points, 0, 1)
    
    return optimized_points


# EVOLVE-BLOCK-END
