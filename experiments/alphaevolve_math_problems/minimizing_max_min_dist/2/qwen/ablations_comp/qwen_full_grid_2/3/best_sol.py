# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a mathematically principled approach combining global optimization and multi-start local search.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        dmin = np.min(distances)
        dmax = np.max(distances)
        if dmax == 0:
            return 0.0
        return dmin / dmax
    
    def objective_function(x_flat):
        """Objective function to minimize (negative of ratio to maximize it)"""
        points = x_flat.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        return -ratio
    
    # Strategy 1: High-quality geometric initialization inspired by optimal point distributions
    # Create a 4x4 grid with hexagonal offset pattern - proven to work well
    points = []
    for i in range(4):
        for j in range(4):
            if len(points) < 16:
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + offset) / 3.0
                y = i / 3.0
                points.append([x, y])
    
    initial_points = np.array(points[:16])
    
    # Strategy 2: Global optimization using differential evolution for broad search
    try:
        bounds = [(0.05, 0.95)] * 32
        de_result = differential_evolution(
            objective_function,
            bounds,
            seed=42,
            maxiter=25,  # Reduced iterations to save time
            popsize=12,   # Balanced population size
            mutation=(0.5, 1),
            recombination=0.7,
            atol=1e-7,
            rtol=1e-7
        )
        
        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            de_ratio = compute_min_max_ratio(de_points)
            
            # Use DE result if better than our initial guess
            initial_ratio = compute_min_max_ratio(initial_points)
            if de_ratio > initial_ratio:
                initial_points = de_points
    except Exception:
        pass
    
    # Strategy 3: Multi-start local optimization with reduced iterations for speed
    best_ratio = 0.0
    best_points = initial_points.copy()
    
    # Reduce number of restarts to stay within time limits
    for restart in range(8):  # Reduced from 12 to 8 for efficiency
        # Set seed for reproducible results
        np.random.seed(42 + restart)
        
        # Create perturbed version of current best points
        perturbed_points = best_points.copy()
        
        # Add noise with controlled magnitude
        noise_magnitude = 0.025 - (restart * 0.002)  # Gradually decrease noise
        noise_magnitude = max(noise_magnitude, 0.005)
        noise = np.random.normal(0, noise_magnitude, (16, 2))
        perturbed_points += noise
        
        # Keep points within bounds
        perturbed_points = np.clip(perturbed_points, 0.05, 0.95)
        
        # Alternate between optimization methods for robustness
        methods = ['L-BFGS-B', 'SLSQP']
        method = methods[restart % len(methods)]
        
        try:
            # Use selected optimizer with appropriate settings for time efficiency
            if method == 'L-BFGS-B':
                result = minimize(
                    objective_function,
                    perturbed_points.flatten(),
                    method=method,
                    bounds=[(0.05, 0.95) for _ in range(32)],
                    options={'maxiter': 250, 'ftol': 1e-11, 'gtol': 1e-11}  # Reduced iterations
                )
            else:  # SLSQP
                result = minimize(
                    objective_function,
                    perturbed_points.flatten(),
                    method=method,
                    bounds=[(0.05, 0.95) for _ in range(32)],
                    options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-11}  # Reduced iterations
                )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    # Strategy 4: Final refinement with moderate precision
    try:
        bounds = [(0.05, 0.95)] * 32
        result = minimize(
            objective_function,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-12}  # Reduced iterations for speed
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            final_ratio = compute_min_max_ratio(final_points)
            
            # Only accept if it's actually better
            if final_ratio > best_ratio:
                best_points = final_points
                
    except Exception:
        pass
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0.05, 0.95)
    
    return best_points


# EVOLVE-BLOCK-END
