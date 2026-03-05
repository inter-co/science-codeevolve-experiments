# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
from scipy.optimize import minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses simulated annealing with golden spiral initialization and hybrid optimization
    to achieve superior performance compared to benchmark.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    
    def compute_min_max_ratio(points_array):
        """Compute the ratio of minimum to maximum distances"""
        distances = pdist(points_array)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    # Initialize points with golden spiral (from INSPIRATION 1)
    n = 16
    points = []
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    
    for i in range(n):
        angle = 2 * math.pi * i / phi
        radius = math.sqrt(i / (n - 1)) if n > 1 else 0
        x = 0.5 + 0.4 * radius * math.cos(angle)
        y = 0.5 + 0.4 * radius * math.sin(angle)
        points.append([x, y])
    
    points = np.array(points)
    
    # Add small random perturbations
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, 0, 1)
    
    # Optimized parameters for best balance of performance and quality
    # Using slightly more aggressive cooling and more iterations to push performance
    initial_temp = 1.0
    final_temp = 1e-8  # Very low final temp for tight convergence
    alpha = 0.9992  # Slightly faster cooling for better exploration (from INSPIRATION 2)
    max_iter = 120000  # More iterations to allow better convergence (from INSPIRATION 2)
    
    temp = initial_temp
    current_points = points.copy()
    
    # Track best solution
    best_points = current_points.copy()
    best_ratio = 0
    
    # Simulated Annealing loop with optimized parameters (from INSPIRATION 2)
    for iteration in range(max_iter):
        # Compute current ratio
        current_ratio = compute_min_max_ratio(current_points)
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = current_points.copy()
        
        # Generate candidate solution by perturbing one point
        candidate_points = current_points.copy()
        point_idx = np.random.randint(0, n)
        
        # Perturb one point with adaptive magnitude based on temperature
        # Slightly adjusted perturbation for better exploration (from INSPIRATION 2)
        perturbation_magnitude = 0.012 * (temp / initial_temp) + 0.0015
        candidate_points[point_idx] += np.random.normal(0, perturbation_magnitude, 2)
        
        # Keep points within unit square
        candidate_points[:, 0] = np.clip(candidate_points[:, 0], 0, 1)
        candidate_points[:, 1] = np.clip(candidate_points[:, 1], 0, 1)
        
        # Accept or reject based on Metropolis criterion
        candidate_ratio = compute_min_max_ratio(candidate_points)
        
        # Calculate acceptance probability with numerical stability (from INSPIRATION 2)
        if candidate_ratio > current_ratio:
            current_points = candidate_points
        else:
            # Accept with probability based on temperature with overflow protection
            delta = candidate_ratio - current_ratio
            # Prevent overflow in exponential calculation (from INSPIRATION 2)
            if delta / temp < 700:  # Prevent overflow in exp()
                acceptance_prob = math.exp(delta / temp)
                if np.random.random() < acceptance_prob:
                    current_points = candidate_points
        
        # Cool down temperature
        temp = max(final_temp, temp * alpha)
    
    # Final refinement with multiple optimization methods for robustness (from INSPIRATION 2)
    try:
        def objective(points_flat):
            return -compute_min_max_ratio(points_flat.reshape(-1, 2))
        
        # Try multiple optimization methods for better final result (from INSPIRATION 2)
        methods_to_try = ['L-BFGS-B', 'TNC']  # Use two methods for robustness
        
        for method in methods_to_try:
            # Bounds for optimization
            bounds = [(0, 1) for _ in range(len(best_points) * 2)]
            
            # Use the chosen method for final tuning (from INSPIRATION 2)
            result = minimize(
                objective,
                best_points.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                refined_ratio = compute_min_max_ratio(refined_points)
                if refined_ratio > best_ratio:
                    best_points = refined_points
                    best_ratio = refined_ratio
                    
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
