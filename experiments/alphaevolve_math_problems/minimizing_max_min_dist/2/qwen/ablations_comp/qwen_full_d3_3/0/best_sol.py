# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import math
import warnings


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses golden spiral initialization with enhanced hybrid optimization approach combining SA and multiple L-BFGS refinements.
    Inspired by successful approaches from both inspirations.

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
    
    def golden_spiral_initialization():
        """Initialize points using golden spiral for better distribution"""
        n = 16
        points = []
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        
        for i in range(n):
            angle = 2 * math.pi * i / phi
            radius = math.sqrt(i / (n - 1)) if n > 1 else 0
            x = 0.5 + 0.4 * radius * math.cos(angle)
            y = 0.5 + 0.4 * radius * math.sin(angle)
            points.append([x, y])
        
        return np.array(points)
    
    def objective(points_flat):
        """Objective function to minimize (negative of ratio)"""
        points = points_flat.reshape(-1, 2)
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist  # Negative because we minimize
    
    # Strategy 1: Use golden spiral initialization with better perturbations
    best_points = golden_spiral_initialization()
    
    # Add small random perturbations to break symmetry with better magnitude
    best_points += np.random.normal(0, 0.01, best_points.shape)
    best_points = np.clip(best_points, 0, 1)
    
    # Track best ratio found
    best_ratio = compute_min_max_ratio(best_points)
    
    # Strategy 2: Enhanced Simulated Annealing with parameters from best performing inspirations
    # Using more aggressive cooling and higher iteration count from INSPIRATION 2
    initial_temp = 1.0
    final_temp = 1e-8  # Even lower final temp for tighter convergence (from INSPIRATION 2)
    alpha = 0.9992  # Slightly faster cooling (from INSPIRATION 2)
    max_iter = 120000  # More iterations for better convergence (from INSPIRATION 2)
    
    temp = initial_temp
    current_points = best_points.copy()
    
    # Simulated Annealing loop with improved numerical stability
    for iteration in range(max_iter):
        # Compute current ratio
        current_ratio = compute_min_max_ratio(current_points)
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = current_points.copy()
        
        # Generate candidate solution by perturbing one point
        candidate_points = current_points.copy()
        point_idx = np.random.randint(0, 16)
        
        # Perturb one point with adaptive magnitude based on temperature
        # Using better scaling from INSPIRATION 2
        perturbation_magnitude = 0.012 * (temp / initial_temp) + 0.0015
        candidate_points[point_idx] += np.random.normal(0, perturbation_magnitude, 2)
        
        # Keep points within unit square
        candidate_points[:, 0] = np.clip(candidate_points[:, 0], 0, 1)
        candidate_points[:, 1] = np.clip(candidate_points[:, 1], 0, 1)
        
        # Accept or reject based on Metropolis criterion
        candidate_ratio = compute_min_max_ratio(candidate_points)
        
        # Calculate acceptance probability with better numerical stability
        if candidate_ratio > current_ratio:
            current_points = candidate_points
        else:
            # Accept with probability based on temperature
            delta = candidate_ratio - current_ratio
            # Prevent overflow in exponential calculation (same approach as INSPIRATION 2)
            if delta / temp < 700:  # Prevent overflow in exp()
                acceptance_prob = math.exp(delta / temp)
                if np.random.random() < acceptance_prob:
                    current_points = candidate_points
        
        # Cool down temperature
        temp = max(final_temp, temp * alpha)
    
    # Strategy 3: Multiple final refinements with different L-BFGS settings to maximize chance of finding better solution
    bounds = [(0, 1) for _ in range(32)]
    
    # Try several different refinement approaches with better parameters from INSPIRATION 2
    refinement_attempts = [
        {'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-13},  # From INSPIRATION 2
        {'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14},  # From INSPIRATION 2
        {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}, # From INSPIRATION 2
    ]
    
    for ref_params in refinement_attempts:
        try:
            result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options=ref_params
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                refined_ratio = compute_min_max_ratio(refined_points)
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points
        except Exception as e:
            warnings.warn(f"Refinement attempt failed: {e}")
            continue
    
    return best_points


# EVOLVE-BLOCK-END
