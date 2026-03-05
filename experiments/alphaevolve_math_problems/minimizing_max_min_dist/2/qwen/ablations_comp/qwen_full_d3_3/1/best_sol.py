# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses golden spiral initialization with multi-start simulated annealing optimization.
    Based on proven approach from inspirations with optimized parameters.

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
    
    # Use the golden spiral initialization as base
    base_points = golden_spiral_initialization()
    
    # Multi-start approach for robust optimization - only 3 starts to save time
    best_points = base_points.copy()
    best_ratio = compute_min_max_ratio(best_points)
    
    # Run multiple simulated annealing instances with fewer iterations for speed
    num_starts = 3  # Reduced from 8 to stay within time budget
    
    for start in range(num_starts):
        # Start with golden spiral + random perturbation
        if start == 0:
            current_points = base_points.copy()
        else:
            # Perturb the golden spiral configuration with varying variance
            current_points = base_points.copy()
            # Use different perturbation strength for variety but keep it moderate
            perturbation_strength = 0.015 + 0.005 * (start % 3) 
            current_points += np.random.normal(0, perturbation_strength, current_points.shape)
            current_points = np.clip(current_points, 0, 1)
        
        # Simulated Annealing parameters exactly matching INSPIRATION 2 for consistency
        initial_temp = 1.0
        final_temp = 1e-8  # Much lower final temp for better convergence
        alpha = 0.999  # Cooling rate - same as INSPIRATION 2
        max_iter = 100000  # Same number of iterations as INSPIRATION 2
        
        temp = initial_temp
        current_points_sa = current_points.copy()
        
        # Simulated Annealing loop with exact parameters from INSPIRATION 2
        for iteration in range(max_iter):
            # Compute current ratio
            current_ratio = compute_min_max_ratio(current_points_sa)
            
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points_sa.copy()
            
            # Generate candidate solution by perturbing one point
            candidate_points = current_points_sa.copy()
            point_idx = np.random.randint(0, 16)
            
            # Perturb one point with adaptive magnitude based on temperature
            # Same approach as INSPIRATION 2
            perturbation_magnitude = 0.01 * (temp / initial_temp) + 0.002
            candidate_points[point_idx] += np.random.normal(0, perturbation_magnitude, 2)
            
            # Keep points within unit square
            candidate_points[:, 0] = np.clip(candidate_points[:, 0], 0, 1)
            candidate_points[:, 1] = np.clip(candidate_points[:, 1], 0, 1)
            
            # Accept or reject based on Metropolis criterion
            candidate_ratio = compute_min_max_ratio(candidate_points)
            
            # Calculate acceptance probability with numerical stability
            if candidate_ratio > current_ratio:
                current_points_sa = candidate_points
            else:
                # Accept with probability based on temperature with overflow protection
                delta = candidate_ratio - current_ratio
                # Prevent overflow in exponential calculation
                if delta / temp < 700:  # Prevent overflow in exp()
                    acceptance_prob = math.exp(delta / temp)
                    if np.random.random() < acceptance_prob:
                        current_points_sa = candidate_points
            
            # Cool down temperature
            temp = max(final_temp, temp * alpha)
    
    return best_points


# EVOLVE-BLOCK-END
