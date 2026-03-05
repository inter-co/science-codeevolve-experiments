# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses enhanced simulated annealing with multiple initialization strategies and hybrid optimization
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
    
    def uniform_random_initialization():
        """Alternative initialization strategy"""
        return np.random.uniform(0, 1, (16, 2))
    
    def hexagonal_initialization():
        """Hexagonal grid initialization for good coverage"""
        points = []
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 1)
        spacing_y = 1.0 / (rows - 1)
        
        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x
                y = i * spacing_y
                # Add slight jitter
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                points.append([x, y])
        
        return np.clip(np.array(points), 0, 1)
    
    # Try multiple initialization strategies and keep the best
    initial_strategies = [
        golden_spiral_initialization,
        uniform_random_initialization,
        hexagonal_initialization
    ]
    
    best_initial_points = None
    best_initial_ratio = 0
    
    for strategy in initial_strategies:
        points = strategy()
        ratio = compute_min_max_ratio(points)
        if ratio > best_initial_ratio:
            best_initial_ratio = ratio
            best_initial_points = points.copy()
    
    # Use the best initialization
    points = best_initial_points
    
    # Enhanced parameters for better optimization (from inspiration program 2)
    initial_temp = 1.0
    final_temp = 1e-10  # Even lower final temp for better convergence
    alpha = 0.997  # Slightly faster cooling (from inspiration program 2)
    max_iter = 150000  # More iterations for better exploration (from inspiration program 2)
    
    temp = initial_temp
    current_points = points.copy()
    
    # Track best solution
    best_points = current_points.copy()
    best_ratio = best_initial_ratio
    
    # Simulated Annealing loop with improvements (from inspiration program 2)
    for iteration in range(max_iter):
        # Compute current ratio
        current_ratio = compute_min_max_ratio(current_points)
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = current_points.copy()
        
        # Generate candidate solution by perturbing one point
        candidate_points = current_points.copy()
        point_idx = np.random.randint(0, 16)
        
        # Adaptive perturbation with better scaling (from inspiration program 2)
        # Start with larger perturbations and decrease with temperature
        base_perturbation = 0.02
        perturbation_magnitude = base_perturbation * (temp / initial_temp) + 0.001
        
        candidate_points[point_idx] += np.random.normal(0, perturbation_magnitude, 2)
        
        # Keep points within unit square
        candidate_points[:, 0] = np.clip(candidate_points[:, 0], 0, 1)
        candidate_points[:, 1] = np.clip(candidate_points[:, 1], 0, 1)
        
        # Accept or reject based on Metropolis criterion
        candidate_ratio = compute_min_max_ratio(candidate_points)
        
        # Calculate acceptance probability with better numerical stability (from inspiration program 2)
        if candidate_ratio > current_ratio:
            current_points = candidate_points
        else:
            # Accept with probability based on temperature
            delta = candidate_ratio - current_ratio
            # More careful overflow protection
            if delta / temp < 700 and temp > 1e-15:  # Prevent overflow and underflow
                acceptance_prob = math.exp(delta / temp)
                if np.random.random() < acceptance_prob:
                    current_points = candidate_points
        
        # Cool down temperature
        temp = max(final_temp, temp * alpha)
    
    # Multiple refinement strategies for better final result (from inspiration program 2)
    try:
        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 1e10
            ratio = min_dist / max_dist
            return -ratio  # Negative because we want to maximize
        
        # Try multiple optimization methods for robustness (from inspiration program 2)
        methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
        
        for method in methods_to_try:
            bounds = [(0, 1) for _ in range(32)]
            
            result = minimize(
                objective,
                best_points.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
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
