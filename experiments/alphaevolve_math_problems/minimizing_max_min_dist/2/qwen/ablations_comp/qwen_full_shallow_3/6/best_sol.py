# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction and multiple optimization strategies.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        distances = distances[distances > 0]  # Remove zero distances
        
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def objective_function(params):
        """Objective function to minimize (negative ratio)."""
        # Reshape parameters into points
        points = params.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        # Return negative because we want to maximize the ratio
        return -ratio
    
    def constraint_bounds(x_flat):
        points = x_flat.reshape(-1, 2)
        # Ensure all points are within [0,1] x [0,1]
        return np.concatenate([
            points[:, 0],           # x-coordinates >= 0
            1 - points[:, 0],       # x-coordinates <= 1
            points[:, 1],           # y-coordinates >= 0
            1 - points[:, 1]        # y-coordinates <= 1
        ])
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    best_points = None
    best_ratio = -float('inf')
    
    # Strategy 1: Golden spiral initialization with multiple optimization attempts
    n = 16
    points = np.zeros((n, 2))
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Generate points using golden spiral with better distribution
    for i in range(n):
        angle = 2 * np.pi * i / phi
        radius = np.sqrt(i / (n - 1)) if i < n - 1 else 1.0
        points[i] = [
            0.5 + 0.4 * radius * np.cos(angle),
            0.5 + 0.4 * radius * np.sin(angle)
        ]
    
    # Add structured perturbations to break symmetries
    points += np.random.normal(0, 0.02, points.shape)
    points = np.clip(points, 0, 1)
    
    # Try multiple optimization methods with golden spiral initialization
    methods_and_settings = [
        ('SLSQP', {'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14}),
        ('L-BFGS-B', {'maxiter': 2000, 'ftol': 1e-12}),
        ('TNC', {'maxiter': 1500, 'ftol': 1e-12})
    ]
    
    for method, options in methods_and_settings:
        try:
            x0 = points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            cons = {'type': 'ineq', 'fun': constraint_bounds}
            
            result = minimize(
                objective_function,
                x0,
                method=method,
                bounds=bounds,
                constraints=cons,
                options=options
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    # Strategy 2: Hexagonal grid initialization
    if best_points is None:
        # Create a more evenly distributed hexagonal-like pattern
        points_hex = []
        for i in range(4):
            for j in range(4):
                offset = 0.5 if i % 2 == 1 else 0.0
                x = j * 0.25 + offset * 0.25 + np.random.normal(0, 0.01)
                y = i * 0.25 * np.sqrt(3) / 2 + np.random.normal(0, 0.01)
                points_hex.append([x, y])
        
        points_hex = np.array(points_hex[:16])
        points_hex = np.clip(points_hex, 0, 1)
        
        try:
            x0 = points_hex.flatten()
            bounds = [(0, 1) for _ in range(32)]
            cons = {'type': 'ineq', 'fun': constraint_bounds}
            
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            pass
    
    # Strategy 3: Differential evolution for global optimization
    if best_points is not None:
        try:
            # Use more aggressive differential evolution with higher iterations
            bounds = [(0, 1) for _ in range(32)]
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,  # More iterations for better exploration
                popsize=20,   # Larger population for better diversity
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                tol=1e-12
            )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 2)
                de_points = np.clip(de_points, 0, 1)
                ratio = compute_min_max_ratio(de_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = de_points.copy()
                    
        except Exception:
            pass
    
    # Strategy 4: Multiple restarts with different patterns to escape local optima
    if best_points is not None:
        restart_strategies = [
            # Strategy A: Random uniform distribution with high precision
            lambda: np.random.uniform(0, 1, (16, 2)),
            # Strategy B: Perturbed square grid
            lambda: np.array([[i * 0.25 + np.random.normal(0, 0.02), 
                              j * 0.25 + np.random.normal(0, 0.02)] 
                             for i in range(4) for j in range(4)])[:16]
        ]
        
        for i, strategy in enumerate(restart_strategies):
            try:
                # Generate initial points using the strategy
                init_points = strategy()
                init_points = np.clip(init_points, 0, 1)
                
                # Optimize this initialization with L-BFGS-B
                result = minimize(
                    objective_function,
                    init_points.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(32)],
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    current_points = result.x.reshape(-1, 2)
                    current_ratio = compute_min_max_ratio(current_points)
                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_points = current_points.copy()
            except Exception:
                continue
    
    # Strategy 5: Final refinement with high precision optimization
    if best_points is not None and best_ratio > 0.05:  # Only refine if we have a decent solution
        try:
            # Try SLSQP again with even higher precision
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            cons = {'type': 'ineq', 'fun': constraint_bounds}
            
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                ratio = compute_min_max_ratio(refined_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points.copy()
                    
        except Exception:
            pass
    
    # Final fallback to golden spiral if nothing worked
    if best_points is None:
        return points
    
    return best_points


# EVOLVE-BLOCK-END
