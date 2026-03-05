# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction and robust optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    
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
    
    # Strategy 1: Golden spiral initialization (from inspiration 2)
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
    
    # Add structured perturbations to improve initial spread and break symmetries
    points += np.random.normal(0, 0.02, points.shape)
    points = np.clip(points, 0, 1)
    
    best_points = None
    best_ratio = -float('inf')
    
    # Strategy 2: Try multiple initialization strategies to find better solutions
    # This approach inspired by inspiration 1's multi-attempt strategy
    initial_strategies = []
    
    # Golden spiral (main strategy)
    initial_strategies.append(points.copy())
    
    # Grid-based initialization (inspiration 2)
    grid_points = []
    for i in range(4):
        for j in range(4):
            x = (i + 0.5) / 4.0
            y = (j + 0.5) / 4.0
            grid_points.append([x, y])
    initial_strategies.append(np.array(grid_points))
    
    # Hexagonal-like arrangement (inspiration 3)
    hex_points = []
    for i in range(4):
        for j in range(4):
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (i * 0.333 + offset) % 1.0
            y = j * 0.333
            hex_points.append([x, y])
    initial_strategies.append(np.array(hex_points))
    
    # Try optimization with different initializations and methods
    methods_and_settings = [
        ('SLSQP', {'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}),
        ('L-BFGS-B', {'maxiter': 1500, 'ftol': 1e-10}),
        ('TNC', {'maxiter': 1000, 'ftol': 1e-8}),
        ('COBYLA', {'maxiter': 1000}),
        ('Nelder-Mead', {'maxiter': 2000, 'adaptive': True})
    ]
    
    # Try each initial strategy with multiple optimization methods
    for initial_points in initial_strategies:
        for method, options in methods_and_settings:
            try:
                x0 = initial_points.flatten()
                bounds = [(0, 1) for _ in range(32)]
                cons = {'type': 'ineq', 'fun': constraint_bounds}
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
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
    
    # Strategy 4: Additional robust optimization with more iterations for best candidates
    if best_points is not None:
        try:
            # Run one final fine-tuning optimization on the best configuration found so far
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            cons = {'type': 'ineq', 'fun': constraint_bounds}
            
            # Very aggressive optimization with high precision
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
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
    
    # Strategy 4: Additional robust optimization with more iterations for best candidates
    if best_points is not None:
        try:
            # Run one final fine-tuning optimization on the best configuration found so far
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            cons = {'type': 'ineq', 'fun': constraint_bounds}
            
            # Very aggressive optimization with high precision
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
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
    
    # Strategy 3: If no good solution found, try a more aggressive optimization
    if best_points is None:
        # Start with the golden spiral configuration
        points = initial_strategies[0]
        
        # Try with even more aggressive optimization settings
        try:
            x0 = points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            cons = {'type': 'ineq', 'fun': constraint_bounds}
            
            # Try with very tight tolerances and more iterations
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15}
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
    
    # Final fallback: return the best configuration found
    if best_points is None:
        # Return the golden spiral configuration as last resort
        return initial_strategies[0]
    
    return best_points


# EVOLVE-BLOCK-END
