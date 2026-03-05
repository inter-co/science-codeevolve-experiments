# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing, minimize
import warnings
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining geometric initialization with advanced optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all pairs"""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return 0.0
            
        return d_min / d_max
    
    def objective(params):
        """Objective function to minimize (negative of ratio to maximize ratio)"""
        # Reshape parameters back to points
        points = params.reshape(-1, 2)
        
        # Compute distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return np.inf  # Return large value to penalize invalid configurations
            
        ratio = d_min / d_max
        return -ratio  # Negative because we want to maximize ratio
    
    def create_hexagonal_config():
        """Create a hexagonal lattice pattern that works well for point dispersion"""
        points = []
        # 4 rows, 4 columns with alternating offset for hexagonal packing
        for i in range(4):
            for j in range(4):
                # Offset every other row
                x = 0.1 + 0.8 * j / 3
                y = 0.1 + 0.8 * i / 3
                if i % 2 == 1:
                    x += 0.8 * 0.25 / 3  # Offset odd rows
                points.append([x, y])
        return np.array(points[:16])  # Ensure exactly 16 points
    
    def create_spiral_config():
        """Create a spiral pattern to get good initial spread"""
        points = []
        angles = np.linspace(0, 4*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.4, 16)
        for angle, radius in zip(angles, radii):
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def create_fibonacci_spiral():
        """Create points using Fibonacci spiral pattern"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(16):
            angle = i * 2.4  # Modified angle for better spread
            radius = np.sqrt(i / 15.0) * 0.4  # Radius increases slowly
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def create_regular_polygon():
        """Create points arranged in a regular polygon configuration"""
        points = []
        # Arrange 16 points in a circle with some variation
        for i in range(16):
            angle = 2 * np.pi * i / 16
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def create_concentric_rings():
        """Create points in concentric ring patterns"""
        points = []
        # Outer ring: 8 points
        for k in range(8):
            angle = k * np.pi/4
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        
        # Inner ring: 8 points
        for k in range(8):
            angle = k * np.pi/4
            x = 0.5 + 0.2 * np.cos(angle)
            y = 0.5 + 0.2 * np.sin(angle)
            points.append([x, y])
            
        # Ensure exactly 16 points
        return np.array(points[:16])
    
    def create_random_config():
        """Create a completely random configuration"""
        np.random.seed(42)
        points = np.random.rand(16, 2)
        return points
    
    def create_grid_with_noise():
        """Create a grid pattern with added noise"""
        points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + 0.125 + np.random.normal(0, 0.02)
                y = j * 0.25 + 0.125 + np.random.normal(0, 0.02)
                points.append([x, y])
        return np.array(points)
    
    # Generate better initial configurations like inspiration 1
    def create_multiple_initial_configs():
        configs = []
        
        # Use all the configuration generation functions from inspiration 1
        configs.append(("hexagonal", create_hexagonal_config()))
        configs.append(("spiral", create_spiral_config()))
        configs.append(("fibonacci", create_fibonacci_spiral()))
        configs.append(("polygon", create_regular_polygon()))
        configs.append(("concentric", create_concentric_rings()))
        configs.append(("random", create_random_config()))
        configs.append(("grid_noise", create_grid_with_noise()))
        
        return configs
    
    # Create bounds for optimization (points must stay in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    # Try multiple initial configurations and pick the best one (like inspiration 1)
    np.random.seed(42)
    configs = create_multiple_initial_configs()
    
    best_ratio = 0
    best_config = None
    
    for name, config in configs:
        ratio = compute_min_max_ratio(config)
        if ratio > best_ratio:
            best_ratio = ratio
            best_config = config.copy()
    
    # Strategy 1: Use enhanced dual annealing with multiple restarts and better parameters
    # Inspired by inspiration 1's approach with more aggressive settings
    try:
        # Run optimization with multiple restarts to improve chances of finding better solution
        best_result = None
        best_ratio = -np.inf
        
        # Use more restarts and better parameters for dual annealing (like inspiration 1)
        restart_configs = [
            (42, 2000, 1000),  # High temp for broad search
            (43, 1500, 800),   # Medium temp
            (44, 1000, 600),   # Lower temp
            (45, 1200, 1000),  # Another high temp
            (46, 800, 500),    # Low temp for fine tuning
        ]
        
        for restart, seed_val, temp in restart_configs:
            # Start with the best configuration found so far plus perturbation
            initial_guess = best_config.flatten() + np.random.normal(0, 0.02, 32)
            # Clip to bounds
            initial_guess = np.clip(initial_guess, 0, 1)
            
            result = dual_annealing(
                objective, 
                bounds, 
                maxiter=1000,  # More iterations for better convergence
                initial_temp=temp,
                seed=seed_val,
                no_local_search=True
            )
            
            # Evaluate the result
            if result.success:
                test_points = result.x.reshape(-1, 2)
                distances = pdist(test_points)
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_result = result
        
        # If optimization improved results, use those
        if best_result is not None and best_result.success:
            optimized_points = best_result.x.reshape(-1, 2)
            return optimized_points
                
    except Exception as e:
        warnings.warn(f"Dual annealing optimization failed: {e}")
    
    # Strategy 2: Multi-stage refinement approach like inspiration 2
    # First try local optimization around the best configuration found
    try:
        # Flatten the best configuration found
        x0 = best_config.flatten()
        
        # Use L-BFGS-B with more iterations for refinement
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 2)
            distances = pdist(refined_points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 0:
                ratio = d_min / d_max
                # Return refined points if they're better than the starting point
                original_distances = pdist(best_config)
                original_ratio = np.min(original_distances) / np.max(original_distances) if np.max(original_distances) > 0 else 0
                
                if ratio > original_ratio * 1.001:  # Small improvement threshold
                    return refined_points
                    
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
    
    # Strategy 3: Try alternative optimization methods if needed
    try:
        # Try with TNC method as an alternative
        x0 = best_config.flatten()
        
        result = minimize(
            objective,
            x0,
            method='TNC',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 2)
            distances = pdist(refined_points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 0:
                ratio = d_min / d_max
                original_distances = pdist(best_config)
                original_ratio = np.min(original_distances) / np.max(original_distances) if np.max(original_distances) > 0 else 0
                
                if ratio > original_ratio * 1.001:
                    return refined_points
                    
    except Exception as e:
        warnings.warn(f"TNC optimization failed: {e}")
    
    # Return the best configuration as fallback
    return best_config


# EVOLVE-BLOCK-END
