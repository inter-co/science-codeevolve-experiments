# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing, minimize
import math
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
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
    
    # Generate initial configuration using hexagonal lattice approach (from inspiration 1)
    def generate_hexagonal_config():
        # Create a hexagonal pattern that's well-suited for dispersion problems
        sqrt3 = np.sqrt(3)
        
        # Create hexagonal grid points - 4x4 grid
        points_list = []
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                if len(points_list) >= 16:
                    break
                # Offset every other row
                offset = 0.5 if i % 2 == 1 else 0.0
                x = offset + j * 1.0
                y = i * sqrt3 * 0.5
                
                points_list.append([x, y])
        
        points = np.array(points_list[:16])
        
        # Normalize to fit in [0,1] x [0,1] 
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0 and y_range > 0:
                # Scale to fit in [0,1]x[0,1] but keep aspect ratio reasonable
                scale_x = 0.8 / x_range if x_range > 0 else 1.0
                scale_y = 0.8 / y_range if y_range > 0 else 1.0
                scale = min(scale_x, scale_y)
                
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) * scale
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) * scale
                
                # Center in [0,1]x[0,1]
                points[:, 0] += (0.1 - np.min(points[:, 0]))
                points[:, 1] += (0.1 - np.min(points[:, 1]))
        
        return points
    
    # Generate better initial configurations like inspiration 2
    def create_multiple_initial_configs():
        configs = []
        
        # Configuration 1: Hexagonal (inspired by inspiration 1)
        hex_config = generate_hexagonal_config()
        configs.append(("hexagonal", hex_config))
        
        # Configuration 2: Concentric rings (like inspiration 2)
        points = []
        # Outer ring (8 points)
        for i in range(8):
            angle = 2 * np.pi * i / 8
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        
        # Inner ring (8 points) offset
        for i in range(8):
            angle = 2 * np.pi * i / 8 + np.pi/8
            x = 0.5 + 0.2 * np.cos(angle)
            y = 0.5 + 0.2 * np.sin(angle)
            points.append([x, y])
        
        configs.append(("concentric", np.array(points[:16])))
        
        # Configuration 3: Grid with small random perturbations (like inspiration 1)
        points = []
        for i in range(4):
            for j in range(4):
                x = 0.1 + 0.8 * i / 3 + np.random.normal(0, 0.02, 1)[0]
                y = 0.1 + 0.8 * j / 3 + np.random.normal(0, 0.02, 1)[0]
                points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        configs.append(("grid_perturbed", points))
        
        # Configuration 4: Regular polygon pattern (like inspiration 1)
        points = []
        for i in range(16):
            angle = 2 * np.pi * i / 16
            radius = 0.4
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        configs.append(("regular_polygon", np.array(points)))
        
        return configs
    
    # Create bounds for optimization (points must stay in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    # Try multiple initial configurations and pick the best one
    np.random.seed(42)
    configs = create_multiple_initial_configs()
    
    best_ratio = 0
    best_config = None
    
    for name, config in configs:
        ratio = compute_min_max_ratio(config)
        if ratio > best_ratio:
            best_ratio = ratio
            best_config = config.copy()
    
    # Strategy 1: Use dual annealing with multiple restarts for global optimization
    # This is more robust than local optimization and matches INSPIRATION 2 approach
    try:
        # Run optimization with multiple restarts to improve chances of finding better solution
        best_result = None
        best_ratio = -np.inf
        
        # Use more restarts and better parameters like INSPIRATION 2 for better performance
        max_restarts = 8  # Increased from 5 to 8 for better exploration
        for restart in range(max_restarts):
            # Start with the best configuration found so far plus perturbation
            initial_guess = best_config.flatten() + np.random.normal(0, 0.02, 32)
            # Clip to bounds
            initial_guess = np.clip(initial_guess, 0, 1)
            
            # Use dual_annealing for global optimization (like INSPIRATION 2)
            result = dual_annealing(
                objective, 
                bounds, 
                maxiter=600,  # INSPIRATION 2 uses 600 iterations
                initial_temp=600,  # INSPIRATION 2 uses 600 initial temp
                seed=42 + restart,
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
    
    # Strategy 2: Fallback to local optimization with L-BFGS-B for final refinement
    try:
        # Flatten the best configuration found
        x0 = best_config.flatten()
        
        # Use L-BFGS-B with more iterations for refinement (like INSPIRATION 2)
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1200, 'ftol': 1e-12, 'gtol': 1e-12}  # Like INSPIRATION 2
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 2)
            distances = pdist(refined_points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 0:
                ratio = d_min / d_max
                # Return refined points if they're better than the hexagonal starting point
                original_distances = pdist(best_config)
                original_ratio = np.min(original_distances) / np.max(original_distances) if np.max(original_distances) > 0 else 0
                
                if ratio > original_ratio * 1.001:  # Small improvement threshold
                    return refined_points
                    
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
    
    # Return the best configuration as fallback
    return best_config


# EVOLVE-BLOCK-END
