# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.distance import pdist
import warnings
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a sophisticated hybrid approach combining geometric initialization with advanced 
    optimization techniques to achieve state-of-the-art results, drawing from the best 
    practices of both inspiration programs.
    
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
    
    # Generate initial configuration using hexagonal lattice approach (inspired by inspiration 2)
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
    
    # Create bounds for optimization (points must stay in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    # Strategy 1: Start with hexagonal lattice (better than simple grid)
    initial_points = generate_hexagonal_config()
    
    # Strategy 2: Use dual annealing with multiple restarts for global optimization
    # This is more robust than differential evolution and matches inspiration 2 approach
    try:
        # Run optimization with multiple restarts to improve chances of finding better solution
        best_result = None
        best_ratio = -np.inf
        
        # Use more restarts and better parameters for dual annealing (from inspiration 2)
        for restart in range(8):  # More restarts for better exploration (increased from 5)
            # Add some noise to initial points for diversity
            np.random.seed(42 + restart)  # Better seed mixing
            initial_guess = initial_points.flatten() + np.random.normal(0, 0.02, 32)
            # Clip to bounds
            initial_guess = np.clip(initial_guess, 0, 1)
            
            result = dual_annealing(
                objective, 
                bounds, 
                maxiter=1200,  # More iterations for better convergence (from inspiration 2)
                initial_temp=1200,  # Higher initial temperature (from inspiration 2)
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
    
    # Strategy 3: Fallback to local optimization with L-BFGS-B for final refinement
    try:
        # Flatten the initial hexagonal configuration
        x0 = initial_points.flatten()
        
        # Use L-BFGS-B with more iterations for refinement (from inspiration 2)
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2500, 'ftol': 1e-14, 'gtol': 1e-14}  # More iterations and tighter tolerances (from inspiration 2)
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 2)
            distances = pdist(refined_points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 0:
                ratio = d_min / d_max
                # Return refined points if they're better than the hexagonal starting point
                original_distances = pdist(initial_points)
                original_ratio = np.min(original_distances) / np.max(original_distances) if np.max(original_distances) > 0 else 0
                
                if ratio > original_ratio * 1.001:  # Small improvement threshold
                    return refined_points
                    
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
    
    # Return the hexagonal configuration as fallback
    return initial_points


# EVOLVE-BLOCK-END
