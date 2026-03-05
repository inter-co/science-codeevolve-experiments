# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.distance import pdist
import warnings
import math

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
    
    # Generate multiple initial configurations inspired by successful approaches
    def generate_multiple_initial_configs():
        configs = []
        
        # Configuration 1: Hexagonal lattice (from inspiration 1)
        sqrt3 = math.sqrt(3)
        points_list = []
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                if len(points_list) >= 16:
                    break
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
                scale_x = 0.8 / x_range if x_range > 0 else 1.0
                scale_y = 0.8 / y_range if y_range > 0 else 1.0
                scale = min(scale_x, scale_y)
                
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) * scale
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) * scale
                
                points[:, 0] += (0.1 - np.min(points[:, 0]))
                points[:, 1] += (0.1 - np.min(points[:, 1]))
        
        configs.append(("hexagonal", points.copy()))
        
        # Configuration 2: Perturbed regular grid (from inspiration 2)
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + 0.125
                y = j * 0.25 + 0.125
                grid_points.append([x, y])
        configs.append(("grid_perturbed", np.array(grid_points)))
        
        # Configuration 3: Circle-based with perturbation (from inspiration 2)
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4
        points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        points += np.random.normal(0, 0.03, points.shape)
        points[:, 0] += 0.5
        points[:, 1] += 0.5
        points = np.clip(points, 0, 1)
        configs.append(("circle_perturbed", points))
        
        return configs
    
    # Create bounds for optimization (points must stay in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    # Try multiple initial configurations and pick the best one
    initial_configs = generate_multiple_initial_configs()
    
    best_ratio = 0
    best_points = None
    
    # Multi-start optimization with multiple restarts per configuration for better exploration
    for i, (config_name, initial_config) in enumerate(initial_configs):
        # Try multiple restarts for each configuration to get better results
        for restart in range(3):  # Three restarts per configuration for better exploration
            # Add slight perturbation for diversity
            np.random.seed(i * 100 + restart)
            perturbed_config = initial_config + np.random.normal(0, 0.01, initial_config.shape)
            perturbed_config = np.clip(perturbed_config, 0, 1)
            
            # Flatten for optimization
            x0 = perturbed_config.flatten()
            
            # Optimization with dual annealing for global search - slightly more thorough
            try:
                result = dual_annealing(
                    objective, 
                    bounds, 
                    maxiter=600,  # Slightly more iterations for better convergence
                    initial_temp=300,
                    seed=42 + i * 10 + restart,
                    no_local_search=True
                )
                
                if result.success:
                    test_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(test_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = test_points.copy()
                        
            except Exception as e:
                warnings.warn(f"Optimization failed for config {config_name} restart {restart}: {e}")
                continue
    
    # If we found a better solution, try final refinement with L-BFGS-B
    if best_points is not None:
        try:
            x0 = best_points.flatten()
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(refined_points)
                if ratio > best_ratio:
                    return refined_points
                    
        except Exception as e:
            warnings.warn(f"Final refinement failed: {e}")
        
        return best_points
    
    # Fallback to first configuration if nothing worked
    return initial_configs[0][1]


# EVOLVE-BLOCK-END
