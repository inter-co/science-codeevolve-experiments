# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.distance import pdist
import time
import warnings
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a sophisticated hybrid approach combining geometric initialization with advanced 
    optimization techniques to achieve state-of-the-art results.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
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
    
    # Generate high-quality initial configurations based on mathematical principles
    def generate_initial_configs():
        configs = []
        
        # Configuration 1: Optimized hexagonal lattice (proven to be excellent for dispersion)
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
        
        # Precise normalization to unit square with good aspect ratio preservation
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0 and y_range > 0:
                # Scale to fit nicely in [0,1]x[0,1] while preserving hexagonal structure
                scale = 0.85 / max(x_range, y_range) if max(x_range, y_range) > 0 else 1.0
                
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) * scale
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) * scale
                
                # Center properly in [0,1]x[0,1]
                points[:, 0] += (0.075 - np.min(points[:, 0]))
                points[:, 1] += (0.075 - np.min(points[:, 1]))
        
        configs.append(points.copy())
        
        # Configuration 2: Circle with golden ratio distribution
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.38
        points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        points += np.random.normal(0, 0.015, points.shape)
        points[:, 0] += 0.5
        points[:, 1] += 0.5
        points = np.clip(points, 0, 1)
        configs.append(points)
        
        # Configuration 3: Spiral pattern with golden ratio
        np.random.seed(999)
        spiral_points = []
        angle_step = 2 * np.pi * 0.618
        radius_step = 0.38 / 16
        for i in range(16):
            angle = i * angle_step
            radius = i * radius_step + 0.03
            x = 0.5 + radius * np.cos(angle) * 0.38
            y = 0.5 + radius * np.sin(angle) * 0.38
            spiral_points.append([x, y])
        configs.append(np.array(spiral_points))
        
        # Configuration 4: Perturbed regular grid
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + 0.125
                y = j * 0.25 + 0.125
                grid_points.append([x, y])
        configs.append(np.array(grid_points))
        
        # Configuration 5: Offset grid pattern
        np.random.seed(123)
        points = np.zeros((16, 2))
        idx = 0
        for i in range(4):
            for j in range(4):
                x = (i + 0.5 * ((j + i) % 2)) / 3.2 + 0.08
                y = (j + 0.5 * (i % 2)) / 3.2 + 0.08
                points[idx] = [x, y]
                idx += 1
        configs.append(points)
        
        return configs
    
    # Create bounds for optimization (points must stay in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    # Generate initial configurations
    initial_configs = generate_initial_configs()
    
    best_ratio = 0
    best_points = None
    start_time = time.time()
    
    # Multi-start optimization with strategic approach - inspired by best practices from both inspirations
    num_restarts = 10  # Increased number of restarts for better exploration
    
    # Use multiple configurations with multiple restarts each
    for restart in range(num_restarts):
        # Early exit if we're running out of time
        if time.time() - start_time > 55:
            break
            
        # Select configuration based on restart index to diversify
        config_idx = restart % len(initial_configs)
        initial_config = initial_configs[config_idx]
        
        # Add diverse perturbation for this restart
        np.random.seed(restart * 100 + 42)
        perturbed_config = initial_config + np.random.normal(0, 0.015, initial_config.shape)
        perturbed_config = np.clip(perturbed_config, 0, 1)
        
        # Flatten for optimization
        x0 = perturbed_config.flatten()
        
        # Optimization with dual annealing for global search - using parameters from inspiration 1
        try:
            result = dual_annealing(
                objective, 
                bounds, 
                maxiter=1000,  # More iterations for better convergence (from inspiration 1)
                initial_temp=1000,  # Higher initial temperature (from inspiration 1)
                seed=42 + restart,
                no_local_search=True
            )
            
            if result.success:
                test_points = result.x.reshape(-1, 2)
                distances = pdist(test_points)
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = test_points.copy()
                        
        except Exception as e:
            warnings.warn(f"Optimization failed for restart {restart}: {e}")
            continue
    
    # If we found a better solution, do final refinement with multiple passes
    if best_points is not None and time.time() - start_time < 58:
        try:
            # First refinement with L-BFGS-B for high precision (from inspiration 2)
            x0 = best_points.flatten()
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}  # Tighter tolerances (from inspiration 2)
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                distances = pdist(refined_points)
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        # Additional refinement pass with SLSQP for potential further improvement
                        if time.time() - start_time < 57:
                            np.random.seed(1000)
                            perturbed = refined_points + np.random.normal(0, 0.005, refined_points.shape)
                            perturbed = np.clip(perturbed, 0, 1)
                            
                            x0 = perturbed.flatten()
                            result2 = minimize(
                                objective,
                                x0,
                                method='SLSQP',
                                bounds=bounds,
                                options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-13}
                            )
                            
                            if result2.success:
                                temp_points = result2.x.reshape(-1, 2)
                                distances2 = pdist(temp_points)
                                d_min2 = np.min(distances2)
                                d_max2 = np.max(distances2)
                                
                                if d_max2 > 0:
                                    ratio2 = d_min2 / d_max2
                                    if ratio2 > ratio:
                                        return temp_points
                        
                        return refined_points
                        
        except Exception as e:
            warnings.warn(f"Final refinement failed: {e}")
        
        return best_points
    
    # Fallback to the first configuration if nothing worked
    return initial_configs[0]


# EVOLVE-BLOCK-END
