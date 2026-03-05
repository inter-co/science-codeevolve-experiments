# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing, minimize
import math
import warnings
import time
warnings.filterwarnings('ignore')

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
    
    # Generate multiple initial configurations inspired by best practices
    def generate_multiple_initial_configs():
        configs = []
        
        # Configuration 1: Hexagonal lattice (improved version)
        sqrt3 = np.sqrt(3)
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
        
        configs.append(points.copy())
        
        # Configuration 2: Perturbed regular grid
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + 0.125
                y = j * 0.25 + 0.125
                grid_points.append([x, y])
        configs.append(np.array(grid_points))
        
        # Configuration 3: Circle-based with perturbation
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4
        points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        points += np.random.normal(0, 0.03, points.shape)
        points[:, 0] += 0.5
        points[:, 1] += 0.5
        points = np.clip(points, 0, 1)
        configs.append(points)
        
        # Configuration 4: Optimized spiral pattern
        np.random.seed(123)
        spiral_points = []
        angle_step = 2 * np.pi * 0.618  # Golden ratio
        radius_step = 0.4 / 16
        for i in range(16):
            angle = i * angle_step
            radius = i * radius_step + 0.05
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            spiral_points.append([x, y])
        configs.append(np.array(spiral_points))
        
        # Configuration 5: Another hexagonal variant with different spacing
        hex_points = []
        for i in range(4):
            for j in range(4):
                if len(hex_points) >= 16:
                    break
                # Different spacing for variety
                x = j * 0.25 + 0.125 + (i % 2) * 0.125
                y = i * 0.25 + 0.125
                hex_points.append([x, y])
        hex_config = np.array(hex_points[:16])
        hex_config = np.clip(hex_config, 0, 1)
        configs.append(hex_config)
        
        # Configuration 6: Slightly perturbed regular grid for variety
        np.random.seed(999)
        perturbed_grid = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + 0.125 + np.random.normal(0, 0.02)
                y = j * 0.25 + 0.125 + np.random.normal(0, 0.02)
                perturbed_grid.append([x, y])
        perturbed_points = np.array(perturbed_grid)
        perturbed_points = np.clip(perturbed_points, 0, 1)
        configs.append(perturbed_points)
        
        return configs
    
    # Create bounds for optimization (points must stay in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initial_configs()
    
    best_ratio = 0
    best_points = None
    start_time = time.time()
    
    # Multi-start optimization with dual annealing for global search followed by local refinement
    for i, initial_config in enumerate(initial_configs):
        # Check time budget early
        if time.time() - start_time > 55:
            break
            
        # Try multiple restarts for each configuration to get better results
        for restart in range(5):  # Increase restarts to 5 for better exploration
            # Check time budget before each restart
            if time.time() - start_time > 55:
                break
                
            # Add slight perturbation for diversity
            np.random.seed(i * 1000 + restart * 100)  # Better seeding for diversity
            perturbed_config = initial_config + np.random.normal(0, 0.01, initial_config.shape)
            perturbed_config = np.clip(perturbed_config, 0, 1)
            
            # Flatten for optimization
            x0 = perturbed_config.flatten()
            
            # First, use dual annealing for global search (more robust than L-BFGS alone)
            try:
                # Use higher iteration count and better parameters like inspiration 2
                result = dual_annealing(
                    objective, 
                    bounds, 
                    maxiter=1500,  # More iterations for better global search (from inspiration 2)
                    initial_temp=2000,  # Even higher initial temperature (from inspiration 2)
                    seed=42 + i * 10 + restart,
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
                continue
    
    # If we found a better solution, try final refinement with L-BFGS-B
    if best_points is not None:
        try:
            # Check time budget before refinement
            if time.time() - start_time < 55:
                x0 = best_points.flatten()
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14}  # Even tighter tolerances (from inspiration 2)
                )
                
                if result.success:
                    refined_points = result.x.reshape(-1, 2)
                    distances = pdist(refined_points)
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    
                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            return refined_points
                            
        except Exception as e:
            pass  # Continue with best_points if refinement fails
        
        return best_points
    
    # Fallback to the first configuration if nothing worked
    return initial_configs[0]


# EVOLVE-BLOCK-END
