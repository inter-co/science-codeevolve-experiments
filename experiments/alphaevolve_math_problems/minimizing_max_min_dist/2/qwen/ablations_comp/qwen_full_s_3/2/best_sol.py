# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.distance import pdist
import time
import warnings
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
    
    # Generate high-quality initial configurations with more diversity
    def generate_initial_configs():
        configs = []
        
        # Configuration 1: Hexagonal lattice (inspired by best known arrangements)
        sqrt3 = np.sqrt(3)
        points_list = []
        for i in range(4):
            for j in range(4):
                if len(points_list) >= 16:
                    break
                offset = 0.5 if i % 2 == 1 else 0.0
                x = offset + j * 0.25
                y = i * sqrt3 * 0.25
                points_list.append([x, y])
        
        points = np.array(points_list[:16])
        # Normalize to [0,1] range
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            if x_range > 0 and y_range > 0:
                scale_x = 0.9 / x_range if x_range > 0 else 1.0
                scale_y = 0.9 / y_range if y_range > 0 else 1.0
                scale = min(scale_x, scale_y)
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) * scale + 0.05
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) * scale + 0.05
        configs.append(points.copy())
        
        # Configuration 2: Circle arrangement with good spread
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4
        points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        points += np.random.normal(0, 0.03, points.shape)
        points[:, 0] += 0.5
        points[:, 1] += 0.5
        points = np.clip(points, 0, 1)
        configs.append(points)
        
        # Configuration 3: Perturbed grid (more varied)
        np.random.seed(999)
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + 0.125 + np.random.normal(0, 0.015)
                y = j * 0.25 + 0.125 + np.random.normal(0, 0.015)
                grid_points.append([x, y])
        points = np.array(grid_points)
        points = np.clip(points, 0, 1)
        configs.append(points)
        
        # Configuration 4: Spiral pattern with golden ratio
        np.random.seed(123)
        spiral_points = []
        angle_step = 2 * np.pi * 0.618
        radius_step = 0.35 / 16
        for i in range(16):
            angle = i * angle_step
            radius = i * radius_step + 0.05
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            spiral_points.append([x, y])
        configs.append(np.array(spiral_points))
        
        # Configuration 5: Random with better distribution
        np.random.seed(456)
        random_points = np.random.rand(16, 2) * 0.8 + 0.1
        configs.append(random_points)
        
        return configs
    
    # Create bounds for optimization (points must stay in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    # Generate initial configurations
    initial_configs = generate_initial_configs()
    
    best_ratio = 0
    best_points = None
    start_time = time.time()
    
    # Multi-start optimization with dual annealing for global search
    # Use more aggressive dual annealing parameters and better restart strategy
    for i, initial_config in enumerate(initial_configs):
        # Check time budget
        if time.time() - start_time > 55:
            break
            
        # Try multiple restarts for each configuration
        for restart in range(10):  # Increased restarts for better exploration
            # Check time budget before each restart
            if time.time() - start_time > 55:
                break
                
            # Add perturbation for diversity
            np.random.seed(i * 1000 + restart * 100)
            perturbed_config = initial_config + np.random.normal(0, 0.02, initial_config.shape)
            perturbed_config = np.clip(perturbed_config, 0, 1)
            
            # Flatten for optimization
            x0 = perturbed_config.flatten()
            
            # Use dual annealing with highly aggressive parameters
            try:
                result = dual_annealing(
                    objective, 
                    bounds, 
                    maxiter=2500,  # Even more iterations for better convergence
                    initial_temp=2000,  # Higher initial temperature
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
    
    # If we found a good solution, do final refinement with L-BFGS-B
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
                    options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-13}  # Even tighter tolerances
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
