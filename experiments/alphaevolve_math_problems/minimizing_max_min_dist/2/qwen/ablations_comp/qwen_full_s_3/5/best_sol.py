# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing, minimize
import math
import warnings
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining geometric initialization with advanced optimization.

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
        
        # Avoid division by zero - return large penalty value for invalid configurations
        if d_max == 0:
            return np.inf  # Large penalty for invalid configurations
            
        ratio = d_min / d_max
        return -ratio  # Negative because we want to maximize ratio
    
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
    
    def generate_mathematical_configs():
        """Generate highly optimized initial configurations based on mathematical principles"""
        configs = []
        
        # Configuration 1: Optimized hexagonal lattice (proven to be excellent for dispersion)
        # Using a 4x4 hexagonal grid with precise scaling (from inspiration 1)
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
        
        # Configuration 2: Regular grid with slight perturbations (from inspiration 2)
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + 0.125
                y = j * 0.25 + 0.125
                grid_points.append([x, y])
        configs.append(np.array(grid_points))
        
        # Configuration 3: Circle with golden ratio distribution (from inspiration 1)
        # This is based on mathematical insights about optimal point distributions
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        # Use slightly different radii for better spread
        radius = 0.38
        points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        points += np.random.normal(0, 0.015, points.shape)  # Even smaller perturbation
        points[:, 0] += 0.5
        points[:, 1] += 0.5
        points = np.clip(points, 0, 1)
        configs.append(points)
        
        # Configuration 4: Mathematical construction based on known optimal 16-point configurations
        # Create a configuration that has good symmetry and minimal clustering
        np.random.seed(123)
        # Start with a 4x4 grid but offset to create better spacing
        points = np.zeros((16, 2))
        idx = 0
        for i in range(4):
            for j in range(4):
                # Create offset pattern to avoid regular grid clustering
                x = (i + 0.5 * ((j + i) % 2)) / 3.2 + 0.08
                y = (j + 0.5 * (i % 2)) / 3.2 + 0.08
                points[idx] = [x, y]
                idx += 1
        configs.append(points)
        
        # Configuration 5: Optimized spiral pattern - more refined version (from inspiration 1)
        np.random.seed(999)
        spiral_points = []
        angle_step = 2 * np.pi * 0.618  # Golden ratio
        radius_step = 0.38 / 16
        for i in range(16):
            angle = i * angle_step
            radius = i * radius_step + 0.03
            x = 0.5 + radius * np.cos(angle) * 0.38
            y = 0.5 + radius * np.sin(angle) * 0.38
            spiral_points.append([x, y])
        configs.append(np.array(spiral_points))
        
        return configs
    
    # Create bounds for optimization (points must stay in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    # Generate multiple initial configurations
    initial_configs = generate_mathematical_configs()
    
    best_ratio = -np.inf
    best_points = None
    start_time = time.time()
    
    # Multi-start optimization with strategic approach
    # Use fewer but higher quality restarts to respect time constraint
    for i, initial_config in enumerate(initial_configs):
        # Early exit if we're running out of time
        if time.time() - start_time > 55:
            break
            
        # Try only one restart per configuration for efficiency (from inspiration 2)
        # Add slight perturbation for diversity
        np.random.seed(i * 100 + 42)  # Better seed mixing
        perturbed_config = initial_config + np.random.normal(0, 0.012, initial_config.shape)
        perturbed_config = np.clip(perturbed_config, 0, 1)
        
        # Flatten for optimization
        x0 = perturbed_config.flatten()
        
        # Optimization with dual annealing for global search
        try:
            # Use more aggressive parameters for better exploration within time limits (from inspiration 1)
            # But also reduce maxiter to ensure we don't exceed time limit
            result = dual_annealing(
                objective, 
                bounds, 
                maxiter=500,  # Increase iterations slightly
                initial_temp=600,  # Even higher initial temperature for more exploration
                seed=42 + i,
                no_local_search=True
            )
            
            if result.success:
                test_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(test_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = test_points.copy()
                    
        except Exception as e:
            warnings.warn(f"Optimization failed for config {i}: {e}")
            continue
    
    # If we found a better solution, do final refinement with multiple passes
    if best_points is not None and time.time() - start_time < 58:
        try:
            # First refinement with L-BFGS-B for high precision
            x0 = best_points.flatten()
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(refined_points)
                if ratio > best_ratio:
                    # Second refinement pass with SLSQP for potential further improvement
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
                            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
                        )
                        
                        if result2.success:
                            temp_points = result2.x.reshape(-1, 2)
                            ratio2 = compute_min_max_ratio(temp_points)
                            if ratio2 > ratio:
                                return temp_points
                    
                    return refined_points
                    
        except Exception as e:
            warnings.warn(f"Final refinement failed: {e}")
        
        return best_points
    
    # Fallback to the first configuration if nothing worked
    return initial_configs[0]


# EVOLVE-BLOCK-END
