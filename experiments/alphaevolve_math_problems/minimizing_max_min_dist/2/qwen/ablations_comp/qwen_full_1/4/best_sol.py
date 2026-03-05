# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical constructions and robust optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return 0.0
            
        return d_min / d_max
    
    def objective_function(x):
        """Objective function to maximize (negative because we minimize)."""
        # Reshape flat array back to 2D points
        points = x.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        # Return negative because we want to maximize
        return -ratio
    
    # Generate multiple high-quality initial configurations
    def generate_multiple_initial_configs():
        """Generate several high-quality initial configurations."""
        configs = []
        
        # Configuration 1: Regular 16-gon (mathematical construction)
        n = 16
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        points = points * 0.4 + 0.5  # Scale and center
        configs.append(points.copy())
        
        # Configuration 2: Hexagonal grid pattern (inspired by hexagonal packing)
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25 * np.sqrt(3) / 2
                points.append([x, y])
        points = np.array(points[:16])
        points = np.clip(points, 0, 1)
        configs.append(points.copy())
        
        # Configuration 3: Concentric rings approach (balanced distribution)
        points = []
        # Outer ring with 12 points
        for i in range(12):
            angle = 2 * math.pi * i / 12
            points.append([0.5 + 0.4 * math.cos(angle), 0.5 + 0.4 * math.sin(angle)])
        # Inner ring with 4 points
        for i in range(4):
            angle = 2 * math.pi * i / 4
            points.append([0.5 + 0.15 * math.cos(angle), 0.5 + 0.15 * math.sin(angle)])
        points = np.array(points[:16])
        configs.append(points.copy())
        
        # Configuration 4: Fibonacci spiral projection (good for dispersion)
        # Generate points using Fibonacci spiral on sphere, then project to 2D
        golden_angle = math.pi * (3 - math.sqrt(5))
        points_3d = []
        for i in range(16):
            y = 1 - (i / float(15)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)   # radius at y
            theta = golden_angle * i       # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points_3d.append([x, y, z])
        
        # Project from 3D to 2D using stereographic projection from south pole
        points_2d = []
        for point in points_3d:
            x, y, z = point
            # Stereographic projection from south pole (0,0,-1)
            if abs(z + 1) < 1e-10:  # Handle the special case
                proj_x, proj_y = 0, 0
            else:
                proj_x = x / (1 + z)
                proj_y = y / (1 + z)
            points_2d.append([proj_x, proj_y])
        
        points_2d = np.array(points_2d)
        # Normalize to fit in [0,1] x [0,1]
        if len(points_2d) > 0:
            mean_x, mean_y = np.mean(points_2d, axis=0)
            centered = points_2d - [mean_x, mean_y]
            max_dist = np.max(np.abs(centered))
            if max_dist > 0:
                scaled = centered / max_dist * 0.8 + 0.5
            else:
                scaled = centered + 0.5
            configs.append(scaled.copy())
        
        # Configuration 5: Random with fixed seed for reproducibility
        np.random.seed(42)
        points = np.random.rand(16, 2)
        configs.append(points.copy())
        
        return configs
    
    # Robust optimization with multiple restarts and methods
    def robust_optimization(initial_points, max_restarts=3):
        """Perform robust optimization with multiple restarts and methods."""
        bounds = [(0, 1) for _ in range(32)]
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        # Try multiple optimization restarts with different seeds
        for restart in range(max_restarts):
            try:
                # Add small random perturbations to break symmetry
                np.random.seed(42 + restart * 10)
                perturbed = initial_points.copy()
                noise = np.random.normal(0, 0.01, initial_points.shape)
                perturbed = perturbed + noise
                perturbed = np.clip(perturbed, 0, 1)
                
                x0 = perturbed.flatten()
                
                # Use multiple optimization methods for robustness
                methods = ['L-BFGS-B', 'TNC', 'SLSQP']
                for method in methods:
                    try:
                        result = minimize(
                            objective_function,
                            x0,
                            method=method,
                            bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                        )
                        
                        if result.success:
                            final_points = result.x.reshape(-1, 2)
                            final_points = np.clip(final_points, 0, 1)
                            ratio = compute_min_max_ratio(final_points)
                            
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points.copy()
                                
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        return best_points
    
    # Main execution logic - try multiple strategies
    best_ratio = 0.0
    best_points = None
    
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initial_configs()
    
    # Try each configuration with robust optimization
    for i, config in enumerate(initial_configs):
        try:
            # First optimize with local methods
            optimized = robust_optimization(config, max_restarts=3)
            ratio = compute_min_max_ratio(optimized)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized.copy()
                
        except Exception:
            continue
    
    # If no optimization succeeded, return a good default configuration
    if best_points is None:
        # Fall back to hexagonal grid (generally one of the best starting points)
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25 * np.sqrt(3) / 2
                points.append([x, y])
        best_points = np.array(points[:16])
        best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
