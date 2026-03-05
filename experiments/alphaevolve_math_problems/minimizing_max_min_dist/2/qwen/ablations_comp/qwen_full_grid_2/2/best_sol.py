# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust hybrid approach combining geometric initialization and multi-start optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        dmin = np.min(distances)
        dmax = np.max(distances)
        if dmax == 0:
            return 0.0
        return dmin / dmax
    
    def objective_function(x_flat):
        """Objective function to minimize (negative of ratio to maximize it)"""
        points = x_flat.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        return -ratio
    
    def generate_diverse_initializations():
        """Generate multiple diverse initial configurations."""
        configs = []
        
        # Configuration 1: Regular 4x4 grid with hexagonal offset (inspired by inspiration 1)
        grid_points = []
        for i in range(4):
            for j in range(4):
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + offset) / 3.0
                y = i / 3.0
                grid_points.append([x, y])
        configs.append(np.array(grid_points[:16]))
        
        # Configuration 2: Perturbed version of hexagonal grid
        perturbed_grid = configs[0] + np.random.uniform(-0.03, 0.03, (16, 2))
        configs.append(np.clip(perturbed_grid, 0, 1))
        
        # Configuration 3: Alternative hexagonal pattern (more spread out)
        hex_points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                hex_points.append([x, y])
        configs.append(np.array(hex_points[:16]))
        
        # Configuration 4: Random with bounds checking (inspired by inspiration 2)
        random_points = np.random.uniform(0.05, 0.95, (16, 2))
        configs.append(random_points)
        
        # Configuration 5: Centered pattern with radial distribution
        center_points = np.array([[0.5, 0.5]] * 16)
        radial_points = []
        for i in range(16):
            angle = i * 2 * np.pi / 16
            radius = 0.3 + 0.2 * np.sin(i * 0.5)
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            radial_points.append([x, y])
        configs.append(np.array(radial_points))
        
        return configs
    
    # Generate multiple diverse initial configurations
    initial_configs = generate_diverse_initializations()
    
    best_ratio = 0.0
    best_points = None
    
    # Try each initial configuration with multiple optimization strategies
    for i, initial_config in enumerate(initial_configs):
        # Try multiple optimization strategies with different settings
        strategies = [
            {'method': 'L-BFGS-B', 'options': {'maxiter': 800, 'ftol': 1e-11, 'gtol': 1e-11}},
            {'method': 'L-BFGS-B', 'options': {'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-12}},
            {'method': 'SLSQP', 'options': {'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-11}}
        ]
        
        for strategy in strategies:
            try:
                # Different random seed for each optimization attempt
                np.random.seed(i * 100 + hash(strategy['method']) % 1000)
                
                # Add small random perturbation to break symmetry
                perturbed_config = initial_config + np.random.uniform(-0.02, 0.02, (16, 2))
                perturbed_config = np.clip(perturbed_config, 0, 1)
                
                result = minimize(
                    objective_function,
                    perturbed_config.flatten(),
                    method=strategy['method'],
                    bounds=[(0, 1) for _ in range(32)],
                    options=strategy['options']
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
    
    # Final refinement with highest precision if we found a good solution
    if best_points is not None:
        try:
            # Run one final high-precision optimization
            result = minimize(
                objective_function,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-13}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                ratio = compute_min_max_ratio(refined_points)
                
                if ratio > best_ratio:
                    best_points = refined_points
                    
        except Exception:
            pass
    
    # If no good solution was found, return the best from initial set
    if best_points is None:
        # Return the first configuration as fallback
        best_points = initial_configs[0]
    
    return best_points


# EVOLVE-BLOCK-END
