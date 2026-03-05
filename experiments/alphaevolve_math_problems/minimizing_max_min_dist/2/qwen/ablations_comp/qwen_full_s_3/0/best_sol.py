# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust optimization approach with multiple restarts and adaptive strategies.

    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Create a high-quality initial configuration using a hexagonal pattern
    def create_hexagonal_config():
        # Create a hexagonal lattice pattern that works well for point dispersion
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
        return np.array(points)
    
    # Alternative: Create a more sophisticated initial configuration
    def create_spiral_config():
        # Create a spiral pattern to get good initial spread
        points = []
        angles = np.linspace(0, 4*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.4, 16)
        for angle, radius in zip(angles, radii):
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    # Objective function to maximize min/max distance ratio
    def objective(params):
        # Reshape parameters back to points
        points = params.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio since we want to maximize the ratio
        # Small epsilon to avoid division by zero
        if d_max <= 1e-12:
            return -1e10  # Invalid case
        return -d_min / d_max
    
    # Optimization settings
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    best_ratio = -np.inf
    best_points = None
    
    # Try multiple initialization strategies
    initial_configs = []
    
    # Configuration 1: Hexagonal grid (inspired by inspiration programs)
    config1 = create_hexagonal_config()
    initial_configs.append(config1.flatten())
    
    # Configuration 2: Spiral pattern
    config2 = create_spiral_config()
    initial_configs.append(config2.flatten())
    
    # Configuration 3: Simple grid with noise
    config3 = np.random.rand(16, 2) * 0.8 + 0.1  # Random points in [0.1, 0.9] range
    initial_configs.append(config3.flatten())
    
    # Try multiple optimization runs with different initializations and methods
    for restart_idx, initial_params in enumerate(initial_configs):
        try:
            # Multiple optimization attempts with different methods
            methods = ['L-BFGS-B', 'SLSQP']
            
            for method in methods:
                # Add small random perturbation to break symmetry
                np.random.seed(restart_idx * 100 + hash(method) % 1000)
                perturbed_params = initial_params + np.random.normal(0, 0.01, len(initial_params))
                perturbed_params = np.clip(perturbed_params, 0, 1)
                
                # Optimize with bounds
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        objective,
                        perturbed_params,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                    )
                
                if result.success:
                    # Calculate actual ratio for this result
                    points_test = result.x.reshape(-1, 2)
                    distances = pdist(points_test)
                    if len(distances) > 0:
                        d_min = np.min(distances)
                        d_max = np.max(distances)
                        if d_max > 1e-12:  # Avoid division by near-zero
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = points_test.copy()
                                
        except Exception as e:
            continue
    
    # If no optimization worked, return a simple good configuration
    if best_points is None:
        # Return the hexagonal configuration as a fallback
        best_points = create_hexagonal_config()
    
    # Final refinement with one more optimization run
    if best_points is not None:
        try:
            # Add more precise optimization
            np.random.seed(42)
            final_params = best_points.flatten() + np.random.normal(0, 0.005, 32)
            final_params = np.clip(final_params, 0, 1)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    final_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-9}
                )
            
            if result.success:
                points_test = result.x.reshape(-1, 2)
                distances = pdist(points_test)
                if len(distances) > 0:
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    if d_max > 1e-12:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_points = points_test
        except Exception:
            pass
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
