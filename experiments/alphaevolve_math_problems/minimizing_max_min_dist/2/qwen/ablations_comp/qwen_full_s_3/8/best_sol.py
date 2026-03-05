# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing, minimize
import warnings
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining geometric initialization with advanced 
    global optimization techniques and systematic restart strategies.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(params):
        """Objective function to minimize (negative of ratio to maximize ratio)"""
        # Reshape parameters back to points
        points = params.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero or near-zero
        if d_max <= 1e-12:
            return np.inf
            
        ratio = d_min / d_max
        return -ratio  # Negative because we want to maximize ratio
    
    def create_hexagonal_config():
        """Create a hexagonal lattice pattern that works well for point dispersion"""
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
        return np.array(points[:16])  # Ensure exactly 16 points
    
    def create_spiral_config():
        """Create a spiral pattern to get good initial spread"""
        points = []
        angles = np.linspace(0, 4*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.4, 16)
        for angle, radius in zip(angles, radii):
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def create_fibonacci_spiral():
        """Create points using Fibonacci spiral pattern"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(16):
            angle = i * 2.4  # Modified angle for better spread
            radius = np.sqrt(i / 15.0) * 0.4  # Radius increases slowly
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def create_regular_polygon():
        """Create points arranged in a regular polygon configuration"""
        points = []
        # Arrange 16 points in a circle with some variation
        for i in range(16):
            angle = 2 * np.pi * i / 16
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def create_concentric_rings():
        """Create points in concentric ring patterns"""
        points = []
        # Outer ring: 8 points
        for k in range(8):
            angle = k * np.pi/4
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        
        # Inner ring: 8 points
        for k in range(8):
            angle = k * np.pi/4
            x = 0.5 + 0.2 * np.cos(angle)
            y = 0.5 + 0.2 * np.sin(angle)
            points.append([x, y])
            
        # Ensure exactly 16 points
        return np.array(points[:16])
    
    def create_random_config():
        """Create a completely random configuration"""
        np.random.seed(42)
        points = np.random.rand(16, 2)
        return points
    
    def create_grid_with_noise():
        """Create a grid pattern with added noise"""
        points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + 0.125 + np.random.normal(0, 0.02)
                y = j * 0.25 + 0.125 + np.random.normal(0, 0.02)
                points.append([x, y])
        return np.array(points)
    
    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    # Create multiple diverse initial configurations - inspired by best practices
    initial_configs = [
        create_hexagonal_config(),           # Hexagonal packing
        create_spiral_config(),              # Spiral pattern
        create_fibonacci_spiral(),           # Golden spiral
        create_regular_polygon(),            # Regular polygon
        create_concentric_rings(),           # Concentric rings
        create_random_config(),              # Random configuration
        create_grid_with_noise()             # Grid with noise
    ]
    
    # Track best solution found
    best_ratio = -np.inf
    best_points = None
    
    # Time tracking to ensure we don't exceed time limits
    start_time = time.time()
    max_time = 55  # Leave 5 seconds for final processing
    
    # Try multiple restarts with different strategies
    # Use more aggressive restart strategy with better optimization parameters
    total_restarts = 20
    
    for restart_idx in range(total_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Select initial configuration with cycling
        config_idx = restart_idx % len(initial_configs)
        initial_points = initial_configs[config_idx].copy()
        
        # Add controlled random perturbation to break symmetry
        np.random.seed(restart_idx * 1000 + 42)
        perturbation_scale = 0.015 if restart_idx < 10 else 0.005  # Different scales
        perturbed_points = initial_points + np.random.normal(0, perturbation_scale, initial_points.shape)
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        # Flatten for optimization
        initial_params = perturbed_points.flatten()
        
        # Try dual annealing first (better for global optimization)
        try:
            # Use dual_annealing with more aggressive settings for better exploration
            result = dual_annealing(
                objective,
                bounds,
                maxiter=800,  # More iterations for better exploration
                initial_temp=2000,  # Higher initial temperature
                seed=42 + restart_idx,
                no_local_search=True
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
            # If dual_annealing fails, try L-BFGS-B
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        objective,
                        initial_params,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
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
                                    
            except Exception:
                continue
    
    # If no optimization worked, return the best initial configuration
    if best_points is None:
        # Find the best among initial configurations
        best_initial_ratio = -np.inf
        for config in initial_configs:
            distances = pdist(config)
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 1e-12:
                    ratio = d_min / d_max
                    if ratio > best_initial_ratio:
                        best_initial_ratio = ratio
                        best_points = config.copy()
    
    # Final refinement with a quick optimization run
    if best_points is not None and time.time() - start_time < max_time - 2:
        try:
            # Add very small random perturbation for final refinement
            np.random.seed(999)
            final_params = best_points.flatten() + np.random.normal(0, 0.001, 32)
            final_params = np.clip(final_params, 0, 1)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    final_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
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
    if best_points is not None:
        best_points = np.clip(best_points, 0, 1)
    else:
        # Last resort fallback
        best_points = create_hexagonal_config()
    
    return best_points


# EVOLVE-BLOCK-END
