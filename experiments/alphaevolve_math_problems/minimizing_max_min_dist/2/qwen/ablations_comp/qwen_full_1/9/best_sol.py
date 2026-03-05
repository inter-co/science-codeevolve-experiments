# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import math
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust, multi-stage optimization approach with enhanced initialization strategies
    and better optimization methodology.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    def objective(x_flat):
        """Minimize negative of min/max ratio (equivalent to maximizing the ratio)"""
        points = x_flat.reshape(-1, 2)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return np.inf
        return -min_dist / max_dist
    
    # Generate multiple high-quality initial configurations inspired by discrete geometry
    def generate_initial_configurations():
        configs = []
        
        # Configuration 1: Golden spiral approach (inspired by mathematical principles)
        golden_points = []
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        for i in range(16):
            angle = 2 * math.pi * i * phi
            radius = 0.4 * math.sqrt(i / 15.0) + 0.05
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            # Add small random perturbation
            x += 0.01 * np.random.randn()
            y += 0.01 * np.random.randn()
            golden_points.append([x, y])
        configs.append(np.array(golden_points))
        
        # Configuration 2: Hexagonal grid with perturbations (inspired by optimal packings)
        hex_points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                # Add structured perturbations for better distribution
                x += np.random.normal(0, 0.015) * (1 + 0.1 * np.sin(i * 0.5))
                y += np.random.normal(0, 0.015) * (1 + 0.1 * np.cos(j * 0.5))
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                hex_points.append([x, y])
        configs.append(np.array(hex_points))
        
        # Configuration 3: Modified 4x4 grid with offset rows (inspired by hexagonal patterns)
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                # Add offset for hexagonal-like structure
                if j % 2 == 1:
                    x += 0.125
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                grid_points.append([x, y])
        configs.append(np.array(grid_points))
        
        # Configuration 4: Circular arrangement with perturbations (inspired by spherical codes)
        circular_points = []
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        base_radius = 0.4
        for i, angle in enumerate(angles):
            x = 0.5 + base_radius * np.cos(angle)
            y = 0.5 + base_radius * np.sin(angle)
            # Add controlled perturbations
            angle_offset = angle + 0.1 * np.sin(i * 0.7)
            radius_offset = base_radius + 0.05 * np.cos(i * 0.5)
            x = 0.5 + radius_offset * np.cos(angle_offset)
            y = 0.5 + radius_offset * np.sin(angle_offset)
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            circular_points.append([x, y])
        configs.append(np.array(circular_points))
        
        # Configuration 5: Regular grid with more sophisticated perturbations (inspired by spherical codes)
        regular_grid = []
        for i in range(4):
            for j in range(4):
                x = i / 3.0 if i < 3 else 1.0
                y = j / 3.0 if j < 3 else 1.0
                # Add structured perturbations to avoid degeneracy
                pert_x = 0.05 * np.sin(i * 0.8 + j * 0.3)
                pert_y = 0.05 * np.cos(i * 0.3 + j * 0.8)
                x = max(0.05, min(0.95, x + pert_x))
                y = max(0.05, min(0.95, y + pert_y))
                regular_grid.append([x, y])
        configs.append(np.array(regular_grid))
        
        # Configuration 6: Random but constrained initialization (robust fallback)
        random_points = np.random.rand(16, 2)
        configs.append(random_points)
        
        return configs
    
    # Get initial configurations
    initial_configs = generate_initial_configurations()
    
    best_ratio = -np.inf
    best_points = None
    
    # Multi-start optimization with enhanced diversity and better parameter tuning
    num_restarts = 20  # More restarts for better exploration
    
    for restart in range(num_restarts):
        # Select different initial configuration for each restart
        config_idx = restart % len(initial_configs)
        initial_config = initial_configs[config_idx].copy()
        
        # Add more diverse perturbations for each restart
        np.random.seed(restart * 100 + 42)
        perturbed_config = initial_config.copy()
        
        # Different perturbation strategies based on restart number
        if restart < 5:
            # Stronger perturbations for early restarts
            perturbed_config += np.random.normal(0, 0.02, size=perturbed_config.shape)
        elif restart < 10:
            # Medium perturbations
            perturbed_config += np.random.normal(0, 0.01, size=perturbed_config.shape)
        else:
            # Light perturbations
            perturbed_config += np.random.normal(0, 0.005, size=perturbed_config.shape)
            
        perturbed_config = np.clip(perturbed_config, 0, 1)
        
        # Try multiple optimization methods with better parameter tuning
        methods_and_params = [
            ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}),
            ('TNC', {'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}),
            ('SLSQP', {'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
        ]
        
        for method, params in methods_and_params:
            try:
                x0 = perturbed_config.flatten()
                
                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(32)]
                
                # Optimize with selected method
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    options=params
                )
                
                if result.success:
                    optimized_points = result.x.reshape((16, 2))
                    optimized_points = np.clip(optimized_points, 0, 1)
                    
                    # Calculate actual ratio
                    distances = pdist(optimized_points)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    
                    if max_dist > 1e-12:  # Use stricter threshold
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
            except Exception:
                continue
    
    # If no optimization worked, use the best initial configuration
    if best_points is None:
        # Return the first initial configuration as fallback
        return initial_configs[0]
    
    # Final refinement with differential evolution for global search
    try:
        bounds = [(0, 1) for _ in range(32)]
        
        # Use differential evolution for final global refinement with better parameters
        result_de = differential_evolution(
            objective,
            bounds,
            maxiter=100,  # More iterations for final refinement
            popsize=30,   # Larger population size for better exploration
            mutation=(0.8, 1),  # Higher mutation rate for more exploration
            recombination=0.9,   # Higher recombination for better mixing
            seed=42,
            disp=False,
            atol=1e-12,
            rtol=1e-12
        )
        
        if result_de.success:
            de_points = result_de.x.reshape((16, 2))
            de_points = np.clip(de_points, 0, 1)
            
            # Calculate actual ratio
            distances = pdist(de_points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist > 1e-12:
                de_ratio = min_dist / max_dist
                if de_ratio > best_ratio:
                    best_ratio = de_ratio
                    best_points = de_points.copy()
                    
    except Exception:
        pass
    
    # Final check to ensure we have a valid solution
    if best_points is None:
        return initial_configs[0]
    
    return best_points


# EVOLVE-BLOCK-END
