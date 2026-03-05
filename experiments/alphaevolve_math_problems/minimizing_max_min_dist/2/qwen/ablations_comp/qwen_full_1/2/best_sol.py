# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining global optimization with local refinement and multiple geometric initializations.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
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
    
    def generate_multiple_initial_configs():
        """Generate multiple high-quality initial configurations"""
        configs = []
        
        # Configuration 1: Hexagonal grid pattern
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
        
        # Configuration 2: Circular arrangement with perturbations
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
        
        # Configuration 3: Modified 4x4 grid with offset rows
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
        
        # Configuration 4: Golden spiral approach
        golden_points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(16):
            angle = 2 * np.pi * i * phi
            radius = 0.4 * np.sqrt(i / 15.0) + 0.05
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            # Add small random perturbation
            x += 0.01 * np.random.randn()
            y += 0.01 * np.random.randn()
            golden_points.append([x, y])
        configs.append(np.array(golden_points))
        
        # Configuration 5: Standard 4x4 grid with perturbations
        standard_points = []
        row_positions = np.linspace(0.1, 0.9, 4)
        col_positions = np.linspace(0.1, 0.9, 4)
        for i, row in enumerate(row_positions):
            for j, col in enumerate(col_positions):
                x = col + (np.random.rand() - 0.5) * 0.05
                y = row + (np.random.rand() - 0.5) * 0.05
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                standard_points.append([x, y])
        configs.append(np.array(standard_points))
        
        return configs
    
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initial_configs()
    
    best_ratio = -np.inf
    best_points = None
    
    # Multi-start optimization with different initial configurations
    for config_idx, initial_config in enumerate(initial_configs):
        # Try different optimization methods with multiple restarts
        methods_to_try = ['L-BFGS-B', 'SLSQP']
        
        for method in methods_to_try:
            # Try with original configuration
            try:
                x0 = initial_config.flatten()
                bounds = [(0, 1) for _ in range(32)]
                
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    optimized_points = result.x.reshape((16, 2))
                    optimized_points = np.clip(optimized_points, 0, 1)
                    
                    # Calculate actual ratio
                    distances = pdist(optimized_points)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
            except Exception:
                continue
            
            # Try with multiple random restarts from each initial configuration
            for restart in range(3):
                try:
                    np.random.seed(config_idx * 100 + restart * 42)
                    perturbed_config = initial_config.copy()
                    perturbed_config += np.random.normal(0, 0.01, size=perturbed_config.shape)
                    perturbed_config = np.clip(perturbed_config, 0, 1)
                    
                    x0 = perturbed_config.flatten()
                    bounds = [(0, 1) for _ in range(32)]
                    
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape((16, 2))
                        optimized_points = np.clip(optimized_points, 0, 1)
                        
                        # Calculate actual ratio
                        distances = pdist(optimized_points)
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
                                
                except Exception:
                    continue
    
    # If no optimization worked, use the first initial configuration
    if best_points is None:
        return initial_configs[0]
    
    return best_points


# EVOLVE-BLOCK-END
