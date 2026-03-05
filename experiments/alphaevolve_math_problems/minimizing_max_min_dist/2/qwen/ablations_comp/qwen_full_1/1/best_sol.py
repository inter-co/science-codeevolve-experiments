# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple geometric initializations with robust optimization.
    Inspired by mathematical constructions and optimal point distributions.

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
    
    def constraint_func(x_flat):
        """Constraint function to keep points within unit square"""
        points = x_flat.reshape(-1, 2)
        # Return positive values when constraints are satisfied
        # Points should be in [0,1] x [0,1]
        return np.concatenate([
            points[:, 0],      # x >= 0
            1 - points[:, 0],  # x <= 1
            points[:, 1],      # y >= 0
            1 - points[:, 1]   # y <= 1
        ])
    
    # Generate multiple high-quality initial configurations
    def generate_multiple_initial_configs():
        configs = []
        n = 16
        
        # Configuration 1: Golden spiral approach (inspired by mathematical optimality)
        golden_points = []
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            angle = 2 * math.pi * i * phi
            radius = 0.4 * math.sqrt(i / (n - 1.0)) + 0.05
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            # Add small random perturbation
            x += 0.01 * np.random.randn()
            y += 0.01 * np.random.randn()
            golden_points.append([x, y])
        configs.append(np.array(golden_points))
        
        # Configuration 2: Hexagonal grid with perturbations (mathematically motivated)
        hex_points = []
        for i in range(4):
            for j in range(4):
                if len(hex_points) < n:
                    x = (i + 0.5) / 4.0
                    y = (j + 0.5) / 4.0
                    # Add structured perturbations for better distribution
                    x += np.random.normal(0, 0.015) * (1 + 0.1 * np.sin(i * 0.5))
                    y += np.random.normal(0, 0.015) * (1 + 0.1 * np.cos(j * 0.5))
                    x = max(0.01, min(0.99, x))
                    y = max(0.01, min(0.99, y))
                    hex_points.append([x, y])
        configs.append(np.array(hex_points))
        
        # Configuration 3: Modified 4x4 grid with offset rows (symmetric structure)
        grid_points = []
        for i in range(4):
            for j in range(4):
                if len(grid_points) < n:
                    x = (i + 0.5) / 4.0
                    y = (j + 0.5) / 4.0
                    # Add offset for hexagonal-like structure
                    if j % 2 == 1:
                        x += 0.125
                    x = max(0.01, min(0.99, x))
                    y = max(0.01, min(0.99, y))
                    grid_points.append([x, y])
        configs.append(np.array(grid_points))
        
        # Configuration 4: Circular arrangement with perturbations (good for uniformity)
        circular_points = []
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
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
        
        # Configuration 5: Standard 4x4 grid with random perturbations (robust baseline)
        standard_points = []
        row_positions = np.linspace(0.1, 0.9, 4)
        col_positions = np.linspace(0.1, 0.9, 4)
        for i, row in enumerate(row_positions):
            for j, col in enumerate(col_positions):
                if len(standard_points) < n:
                    x = col + (np.random.rand() - 0.5) * 0.05
                    y = row + (np.random.rand() - 0.5) * 0.05
                    x = max(0.01, min(0.99, x))
                    y = max(0.01, min(0.99, y))
                    standard_points.append([x, y])
        configs.append(np.array(standard_points))
        
        # Configuration 6: Equiangular configuration (based on optimal spherical codes)
        equiangular_points = []
        # Arrange points on a circle but with more careful spacing
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            # Add small perturbations to avoid degeneracies
            x += 0.02 * np.random.randn()
            y += 0.02 * np.random.randn()
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            equiangular_points.append([x, y])
        configs.append(np.array(equiangular_points))
        
        # Configuration 7: Modified hexagonal lattice (more mathematically principled)
        hex_lattice_points = []
        # Generate points in a hexagonal lattice pattern but with better distribution
        k = 0
        for i in range(-1, 3):
            for j in range(-1, 3):
                if k >= n:
                    break
                x = 0.25 + i * 0.25
                y = 0.25 + j * 0.25
                # Apply hexagonal offset
                if j % 2 == 1:
                    x += 0.125
                # Add slight randomization
                x += 0.01 * np.random.randn()
                y += 0.01 * np.random.randn()
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                hex_lattice_points.append([x, y])
                k += 1
        configs.append(np.array(hex_lattice_points[:n]))
        
        return configs
    
    # Get multiple initial configurations
    initial_configs = generate_multiple_initial_configs()
    
    # Set up bounds and constraints for optimization
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_ratio = -np.inf
    best_points = None
    
    # Multi-start optimization with different initial configurations
    for config_idx, initial_config in enumerate(initial_configs):
        # Try optimization from this initial configuration with multiple methods
        methods_to_try = ['L-BFGS-B', 'SLSQP', 'TNC']
        
        for method in methods_to_try:
            try:
                # Create a slightly perturbed version of the initial config
                np.random.seed(config_idx * 100 + 42)
                perturbed_config = initial_config.copy()
                # Add small random perturbations to escape local minima
                perturbed_config += np.random.normal(0, 0.01, size=perturbed_config.shape)
                perturbed_config = np.clip(perturbed_config, 0, 1)
                
                x0 = perturbed_config.flatten()
                
                # Optimize with selected method - increased iteration limits for better convergence
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
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
    
    # If no optimization worked, return the first initial configuration
    if best_points is None:
        return initial_configs[0]
    
    return best_points


# EVOLVE-BLOCK-END
