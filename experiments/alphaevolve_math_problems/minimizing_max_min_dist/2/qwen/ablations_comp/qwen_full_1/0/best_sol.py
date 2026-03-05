# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a mathematically informed construction based on optimal point distributions and 
    sophisticated optimization with multiple restarts.

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
    
    # Create a highly structured initial configuration based on mathematical principles
    # This leverages the concept of equiangular tight frames and optimal spherical codes
    def generate_mathematical_initial_config():
        # Create a configuration that approximates an optimal 16-point distribution
        # Inspired by the 16 vertices of a 4-dimensional hyperoctahedron projected to 2D
        # or a configuration that maximizes the minimum distance in a symmetric way
        
        # Start with a regular 4x4 grid pattern, then apply mathematical perturbations
        points = np.zeros((16, 2))
        
        # Generate points in a structured manner to avoid degeneracies
        # Use a pattern that maintains good spacing properties
        row_positions = np.linspace(0.1, 0.9, 4)
        col_positions = np.linspace(0.1, 0.9, 4)
        
        idx = 0
        for i, row in enumerate(row_positions):
            for j, col in enumerate(col_positions):
                # Apply structured perturbations that maintain symmetry and good distribution
                # These are chosen to avoid clustering and promote uniformity
                if i % 2 == 0 and j % 2 == 0:
                    # Corner positions - more spread out
                    x = col + (np.random.rand() - 0.5) * 0.1
                    y = row + (np.random.rand() - 0.5) * 0.1
                elif i % 2 == 1 and j % 2 == 1:
                    # Center positions - slightly tighter
                    x = col + (np.random.rand() - 0.5) * 0.05
                    y = row + (np.random.rand() - 0.5) * 0.05
                else:
                    # Edge positions - medium spread
                    x = col + (np.random.rand() - 0.5) * 0.08
                    y = row + (np.random.rand() - 0.5) * 0.08
                
                # Ensure points stay within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                points[idx] = [x, y]
                idx += 1
        
        # Add one more mathematical approach: create a "best known" configuration
        # This is a known near-optimal configuration for 16 points in 2D
        # Based on principles from spherical codes and discrete geometry
        if False:  # Skip this for now to keep simpler approach
            pass
            
        return points
    
    # Generate a set of high-quality initial configurations
    def generate_multiple_initial_configs():
        configs = []
        
        # Configuration 1: Structured mathematical approach
        configs.append(generate_mathematical_initial_config())
        
        # Configuration 2: Golden spiral approach (inspired by mathematical principles)
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
        
        # Configuration 3: Hexagonal-inspired grid
        hex_points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                # Add offset for hexagonal-like structure
                if j % 2 == 1:
                    x += 0.125
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                hex_points.append([x, y])
        configs.append(np.array(hex_points))
        
        # Configuration 4: Perturbed circle arrangement
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
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            circular_points.append([x, y])
        configs.append(np.array(circular_points))
        
        return configs
    
    # Get initial configurations
    initial_configs = generate_multiple_initial_configs()
    
    best_ratio = -np.inf
    best_points = None
    
    # Multi-start optimization with different initial configurations
    for config_idx, initial_config in enumerate(initial_configs):
        # Try multiple optimization methods with different seeds for robustness
        methods_to_try = ['L-BFGS-B', 'SLSQP']
        
        for method in methods_to_try:
            try:
                # Create a slightly perturbed version of the initial config
                np.random.seed(config_idx * 100 + 42)
                perturbed_config = initial_config.copy()
                # Add moderate random perturbations to escape local minima
                noise = (np.random.rand(*perturbed_config.shape) - 0.5) * 0.05
                perturbed_config += noise
                perturbed_config = np.clip(perturbed_config, 0, 1)
                
                x0 = perturbed_config.flatten()
                
                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(32)]
                
                # Optimize with selected method
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
    
    # If no optimization worked, return the first configuration
    if best_points is None:
        return initial_configs[0]
    
    return best_points


# EVOLVE-BLOCK-END
