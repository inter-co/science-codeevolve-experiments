# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and multi-start optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero or near-zero values
        if d_max <= 1e-12:
            return 0
        
        return d_min / d_max
    
    def objective(params):
        # Reshape parameters back to points
        points = params.reshape((16, 2))
        
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 1e10
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero or near-zero values
        if d_max <= 1e-12:
            return 1e10
            
        # Return negative ratio since we want to maximize
        return -d_min / d_max
    
    def initialize_hexagonal_pattern():
        """Initialize points in a refined hexagonal pattern"""
        # Create a more refined hexagonal grid pattern
        # Using 4x4 grid approach for better coverage
        rows = 4
        cols = 4
        
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Hexagonal offset - more precise placement
                x = j + 0.5 * (i % 2)
                y = i * np.sqrt(3) / 2
                points.append([x, y])
        
        # Normalize to fit in unit square and center
        points = np.array(points[:16])
        
        if len(points) > 0:
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)
            # Avoid division by zero
            range_x = max_x - min_x if max_x - min_x > 1e-10 else 1
            range_y = max_y - min_y if max_y - min_y > 1e-10 else 1
            
            # Scale to [0.1, 0.9] to keep margin from boundaries
            points[:, 0] = (points[:, 0] - min_x) / range_x * 0.8 + 0.1
            points[:, 1] = (points[:, 1] - min_y) / range_y * 0.8 + 0.1
        
        return points
    
    # Use hexagonal pattern as primary initialization (inspired by best insps)
    points = initialize_hexagonal_pattern()
    
    # Also try a golden spiral initialization as backup (from inspiration 2)
    def initialize_golden_spiral():
        """Initialize points using golden spiral pattern for better dispersion"""
        golden_angle = np.pi * (3 - np.sqrt(5))
        points = []
        for i in range(16):
            radius = np.sqrt(i / 15)  # Radial distribution
            angle = i * golden_angle
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
        
        initial_points = np.array(points)
        
        # Scale and center in [0,1] x [0,1]
        x_range = np.max(initial_points[:, 0]) - np.min(initial_points[:, 0])
        y_range = np.max(initial_points[:, 1]) - np.min(initial_points[:, 1])
        
        if x_range > 0:
            initial_points[:, 0] = (initial_points[:, 0] - np.min(initial_points[:, 0])) / x_range
        if y_range > 0:
            initial_points[:, 1] = (initial_points[:, 1] - np.min(initial_points[:, 1])) / y_range
            
        # Scale to fit nicely in unit square
        initial_points[:, 0] = initial_points[:, 0] * 0.8 + 0.1
        initial_points[:, 1] = initial_points[:, 1] * 0.8 + 0.1
        
        # Add small random perturbations to break symmetry
        np.random.seed(42)
        perturbed_points = initial_points + np.random.normal(0, 0.005, initial_points.shape)
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        return perturbed_points
    
    # Try golden spiral initialization as second option
    golden_points = initialize_golden_spiral()
    
    # Evaluate both initializations and use the better one
    hex_ratio = compute_min_max_ratio(points)
    golden_ratio = compute_min_max_ratio(golden_points)
    
    # Use the better initial configuration
    points = points if hex_ratio > golden_ratio else golden_points
    
    # Multi-start optimization with multiple strategies
    best_ratio = 0
    best_points = points.copy()
    
    # Strategy 1: Multiple restarts with local optimization (as in inspiration 1 & 3)
    for restart in range(10):  # Reduced from 15 to save time
        np.random.seed(42 + restart * 100)
        
        # Start with slightly perturbed version of best initialization
        initial_points = points + np.random.normal(0, 0.03, (16, 2))
        initial_points = np.clip(initial_points, 0, 1)
        
        # Try multiple optimization methods for robustness
        methods_to_try = ['L-BFGS-B', 'SLSQP']
        
        for method in methods_to_try:
            try:
                result = minimize(
                    objective,
                    initial_points.flatten(),
                    method=method,
                    options={'maxiter': 400, 'ftol': 1e-8, 'gtol': 1e-8}  # Reduced iterations for speed
                )
                
                if result.success:
                    optimized_points = result.x.reshape((16, 2))
                    ratio = compute_min_max_ratio(optimized_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
            except Exception:
                continue
    
    # Strategy 2: Include differential evolution for global search (inspired by INSPIRATION 2)
    try:
        from scipy.optimize import differential_evolution
        bounds = [(0, 1) for _ in range(32)]  # 16 points in 2D
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=20,  # Reduced iterations for speed
            popsize=8,   # Reduced population size for speed
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False
        )
        
        if de_result.success:
            optimized_points = de_result.x.reshape(-1, 2)
            ratio = compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
    except Exception:
        pass
    
    # Strategy 3: Try additional initial configurations (inspired by INSPIRATION 2)
    def create_alternative_initialization():
        """Create alternative initial configurations"""
        configs = []
        
        # Configuration 1: Uniform random
        np.random.seed(123)
        configs.append(np.random.rand(16, 2))
        
        # Configuration 2: Spiral pattern
        angles = np.linspace(0, 4*np.pi, 16)
        radii = np.linspace(0.1, 0.4, 16)
        x = 0.5 + radii * np.cos(angles)
        y = 0.5 + radii * np.sin(angles)
        configs.append(np.column_stack([x, y]))
        
        # Configuration 3: Square grid with jitter
        grid_points = []
        for i in range(4):
            for j in range(4):
                grid_points.append([i/3.0, j/3.0])
        grid_points = np.array(grid_points[:16])
        # Add small jitter
        jitter = np.random.normal(0, 0.02, (16, 2))
        configs.append(np.clip(grid_points + jitter, 0, 1))
        
        return configs
    
    # Try additional initial configurations with local optimization
    alt_configs = create_alternative_initialization()
    for config in alt_configs:
        # Perturb the configuration slightly
        perturbed_config = config + np.random.normal(0, 0.01, (16, 2))
        perturbed_config = np.clip(perturbed_config, 0, 1)
        
        # Local optimization on this configuration
        try:
            result = minimize(
                objective,
                perturbed_config.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                optimized_points = result.x.reshape((16, 2))
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    # Final refinement with gradient-based optimization
    try:
        final_result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if final_result.success:
            optimized_points = final_result.x.reshape((16, 2))
            ratio = compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
