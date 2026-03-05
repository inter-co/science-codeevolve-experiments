# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical construction, global optimization, and multiple restart strategies.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x_flat):
        """Objective function to maximize min/max distance ratio"""
        points = x_flat.reshape(-1, 2)
        distances = pdist(points)
        # Filter out zero distances (same points)
        distances = distances[distances > 0]
        
        if len(distances) == 0:
            return -np.inf
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative because we want to maximize
        if max_dist == 0:
            return -np.inf
        return -min_dist / max_dist
    
    def compute_ratio(points):
        """Helper function to compute min/max ratio"""
        distances = pdist(points)
        distances = distances[distances > 0]  # Remove zero distances
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist
    
    def construct_concentric_circles():
        """Create points in two concentric circles - proven effective configuration"""
        # Create two concentric rings with 8 points each
        points = []
        
        # Inner ring (smaller radius)
        for i in range(8):
            angle = i * np.pi/4
            points.append([0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)])
        
        # Outer ring (larger radius) 
        for i in range(8):
            angle = i * np.pi/4 + np.pi/8  # Offset by 45 degrees
            points.append([0.5 + 0.6 * np.cos(angle), 0.5 + 0.6 * np.sin(angle)])
        
        return np.array(points)
    
    def construct_hexagonal_lattice():
        """
        Generate points in a hexagonal lattice pattern - one of the most effective 
        constructions for point dispersion in 2D.
        """
        # Create a 4x4 grid that forms a hexagonal pattern
        points = []
        # Generate points in a hexagonal pattern
        for i in range(4):
            for j in range(4):
                # Hexagonal offset
                x = j + (i % 2) * 0.5
                y = i * np.sqrt(3)/2
                points.append([x, y])
        
        # Normalize to fit in [0,1] x [0,1]
        points = np.array(points)
        if len(points) > 0:
            # Normalize to fit nicely in unit square
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
            
            if x_max > x_min and y_max > y_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
                
                # Scale and shift to [0.05, 0.95] range to avoid boundary issues
                points[:, 0] = points[:, 0] * 0.9 + 0.05
                points[:, 1] = points[:, 1] * 0.9 + 0.05
        
        return points[:16]  # Ensure exactly 16 points
    
    def construct_fibonacci_spiral():
        """Generate points using Fibonacci spiral approach"""
        n = 16
        golden_ratio = (1 + np.sqrt(5)) / 2
        points = []
        
        for i in range(n):
            angle = i * 2 * np.pi / golden_ratio
            radius = np.sqrt(i / (n - 1)) if n > 1 else 0
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            points.append([x, y])
            
        return np.array(points)
    
    def construct_regular_polygon():
        """Generate points forming a regular 16-gon."""
        points = []
        for i in range(16):
            angle = 2 * np.pi * i / 16
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def construct_better_mathematical_config():
        """Construct a more carefully designed configuration for 16 points"""
        # Create a highly symmetric configuration using rings with different spacings
        points = []
        
        # Ring 1: 4 points evenly spaced at radius 0.25
        for i in range(4):
            angle = i * np.pi/2
            points.append([0.5 + 0.25 * np.cos(angle), 0.5 + 0.25 * np.sin(angle)])
        
        # Ring 2: 4 points evenly spaced at radius 0.5, offset
        for i in range(4):
            angle = i * np.pi/2 + np.pi/8
            points.append([0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)])
        
        # Ring 3: 4 points evenly spaced at radius 0.75
        for i in range(4):
            angle = i * np.pi/2
            points.append([0.5 + 0.75 * np.cos(angle), 0.5 + 0.75 * np.sin(angle)])
        
        # Ring 4: 4 points evenly spaced at radius 0.5, different offset
        for i in range(4):
            angle = i * np.pi/2 + 3*np.pi/8
            points.append([0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)])
        
        return np.array(points)
    
    # Try multiple initialization strategies and optimization runs
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Better mathematical construction (from INSPIRATION 2)
    try:
        initial_config = construct_better_mathematical_config()
        np.random.seed(42)
        perturbed_config = initial_config + np.random.normal(0, 0.01, initial_config.shape)
        perturbed_config = np.clip(perturbed_config, 0, 1)
        x0 = perturbed_config.flatten()
        
        # Use more aggressive optimization with stricter tolerances
        result = minimize(objective, x0, method='SLSQP', 
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            ratio = compute_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 2: Concentric circles (from TARGET)
    try:
        initial_config = construct_concentric_circles()
        np.random.seed(42)
        perturbed_config = initial_config + np.random.normal(0, 0.005, initial_config.shape)
        perturbed_config = np.clip(perturbed_config, 0, 1)
        x0 = perturbed_config.flatten()
        
        result = minimize(objective, x0, method='SLSQP', 
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            ratio = compute_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 3: Fibonacci spiral (from TARGET)
    try:
        initial_config = construct_fibonacci_spiral()
        np.random.seed(42)
        perturbed_config = initial_config + np.random.normal(0, 0.005, initial_config.shape)
        perturbed_config = np.clip(perturbed_config, 0, 1)
        x0 = perturbed_config.flatten()
        
        result = minimize(objective, x0, method='SLSQP', 
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            ratio = compute_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 4: Hexagonal lattice (from INSPIRATION 1)
    try:
        initial_config = construct_hexagonal_lattice()
        np.random.seed(43)
        perturbed_config = initial_config + np.random.normal(0, 0.01, initial_config.shape)
        perturbed_config = np.clip(perturbed_config, 0, 1)
        x0 = perturbed_config.flatten()
        
        result = minimize(objective, x0, method='SLSQP', 
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            ratio = compute_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 5: Regular polygon (from INSPIRATION 1)
    try:
        initial_config = construct_regular_polygon()
        np.random.seed(44)
        perturbed_config = initial_config + np.random.normal(0, 0.01, initial_config.shape)
        perturbed_config = np.clip(perturbed_config, 0, 1)
        x0 = perturbed_config.flatten()
        
        result = minimize(objective, x0, method='SLSQP', 
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            ratio = compute_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 6: Random restarts with different seeds (from INSPIRATION 2)
    for seed in [123, 456, 789]:
        try:
            np.random.seed(seed)
            random_points = np.random.rand(16, 2)
            random_points = np.clip(random_points, 0, 1)
            
            result = minimize(objective, random_points.flatten(), method='SLSQP',
                             options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            continue
    
    # Strategy 7: Fine-tune with L-BFGS-B (from INSPIRATION 1) - with even stricter tolerances
    if best_points is not None:
        try:
            result = minimize(objective, best_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14})
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
    
    # If no successful optimization found, return a default configuration
    if best_points is None:
        return construct_better_mathematical_config()
    
    return best_points


# EVOLVE-BLOCK-END
