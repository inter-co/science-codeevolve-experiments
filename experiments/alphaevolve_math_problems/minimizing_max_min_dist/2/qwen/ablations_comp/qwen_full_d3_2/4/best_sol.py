# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a focused approach combining mathematical construction and robust optimization.
    
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
    
    # Start with the most promising mathematical configuration
    best_ratio = 0.0
    best_points = None
    
    # Strategy 1: Concentric circles with high precision optimization
    initial_points = construct_concentric_circles()
    
    # Add small random perturbations to break symmetry
    np.random.seed(42)
    perturbed_points = initial_points + np.random.normal(0, 0.001, initial_points.shape)
    perturbed_points = np.clip(perturbed_points, 0, 1)
    
    try:
        result = minimize(
            objective, 
            perturbed_points.flatten(), 
            method='SLSQP', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            ratio = compute_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
    except Exception:
        pass
    
    # Strategy 2: Fibonacci spiral with high precision optimization
    initial_points = construct_fibonacci_spiral()
    
    # Add small random perturbations to break symmetry
    np.random.seed(42)
    perturbed_points = initial_points + np.random.normal(0, 0.001, initial_points.shape)
    perturbed_points = np.clip(perturbed_points, 0, 1)
    
    try:
        result = minimize(
            objective, 
            perturbed_points.flatten(), 
            method='SLSQP', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            ratio = compute_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
    except Exception:
        pass
    
    # Strategy 3: Multiple random restarts with refined approach
    for seed in [123, 456, 789, 999]:
        try:
            np.random.seed(seed)
            # Generate points with better distribution
            random_points = np.random.rand(16, 2) * 0.8 + 0.1  # Centered in [0.1, 0.9] range
            random_points = np.clip(random_points, 0, 1)
            
            result = minimize(
                objective, 
                random_points.flatten(), 
                method='SLSQP', 
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    # Strategy 4: Try L-BFGS-B with best solution so far for final refinement
    if best_points is not None:
        try:
            result = minimize(
                objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            pass
    
    # Strategy 5: If no good solution found, try a hybrid approach with grid-based initialization
    if best_points is None:
        try:
            # Try a grid-based initialization
            grid_points = []
            for i in range(4):
                for j in range(4):
                    x = 0.1 + i * 0.225
                    y = 0.1 + j * 0.225
                    grid_points.append([x, y])
            grid_points = np.array(grid_points[:16])  # Take first 16
            
            # Add noise to break symmetries
            np.random.seed(42)
            noisy_points = grid_points + np.random.normal(0, 0.01, grid_points.shape)
            noisy_points = np.clip(noisy_points, 0, 1)
            
            result = minimize(
                objective, 
                noisy_points.flatten(), 
                method='SLSQP', 
                options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            pass
    
    # If no good solution found, return the concentric circles configuration
    if best_points is None:
        return construct_concentric_circles()
    
    return best_points


# EVOLVE-BLOCK-END
