# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust hybrid approach with mathematical construction and aggressive optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        distances = distances[distances > 0]  # Remove zero distances
        if len(distances) == 0:
            return 0.0
        dmin = np.min(distances)
        dmax = np.max(distances)
        return dmin / dmax if dmax > 0 else 0.0
    
    def objective(x_flat):
        """Objective function to maximize min/max distance ratio"""
        points = x_flat.reshape(-1, 2)
        distances = pdist(points)
        distances = distances[distances > 0]  # Remove zero distances
        
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
    
    def construct_hexagonal_lattice():
        """Create points in a hexagonal lattice pattern"""
        points = []
        spacing = 0.25
        row_spacing = spacing * np.sqrt(3)/2
        
        for i in range(4):
            for j in range(4):
                x = j * spacing + (i % 2) * spacing/2
                y = i * row_spacing
                points.append([x, y])
        
        # Convert to numpy array and normalize to [0,1] x [0,1]
        points = np.array(points)
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0 and y_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
                
                # Scale to [0,1] range and center
                points[:, 0] *= 0.95
                points[:, 1] *= 0.95
                points[:, 0] += 0.025
                points[:, 1] += 0.025
        
        return points[:16]  # Ensure exactly 16 points
    
    # Try multiple high-quality initialization strategies with aggressive optimization
    best_ratio = 0.0
    best_points = None
    
    # Strategy 1: Concentric circles (most mathematically sound)
    try:
        initial_points = construct_concentric_circles()
        # Add multiple perturbations for robustness
        for seed in [42, 142, 242]:
            np.random.seed(seed)
            perturbed_points = initial_points + np.random.normal(0, 0.003, initial_points.shape)
            perturbed_points = np.clip(perturbed_points, 0, 1)
            
            result = minimize(
                objective, 
                perturbed_points.flatten(), 
                method='SLSQP', 
                options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 2: Hexagonal lattice (another proven approach)
    try:
        initial_points = construct_hexagonal_lattice()
        # Add multiple perturbations for robustness
        for seed in [43, 143, 243]:
            np.random.seed(seed)
            perturbed_points = initial_points + np.random.normal(0, 0.003, initial_points.shape)
            perturbed_points = np.clip(perturbed_points, 0, 1)
            
            result = minimize(
                objective, 
                perturbed_points.flatten(), 
                method='SLSQP', 
                options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 3: Random restarts with very tight tolerances
    for seed in [100, 200, 300, 400, 500]:
        try:
            np.random.seed(seed)
            random_points = np.random.rand(16, 2)
            random_points = np.clip(random_points, 0, 1)
            
            result = minimize(
                objective, 
                random_points.flatten(), 
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
            continue
    
    # Final aggressive refinement
    if best_points is not None:
        try:
            # Try with even more aggressive optimization
            result = minimize(
                objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                if ratio > best_ratio:
                    best_points = optimized_points.copy()
        except Exception:
            pass
    
    # Return best solution found
    if best_points is None:
        # Fallback to concentric circles if nothing works
        return construct_concentric_circles()
    
    return best_points


# EVOLVE-BLOCK-END
