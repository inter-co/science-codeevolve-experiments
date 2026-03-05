# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and global optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    
    def compute_ratio(points):
        """Compute the min/max distance ratio"""
        if len(points) < 2:
            return 0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return 0
            
        return min_dist / max_dist
    
    def objective(x_flat):
        """Objective function to minimize (negative ratio)"""
        points = x_flat.reshape(-1, 2)
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 1e10
        return -d_min / d_max  # Negative because we want to maximize
    
    def create_regular_polygon_initialization():
        """Create points on a regular 16-gon - proven good starting point"""
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Scale and center in unit square [0,1] x [0,1]
        center = np.mean(points, axis=0)
        scaled_points = (points - center) * 0.4 + 0.5
        
        # Add small random perturbations to break symmetry
        np.random.seed(42)
        scaled_points += np.random.normal(0, 0.01, scaled_points.shape)
        scaled_points = np.clip(scaled_points, 0, 1)
        return scaled_points
    
    # Multiple restarts with different initialization strategies
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Regular polygon initialization with high iterations (from INSPIRATION 1)
    try:
        initial_config = create_regular_polygon_initialization()
        bounds = [(0, 1)] * (2 * n)
        
        # Use dual annealing with high iterations for better exploration (from INSPIRATION 1)
        result = dual_annealing(
            objective, 
            bounds, 
            maxiter=3000,  # Higher iterations for better exploration (from INSPIRATION 1)
            seed=42,       # Fixed seed for reproducibility
            no_local_search=False  # Allow local search for better results (from INSPIRATION 2)
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            ratio = compute_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 2: Additional random restart with different seed and more iterations
    if best_points is None:
        try:
            # Create a different initialization with different random seed
            np.random.seed(123)
            initial_config = np.random.rand(n, 2)
            bounds = [(0, 1)] * (2 * n)
            
            result = dual_annealing(
                objective, 
                bounds, 
                maxiter=3000,  # More iterations (from INSPIRATION 1)
                seed=123,
                no_local_search=False
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
    
    # Strategy 3: Another random restart with yet another seed
    if best_points is None:
        try:
            np.random.seed(456)
            initial_config = np.random.rand(n, 2)
            bounds = [(0, 1)] * (2 * n)
            
            result = dual_annealing(
                objective, 
                bounds, 
                maxiter=2500,
                seed=456,
                no_local_search=False
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
    
    # If no success from global optimization, fallback to regular polygon
    if best_points is None:
        best_points = create_regular_polygon_initialization()
    
    # Final refinement with L-BFGS-B optimization for tight convergence (from INSPIRATION 2)
    try:
        result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(2 * n)],
            options={'maxiter': 4000, 'ftol': 1e-18, 'gtol': 1e-18},  # Tighter tolerances (from INSPIRATION 2)
            tol=1e-18
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            ratio = compute_ratio(final_points)
            
            if ratio > best_ratio:
                best_points = final_points
    except Exception:
        pass
    
    # Additional L-BFGS-B refinement if needed
    try:
        result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(2 * n)],
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16},
            tol=1e-16
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            ratio = compute_ratio(final_points)
            
            if ratio > best_ratio:
                best_points = final_points
    except Exception:
        pass
    
    # Final safeguard: ensure all points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
