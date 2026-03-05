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
    
    def construct_octagon_config():
        """Construct points in two concentric octagons - proven mathematical approach"""
        points = []
        
        # Inner octagon
        for i in range(8):
            angle = i * np.pi/4
            points.append([0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)])
        
        # Outer octagon rotated by π/8
        for i in range(8):
            angle = i * np.pi/4 + np.pi/8
            points.append([0.5 + 0.6 * np.cos(angle), 0.5 + 0.6 * np.sin(angle)])
        
        return np.array(points)
    
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
    
    def construct_hexagonal_lattice():
        """Construct points using hexagonal lattice pattern"""
        # Arrange 16 points in a hexagonal pattern (4 rows, 4 columns with offset)
        points = []
        row_offsets = [0, 0.5, 0, 0.5]  # Alternating column offsets
        spacing = 0.25  # Spacing between points
        
        for row in range(4):
            for col in range(4):
                x = 0.1 + col * spacing + row_offsets[row] * spacing
                y = 0.1 + row * spacing * np.sqrt(3)/2
                points.append([x, y])
        
        return np.array(points)
    
    def construct_regular_polygon():
        """Construct points in regular polygon configuration"""
        points = []
        # 16 points in a circle
        for i in range(16):
            angle = i * 2 * np.pi / 16
            points.append([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
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
    
    # Try multiple high-quality initialization strategies
    strategies = [
        ("octagon", construct_octagon_config()),
        ("concentric_circles", construct_concentric_circles()),
        ("fibonacci", construct_fibonacci_spiral()),
        ("hexagonal", construct_hexagonal_lattice()),
        ("regular_polygon", construct_regular_polygon())
    ]
    
    best_ratio = 0.0
    best_points = None
    
    # More aggressive optimization with better parameters
    for strategy_name, initial_points in strategies:
        # Add small random perturbations to break symmetry and improve optimization
        np.random.seed(42)
        perturbed_points = initial_points + np.random.normal(0, 0.005, initial_points.shape)
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        try:
            # Try multiple optimization methods for better results
            # First try SLSQP with more iterations
            result = minimize(
                objective, 
                perturbed_points.flatten(), 
                method='SLSQP', 
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9, 'disp': False}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    # Try additional random restarts with different seeds for better exploration
    # More aggressive random restarts to find better solutions
    # Use more seeds and different optimization methods
    restart_seeds = [123, 456, 789, 999, 111, 222, 333, 555, 666, 888]
    
    for seed in restart_seeds:
        try:
            np.random.seed(seed)
            random_points = np.random.rand(16, 2)
            random_points = np.clip(random_points, 0, 1)
            
            # Try both SLSQP and L-BFGS-B for better exploration
            for method in ['SLSQP', 'L-BFGS-B']:
                try:
                    result = minimize(
                        objective, 
                        random_points.flatten(), 
                        method=method, 
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 2)
                        ratio = compute_ratio(optimized_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            break  # Found a good solution, move to next seed
                except Exception:
                    continue
                    
        except Exception:
            continue
    
    # If we have a good solution, try fine-tuning with multiple methods
    if best_points is not None:
        try:
            # Try multiple fine-tuning approaches with tighter tolerances
            fine_tune_methods = [
                ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('SLSQP', {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            ]
            
            for method, options in fine_tune_methods:
                try:
                    result = minimize(
                        objective, 
                        best_points.flatten(), 
                        method=method, 
                        options=options
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 2)
                        ratio = compute_ratio(optimized_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            break  # Found better solution, stop fine-tuning
                except Exception:
                    continue
                    
        except Exception:
            pass
    
    # If no good solution found, return the concentric circles configuration
    if best_points is None:
        return construct_concentric_circles()
    
    return best_points


# EVOLVE-BLOCK-END
