# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses an advanced multi-start approach with geometric initialization and adaptive optimization strategies.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance"""
        if len(points) < 2:
            return 0
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 1e-12:
            return 0
        
        return d_min / d_max
    
    def objective(x):
        """Objective function: minimize negative of min/max distance ratio"""
        # Reshape x into 16 points
        points = x.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0:
            return -np.inf
            
        # Compute min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio since we want to maximize
        if d_max <= 1e-12:
            return -np.inf
        return -d_min / d_max
    
    def generate_hexagonal_arrangement():
        """Generate points arranged in a hexagonal pattern"""
        points = []
        # Use 4x4 grid with hexagonal offset
        rows = 4
        cols = 4
        spacing = 0.25  # More compact spacing
        offset = spacing * 0.5
        
        for i in range(rows):
            for j in range(cols):
                x = j * spacing
                y = i * spacing * np.sqrt(3) / 2
                if i % 2 == 1:  # Offset every other row
                    x += offset
                points.append([x, y])
        
        # Trim to exactly 16 points if needed
        points = points[:16]
        # Add slight randomization to avoid degenerate cases
        points = np.array(points) + np.random.normal(0, 0.005, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def generate_fibonacci_arrangement():
        """Generate points using Fibonacci-like distribution"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(16):
            theta = 2 * np.pi * i / golden_ratio
            # Distribute points more evenly
            radius = np.sqrt(i / 15.0) * 0.4
            x = 0.5 + radius * np.cos(theta)
            y = 0.5 + radius * np.sin(theta)
            points.append([x, y])
        
        points = np.array(points) + np.random.normal(0, 0.01, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def generate_grid_arrangement():
        """Generate points in a regular grid"""
        points = []
        for i in range(4):
            for j in range(4):
                points.append([i/3.0, j/3.0])
        
        # Add slight randomization
        points = np.array(points) + np.random.normal(0, 0.01, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def generate_spherical_code_arrangement():
        """Generate points inspired by spherical codes"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(16):
            theta = 2 * np.pi * i / 16
            # Distribute points along a circle with some variation
            r = 0.4 + 0.3 * np.sin(i * golden_ratio)
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            points.append([x, y])
        
        # Add some perturbation
        points = np.array(points) + np.random.normal(0, 0.01, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def generate_custom_arrangement():
        """Generate a custom arrangement with better distribution"""
        # Start with a regular grid and add some perturbation
        points = []
        for i in range(4):
            for j in range(4):
                points.append([i/3.0, j/3.0])
        
        # Perturb slightly with more controlled randomness
        points = np.array(points) + np.random.normal(0, 0.02, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def generate_regular_polygon_arrangement():
        """Generate points based on regular polygon vertices"""
        points = []
        # Regular 16-gon inscribed in unit square
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        # Scale to fit nicely in [0,1] x [0,1]
        scale = 0.4
        center = [0.5, 0.5]
        
        for angle in angles:
            x = center[0] + scale * np.cos(angle)
            y = center[1] + scale * np.sin(angle)
            points.append([x, y])
            
        points_array = np.array(points)
        points_array = np.clip(points_array, 0, 1)
        return points_array
    
    def generate_polar_arrangement():
        """Generate points in a polar arrangement for diversity"""
        points = []
        # Place points in a circular pattern with radial distribution
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.9, 16)
        
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            points.append([x, y])
            
        points_array = np.array(points)
        points_array = np.clip(points_array, 0, 1)
        return points_array
    
    def generate_voronoi_like_arrangement():
        """Generate points inspired by Voronoi-like distributions"""
        # Start with a regular triangular lattice pattern
        points = []
        # Create a triangular grid with some randomness
        for i in range(4):
            for j in range(4):
                x = i * 0.3 + (j % 2) * 0.15
                y = j * 0.3
                points.append([x, y])
        
        # Trim to 16 points and add randomness
        points = points[:16]
        points = np.array(points) + np.random.normal(0, 0.01, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    # Generate multiple diverse initial configurations (enhanced from INSPIRATION 2)
    initial_configs = [
        generate_hexagonal_arrangement(),
        generate_fibonacci_arrangement(), 
        generate_grid_arrangement(),
        generate_spherical_code_arrangement(),
        generate_custom_arrangement(),
        generate_regular_polygon_arrangement(),
        generate_polar_arrangement(),
        generate_voronoi_like_arrangement(),
        # Add some random configurations for extra diversity
        np.random.uniform(0.1, 0.9, (16, 2)),
        np.random.rand(16, 2) * 0.8 + 0.1,  # Scaled random points
        np.random.rand(16, 2)  # Pure random points
    ]
    
    # Evaluate all initial configurations and find the best one
    best_initial_config = None
    best_initial_ratio = -np.inf
    
    for config in initial_configs:
        ratio = compute_min_max_ratio(config)
        if ratio > best_initial_ratio:
            best_initial_ratio = ratio
            best_initial_config = config.copy()
    
    # Multi-start optimization approach with improved parameters:
    bounds = [(0, 1)] * 32
    
    # Strategy 1: Differential Evolution with optimized parameters
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=30,  # Reduced iterations for faster execution
            popsize=12,   # Smaller population size for speed
            mutation=(0.5, 1),  # Standard mutation
            recombination=0.7,   # Standard recombination
            tol=1e-6,
            disp=False
        )
        best_x = de_result.x.copy()
        best_value = de_result.fun
    except Exception:
        best_x = best_initial_config.flatten()
        best_value = objective(best_x)
    
    # Strategy 2: Multiple local optimizations with different methods
    methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
    for method in methods_to_try:
        try:
            local_result = minimize(
                objective,
                best_x,
                method=method,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            
            if local_result.fun < best_value:
                best_x = local_result.x.copy()
                best_value = local_result.fun
        except Exception:
            continue
    
    # Strategy 3: Strategic restarts with better diversification (enhanced from INSPIRATION 2)
    best_points = best_x.reshape(-1, 2)
    best_ratio = compute_min_max_ratio(best_points)
    
    # Perform multiple restarts with different strategies to escape local optima
    for restart in range(10):  # Increased restarts to improve chance of finding better solution
        np.random.seed(restart + 42)
        
        # Create perturbed version with controlled noise - more varied approach
        if restart % 5 == 0:
            # Large perturbation (more aggressive)
            perturbed = best_points + np.random.normal(0, 0.04, best_points.shape)
        elif restart % 5 == 1:
            # Medium-large perturbation  
            perturbed = best_points + np.random.normal(0, 0.025, best_points.shape)
        elif restart % 5 == 2:
            # Medium perturbation
            perturbed = best_points + np.random.normal(0, 0.015, best_points.shape)
        elif restart % 5 == 3:
            # Small perturbation
            perturbed = best_points + np.random.normal(0, 0.01, best_points.shape)
        else:
            # Very small perturbation
            perturbed = best_points + np.random.normal(0, 0.005, best_points.shape)
            
        perturbed = np.clip(perturbed, 0, 1)
        
        try:
            # Local optimization from this perturbed starting point
            restart_result = minimize(
                objective,
                perturbed.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            
            if restart_result.success:
                restart_points = restart_result.x.reshape(-1, 2)
                restart_points = np.clip(restart_points, 0, 1)
                restart_ratio = compute_min_max_ratio(restart_points)
                
                if restart_ratio > best_ratio:
                    best_ratio = restart_ratio
                    best_points = restart_points.copy()
                    
        except Exception:
            continue
    
    return best_points


# EVOLVE-BLOCK-END
