# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Voronoi
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, advanced optimization, 
    and multi-start strategies to find high-quality configurations.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        dmin = np.min(distances)
        dmax = np.max(distances)
        return dmin / dmax if dmax > 0 else 0
    
    def generate_hexagonal_grid():
        """Generate points in a hexagonal grid pattern for good distribution"""
        points = []
        # Create a 4x4 grid with offset rows for hexagonal packing
        for i in range(4):
            for j in range(4):
                offset = 0.5 if i % 2 == 1 else 0.0
                x = 0.1 + 0.8 * (j + offset) / 3.0
                y = 0.1 + 0.8 * i / 3.0
                points.append([x, y])
        return np.array(points[:16])
    
    def generate_concentric_rings():
        """Generate points in concentric rings for balanced distribution"""
        points = []
        
        # Center point
        points.append([0.5, 0.5])
        
        # Two rings around center
        for i in range(3):  # 3 rings
            radius = 0.2 + 0.2 * i
            num_points = 4 + 2 * i  # Increasing points per ring
            for j in range(num_points):
                angle = 2 * np.pi * j / num_points
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        
        # Ensure exactly 16 points
        points = points[:16]
        return np.array(points)
    
    def generate_fibonacci_spiral():
        """Generate points using Fibonacci spiral for even distribution"""
        points = []
        n = 16
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for i in range(n):
            # Spiral distribution
            theta = i * 2 * np.pi / (phi * n)
            r = np.sqrt(i / (n - 1)) * 0.4 + 0.1  # Radial distribution
            
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            points.append([x, y])
        
        return np.array(points)
    
    def generate_regular_polygon():
        """Generate points forming a regular polygon"""
        points = []
        for i in range(16):
            angle = 2 * np.pi * i / 16
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def generate_voronoi_like():
        """Generate points using a Voronoi-like distribution with strategic placement"""
        # Start with a regular hexagon and add some variation
        points = []
        for i in range(16):
            angle = 2 * np.pi * i / 16
            # Add some randomness to break symmetry
            radius = 0.4 + 0.05 * np.sin(3 * angle)
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def generate_random_grid():
        """Generate a random but structured grid to avoid poor local optima"""
        points = []
        # Generate a grid with slight randomness
        for i in range(4):
            for j in range(4):
                x = i / 3.0 + (np.random.random() - 0.5) * 0.1
                y = j / 3.0 + (np.random.random() - 0.5) * 0.1
                points.append([x, y])
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    def generate_spherical_code():
        """Generate points approximating a spherical code (good for uniform distribution)"""
        # Generate points on a sphere and project to 2D
        points = []
        # Use fibonacci sphere generation but project to 2D
        n = 16
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(n):
            # Fibonacci sphere approach
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = np.arccos(y)  # angle from z-axis
            phi = i * 2 * np.pi / golden_ratio  # azimuthal angle
            
            # Convert to 2D points (just taking x and y components)
            x = radius * np.cos(phi)
            z = radius * np.sin(phi)
            
            # Project to 2D by dropping z coordinate and scale appropriately
            points.append([0.5 + 0.4 * x, 0.5 + 0.4 * y])
        
        return np.array(points)
    
    def objective(x):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape points
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        # Return negative ratio to maximize (min/max ratio)
        if dmax == 0:
            return 0
        return -dmin / dmax
    
    def smart_optimize(initial_points, max_iter=1000):
        """Apply smart optimization with multiple restarts and adaptive parameters"""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(initial_points)
        
        bounds = [(0, 1)] * 32
        
        # Multiple restart strategies
        restart_strategies = [
            # Strategy 1: Heavy local optimization with tight tolerances
            {
                'methods': [('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})],
                'attempts': 3
            },
            # Strategy 2: Mixed methods for robustness
            {
                'methods': [
                    ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}),
                    ('SLSQP', {'maxiter': 500, 'ftol': 1e-10}),
                    ('TNC', {'maxiter': 500, 'ftol': 1e-10})
                ],
                'attempts': 2
            }
        ]
        
        for strategy in restart_strategies:
            for _ in range(strategy['attempts']):
                # Small random perturbation to break symmetry
                np.random.seed(np.random.randint(1000))
                perturbed = initial_points + np.random.normal(0, 0.005, initial_points.shape)
                perturbed = np.clip(perturbed, 0, 1)
                
                for method, options in strategy['methods']:
                    try:
                        result = minimize(
                            objective,
                            perturbed.flatten(),
                            method=method,
                            bounds=bounds,
                            options=options,
                            tol=1e-12
                        )
                        
                        if result.success:
                            optimized_points = result.x.reshape(-1, 2)
                            optimized_points = np.clip(optimized_points, 0, 1)
                            
                            ratio = compute_min_max_ratio(optimized_points)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
                                
                    except Exception:
                        continue
                        
        return best_points, best_ratio
    
    # Try multiple initialization strategies and optimize each
    initial_configs = [
        generate_hexagonal_grid(),      # Good for uniformity
        generate_concentric_rings(),    # Balanced distribution
        generate_fibonacci_spiral(),    # Even spacing
        generate_regular_polygon(),     # Regular structure
        generate_voronoi_like(),        # Voronoi-inspired
        generate_random_grid(),         # Random grid
        generate_spherical_code()       # Spherical code approximation
    ]
    
    best_points = None
    best_ratio = -float('inf')
    
    # Try multiple optimization approaches
    for i, initial_config in enumerate(initial_configs):
        # Add small random perturbations to break symmetries
        np.random.seed(42 + i)
        perturbations = np.random.normal(0, 0.01, initial_config.shape)
        initial_config += perturbations
        initial_config = np.clip(initial_config, 0, 1)
        
        # Smart optimization with multiple restarts
        optimized_points, ratio = smart_optimize(initial_config)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # If we still have a weak solution, try global optimization with more iterations
    if best_points is None or best_ratio < 0.15:
        try:
            # Use a more aggressive differential evolution approach
            bounds = [(0, 1)] * 32
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=300,
                popsize=20,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except:
            pass
    
    # Final fallback to the best initial configuration if nothing worked
    if best_points is None:
        return initial_configs[0]
    
    return best_points


# EVOLVE-BLOCK-END
