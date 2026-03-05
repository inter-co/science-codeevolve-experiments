# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a multi-strategy approach combining different initialization methods and optimization techniques.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective_function(coords):
        """Objective function to maximize the min/max distance ratio"""
        # Reshape flat array back to 16x2 points
        points = coords.reshape((16, 2))
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Handle edge case where there are no distances (all points coincide)
        if len(distances) == 0:
            return float('inf')
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio (we want to maximize ratio, so minimize negative ratio)
        if max_dist > 0:
            return -min_dist / max_dist
        else:
            return float('inf')
    
    def generate_hexagonal_perturbed():
        """Generate initial points using a hexagonal pattern with random perturbations"""
        # Create a hexagonal lattice pattern
        points = []
        
        # Hexagonal pattern parameters
        spacing = 1.0
        row_spacing = spacing * math.sqrt(3) / 2
        
        # Generate points in a hexagonal arrangement
        for i in range(4):
            for j in range(4):
                x = j * spacing + (i % 2) * spacing / 2
                y = i * row_spacing
                
                # Ensure points stay within [0,1] bounds
                if x <= 1 and y <= 1:
                    points.append([x, y])
        
        # Trim to exactly 16 points if needed
        if len(points) > 16:
            points = points[:16]
        elif len(points) < 16:
            # Add additional points using a spiral pattern
            for i in range(16 - len(points)):
                angle = i * 0.5
                radius = 0.3 + i * 0.05
                x = 0.5 + radius * math.cos(angle)
                y = 0.5 + radius * math.sin(angle)
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Add small random perturbations to break symmetries
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, points.shape)
        points = points + noise
        points = np.clip(points, 0, 1)
        
        return points
    
    def generate_fibonacci_sphere():
        """Generate points using Fibonacci spiral (approximation)"""
        points = []
        n = 16
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(n):
            # Fibonacci spiral on sphere (projected to 2D)
            phi = math.acos(-1 + (2 * i) / (n - 1))
            theta = math.sqrt(n * math.pi) * phi
            
            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            
            # Project to 2D and scale to unit square
            points.append([0.5 + 0.4 * x, 0.5 + 0.4 * y])
        
        points = np.array(points)
        return np.clip(points, 0, 1)
    
    def generate_random_spherical():
        """Generate points with random distribution that tends to spread well"""
        points = np.random.rand(16, 2)
        # Add some structure to avoid overly clustered solutions
        np.random.seed(42)
        for i in range(16):
            if i % 3 == 0:
                points[i] = [0.5 + 0.3 * (np.random.rand() - 0.5), 0.5 + 0.3 * (np.random.rand() - 0.5)]
        return np.clip(points, 0, 1)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist
    
    def optimize_with_differential_evolution(initial_points):
        """Use differential evolution for global optimization"""
        # Define bounds: [0,1] for both coordinates
        bounds = [(0, 1) for _ in range(32)]
        
        # Use differential evolution with tuned parameters
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=250,     # More iterations for better convergence
                popsize=30,      # Larger population for better exploration
                mutation=(0.5, 1.0),  # Mutation range
                recombination=0.8,    # Higher recombination rate
                seed=42,
                disp=False,
                tol=1e-9         # Tighter tolerance
            )
            
            if result.success:
                optimized_points = result.x.reshape(16, 2)
                return np.clip(optimized_points, 0, 1)
        except Exception:
            pass
        
        return initial_points.copy()
    
    def fine_tune_with_local_optimization(initial_points):
        """Apply local optimization to fine-tune the solution"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Try multiple local optimization methods with different settings
        methods_and_settings = [
            ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-12}),
            ('TNC', {'maxiter': 500, 'ftol': 1e-12}),
            ('SLSQP', {'maxiter': 500, 'ftol': 1e-12})
        ]
        
        for method, options in methods_and_settings:
            try:
                def local_objective(params):
                    points = params.reshape(16, 2)
                    distances = pdist(points)
                    if len(distances) == 0:
                        return 1e10
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist == 0:
                        return 1e10
                    return -min_dist / max_dist
                
                bounds = [(0, 1) for _ in range(32)]
                
                result = minimize(
                    local_objective,
                    current_points.flatten(),
                    method=method,
                    bounds=bounds,
                    options=options
                )
                
                if result.success:
                    optimized_points = result.x.reshape(16, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = -local_objective(result.x)
                    
                    if ratio > current_ratio:
                        current_ratio = ratio
                        current_points = optimized_points.copy()
                        
            except Exception:
                continue
        
        return current_points
    
    # Try multiple initialization approaches and pick the best
    initial_strategies = [
        generate_hexagonal_perturbed,
        generate_fibonacci_sphere,
        generate_random_spherical
    ]
    
    best_points = None
    best_ratio = -float('inf')
    
    for strategy in initial_strategies:
        try:
            points = strategy()
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
        except Exception:
            continue
    
    if best_points is None:
        best_points = generate_hexagonal_perturbed()
    
    # Apply global optimization with higher quality settings
    points = optimize_with_differential_evolution(best_points)
    
    # Apply fine-tuning with local optimization
    points = fine_tune_with_local_optimization(points)
    
    # Final refinement: Try one more round with a different random seed
    try:
        np.random.seed(12345)
        perturbed = points + np.random.normal(0, 0.001, points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        points = fine_tune_with_local_optimization(perturbed)
    except Exception:
        pass
    
    return points


# EVOLVE-BLOCK-END
