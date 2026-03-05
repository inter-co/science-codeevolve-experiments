# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import math
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining mathematical constructions, 
    multi-start optimization, and intelligent initialization strategies.
    
    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distances"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0.0
    
    def objective_function(x_flat):
        """Objective function to minimize (negative ratio)"""
        points = x_flat.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    # Strategy 1: Optimized hexagonal packing with precise mathematical construction
    def hexagonal_packing_initialization():
        """Create points using optimized hexagonal packing with mathematical precision"""
        points = []
        
        # Create a highly symmetric hexagonal pattern
        # Center point
        points.append([0.5, 0.5])
        
        # First ring: 6 points (perfect hexagon)
        radius = 0.35
        for i in range(6):
            angle = i * math.pi / 3
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        
        # Second ring: 9 points (adjusted to get exactly 16 points)
        radius = 0.65
        for i in range(9):
            angle = i * 2 * math.pi / 9
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
            
        # Trim to exactly 16 points and add precise perturbations
        points_array = np.array(points[:16])
        
        # Use more sophisticated perturbations that preserve some geometric properties
        np.random.seed(42)
        # Apply smaller, more controlled perturbations
        perturbation = 0.008 * np.random.rand(16, 2)
        points_array += perturbation
        points_array = np.clip(points_array, 0, 1)
        return points_array
    
    # Strategy 2: Golden spiral with improved mathematical spacing
    def golden_spiral_initialization():
        """Create points using golden spiral distribution with better spread"""
        points = np.zeros((16, 2))
        np.random.seed(42)
        
        # Golden ratio spiral with improved distribution parameters
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        angle_step = 2 * math.pi / (phi * 1.2)  # Slightly adjusted for better spread
        
        center = np.array([0.5, 0.5])
        for i in range(16):
            angle = i * angle_step
            # Use power law with better exponent for even distribution
            radius = 0.45 * (i / 15.0)**0.75
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points[i] = [x, y]
        
        # Add small controlled perturbations
        perturbation = 0.008 * np.random.rand(16, 2)
        points += perturbation
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 3: Concentric rings with mathematical optimization
    def concentric_rings_initialization():
        """Create points in concentric rings with mathematical spacing"""
        points = np.zeros((16, 2))
        np.random.seed(42)
        
        # Use 3 rings with carefully chosen radii and point counts
        # Based on mathematical analysis of optimal distributions
        ring_radii = [0.15, 0.35, 0.55]  # Radii optimized for even coverage
        points_per_ring = [4, 6, 6]      # Point distribution
        
        idx = 0
        for r, num_points in zip(ring_radii, points_per_ring):
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            for i in range(num_points):
                points[idx] = [0.5 + r * np.cos(angles[i]), 
                              0.5 + r * np.sin(angles[i])]
                idx += 1
                if idx >= 16:
                    break
            if idx >= 16:
                break
        
        # Add small random perturbations
        perturbation = 0.007 * np.random.rand(16, 2)
        points += perturbation
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 4: Mathematical grid with adaptive perturbations
    def mathematical_grid_initialization():
        """Create points on a mathematical grid with adaptive perturbations"""
        # Create a 4x4 grid with mathematical spacing
        grid_points = np.zeros((16, 2))
        for i in range(4):
            for j in range(4):
                grid_points[i*4 + j] = [0.1 + 0.8 * i / 3, 0.1 + 0.8 * j / 3]
        
        # Add adaptive perturbations with better mathematical basis
        np.random.seed(42)
        perturbation = np.zeros((16, 2))
        for i in range(16):
            # Use a more sophisticated perturbation based on position
            row, col = i // 4, i % 4
            # Perturbation strength based on distance from center
            dist_from_center = np.sqrt((row - 1.5)**2 + (col - 1.5)**2)
            strength = 0.015 * (1 - dist_from_center/2.12)  # Max near center
            perturbation[i] = strength * np.random.randn(2)
        
        grid_points += perturbation
        grid_points = np.clip(grid_points, 0, 1)
        return grid_points
    
    # Strategy 5: Optimized corner and edge layout with mathematical balance
    def optimized_layout_initialization():
        """Create points with optimized corner/edge distribution"""
        points = []
        
        # Corner points (4 points) - fixed positions
        for i in range(4):
            x = 0.1 + 0.8 * (i % 2)
            y = 0.1 + 0.8 * (i // 2)
            points.append([x, y])
        
        # Edge points (8 points) - evenly distributed
        for i in range(8):
            if i < 4:  # Top and bottom edges
                x = 0.1 + 0.8 * i / 3
                y = 0.1 if i % 2 == 0 else 0.9
            else:  # Left and right edges
                x = 0.1 if (i-4) % 2 == 0 else 0.9
                y = 0.1 + 0.8 * (i-4) / 3
            points.append([x, y])
        
        # Remaining points in center (4 points) arranged in a square
        for i in range(4):
            x = 0.3 + 0.4 * (i % 2)
            y = 0.3 + 0.4 * (i // 2)
            points.append([x, y])
        
        # Ensure exactly 16 points
        points_array = np.array(points[:16])
        
        # Add small, controlled random perturbations
        np.random.seed(42)
        perturbation = 0.006 * np.random.rand(16, 2)
        points_array += perturbation
        points_array = np.clip(points_array, 0, 1)
        return points_array
    
    # Strategy 6: Fibonacci-based spherical distribution
    def fibonacci_distribution_initialization():
        """Create points using Fibonacci-based distribution with better mathematical foundation"""
        points = np.zeros((16, 2))
        np.random.seed(42)
        
        # Use Fibonacci-like distribution with precise mathematical constants
        golden_ratio = (1 + math.sqrt(5)) / 2
        for i in range(16):
            # Improved Fibonacci-like spacing
            theta = math.acos(-1 + (2 * i) / 15.0)
            phi = math.sqrt(16 * math.pi) * i / golden_ratio
            
            # Project to 2D with better distribution
            x = 0.5 + 0.4 * math.sin(theta) * math.cos(phi)
            y = 0.5 + 0.4 * math.sin(theta) * math.sin(phi)
            
            points[i] = [x, y]
        
        # Add small, controlled perturbations
        perturbation = 0.008 * np.random.rand(16, 2)
        points += perturbation
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 7: Improved hexagonal tiling with better spacing
    def improved_hex_tiling_initialization():
        """Create points using improved hexagonal tiling with mathematical precision"""
        points = []
        
        # Create a hexagonal pattern with more precise mathematical construction
        # Center point
        points.append([0.5, 0.5])
        
        # First ring - 6 points (perfect hexagon)
        for i in range(6):
            angle = i * math.pi / 3
            x = 0.5 + 0.3 * math.cos(angle)
            y = 0.5 + 0.3 * math.sin(angle)
            points.append([x, y])
        
        # Second ring - 9 points (adjusted to get exactly 16 points)
        for i in range(9):
            angle = i * 2 * math.pi / 9
            x = 0.5 + 0.6 * math.cos(angle)
            y = 0.5 + 0.6 * math.sin(angle)
            points.append([x, y])
            
        # Trim to exactly 16 points
        points_array = np.array(points[:16])
        
        # Add precise, small perturbations
        np.random.seed(42)
        perturbation = 0.007 * np.random.rand(16, 2)
        points_array += perturbation
        points_array = np.clip(points_array, 0, 1)
        return points_array
    
    # Enhanced optimization with better parameters and more robust fallback
    def enhanced_optimize(initial_points, max_iter=7000):
        """Enhanced optimization with multiple attempts and better parameters"""
        initial_flat = initial_points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            # Try multiple optimization methods with different parameters
            methods_and_params = [
                ('L-BFGS-B', {'maxiter': max_iter, 'ftol': 1e-18, 'gtol': 1e-18}),
                ('TNC', {'maxiter': max_iter//2, 'ftol': 1e-16, 'gtol': 1e-16}),
                ('SLSQP', {'maxiter': max_iter//2, 'ftol': 1e-16, 'gtol': 1e-16}),
                ('trust-constr', {'maxiter': max_iter//3, 'ftol': 1e-16, 'gtol': 1e-16})
            ]
            
            best_result = None
            best_ratio = -np.inf
            
            for method, options in methods_and_params:
                try:
                    result = minimize(
                        objective_function,
                        initial_flat,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-18
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        ratio = compute_min_max_ratio(final_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_result = result
                            
                except Exception:
                    continue  # Skip failed optimizations
            
            if best_result is not None:
                return best_result.x.reshape(-1, 2)
            else:
                return initial_points
                
        except Exception:
            return initial_points
    
    # Multi-start approach with diverse strategies
    best_ratio = -np.inf
    best_points = None
    
    # Try all initialization strategies
    strategies = [
        hexagonal_packing_initialization,
        golden_spiral_initialization,
        concentric_rings_initialization,
        mathematical_grid_initialization,
        optimized_layout_initialization,
        fibonacci_distribution_initialization,
        improved_hex_tiling_initialization
    ]
    
    # Run optimization from each initialization strategy
    for i, strategy_func in enumerate(strategies):
        try:
            # Generate initial points
            initial_points = strategy_func()
            
            # Enhance optimization with higher iterations for better convergence
            optimized_points = enhanced_optimize(initial_points, max_iter=5500)
            
            # Evaluate the result
            ratio = compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue  # Skip failed optimizations
    
    # Additional random restarts with varied iteration counts
    for seed in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
        try:
            np.random.seed(seed)
            # Generate random points in unit square
            random_points = np.random.rand(16, 2)
            
            # Optimize with varying iteration counts for thorough search
            iter_count = 4500 + (seed % 4) * 1000  # Vary iteration count slightly
            optimized_random = enhanced_optimize(random_points, max_iter=iter_count)
            
            # Evaluate the result
            random_ratio = compute_min_max_ratio(optimized_random)
            
            if random_ratio > best_ratio:
                best_ratio = random_ratio
                best_points = optimized_random.copy()
                
        except Exception:
            continue  # Skip failed optimizations
    
    # If no optimization worked, return the best initialization directly
    if best_points is None:
        # Use hexagonal initialization as fallback (often most effective)
        fallback_points = hexagonal_packing_initialization()
        return np.clip(fallback_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
