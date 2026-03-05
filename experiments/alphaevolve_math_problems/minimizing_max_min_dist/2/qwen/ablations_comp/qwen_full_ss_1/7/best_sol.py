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
    
    # Strategy 1: Optimized hexagonal packing with better spacing
    def hexagonal_packing_initialization():
        """Create points using optimized hexagonal packing with mathematical precision"""
        # Create a more precise hexagonal lattice arrangement
        points = []
        
        # Create points in a hexagonal pattern with more mathematical precision
        # Center point
        points.append([0.5, 0.5])
        
        # First ring: 6 points around center (hexagonal)
        radius = 0.35
        for i in range(6):
            angle = i * math.pi / 3
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        
        # Second ring: 9 points (adjusting for 16 total)
        radius = 0.65
        for i in range(9):
            angle = i * 2 * math.pi / 9
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
            
        # Trim to exactly 16 points
        points_array = np.array(points[:16])
        
        # Add small random perturbations to break symmetry
        np.random.seed(42)
        perturbation = 0.015 * np.random.rand(16, 2)
        points_array += perturbation
        points_array = np.clip(points_array, 0, 1)
        return points_array
    
    # Strategy 2: Golden ratio based spiral with better distribution
    def golden_spiral_initialization():
        """Create points using golden spiral distribution with better spread"""
        points = np.zeros((16, 2))
        np.random.seed(42)
        
        # Golden ratio spiral parameters with improved distribution
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        angle_step = 2 * math.pi / (phi * 1.3)
        
        # Create points along spiral with more even spacing
        center = np.array([0.5, 0.5])
        for i in range(16):
            angle = i * angle_step
            # Use a power law to create better distribution
            radius = 0.45 * (i / 15.0)**0.8
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points[i] = [x, y]
        
        # Add small random perturbations
        perturbation = 0.015 * np.random.rand(16, 2)
        points += perturbation
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 3: Concentric rings with optimal spacing and mathematical balance
    def concentric_rings_initialization():
        """Create points in concentric rings with mathematical spacing"""
        points = np.zeros((16, 2))
        np.random.seed(42)
        
        # Three concentric rings for better balance
        # Outer ring (6 points)
        outer_radius = 0.45
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i in range(6):
            points[i] = [0.5 + outer_radius * np.cos(angles[i]), 
                        0.5 + outer_radius * np.sin(angles[i])]
        
        # Middle ring (6 points)
        middle_radius = 0.3
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i in range(6):
            points[6+i] = [0.5 + middle_radius * np.cos(angles[i]), 
                          0.5 + middle_radius * np.sin(angles[i])]
        
        # Inner ring (4 points)
        inner_radius = 0.15
        angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        for i in range(4):
            points[12+i] = [0.5 + inner_radius * np.cos(angles[i]), 
                           0.5 + inner_radius * np.sin(angles[i])]
        
        # Add small random perturbations
        perturbation = 0.01 * np.random.rand(16, 2)
        points += perturbation
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 4: Square grid with adaptive perturbations plus rotation
    def grid_adaptive_initialization():
        """Create points on a grid with adaptive random perturbations and rotation"""
        # Create 4x4 grid
        grid_points = np.zeros((16, 2))
        for i in range(4):
            for j in range(4):
                grid_points[i*4 + j] = [i/3, j/3]
        
        # Add adaptive perturbations with rotation for better distribution
        np.random.seed(42)
        perturbation = np.zeros((16, 2))
        for i in range(16):
            # Adaptive scaling based on position
            row, col = i // 4, i % 4
            scale = 0.025 * (1 - abs(row - 1.5)/1.5) * (1 - abs(col - 1.5)/1.5)
            perturbation[i] = scale * np.random.rand(2)
        
        # Apply a small rotation to break symmetry
        rotation_angle = 0.1
        cos_a, sin_a = math.cos(rotation_angle), math.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Apply rotation to perturbations
        rotated_perturbations = np.dot(perturbation, rotation_matrix.T)
        grid_points += rotated_perturbations
        grid_points = np.clip(grid_points, 0, 1)
        return grid_points
    
    # Strategy 5: Mathematical optimization-inspired configuration with better distribution
    def mathematical_optimization_initialization():
        """Create points based on known mathematical optimization principles with improved layout"""
        # Create points that form a balanced distribution
        points = []
        
        # Corner points (4 points)
        for i in range(4):
            x = 0.1 + 0.8 * (i % 2)
            y = 0.1 + 0.8 * (i // 2)
            points.append([x, y])
        
        # Edge points (8 points) - distributed evenly
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
        
        # Add small random perturbations
        np.random.seed(42)
        perturbation = 0.01 * np.random.rand(16, 2)
        points_array += perturbation
        points_array = np.clip(points_array, 0, 1)
        return points_array
    
    # Strategy 6: Spherical-like distribution with improved mathematical foundation
    def spherical_distribution_initialization():
        """Create points distributed like on a sphere projection"""
        points = np.zeros((16, 2))
        np.random.seed(42)
        
        # Distribute points more evenly using Fibonacci-like distribution
        golden_ratio = (1 + math.sqrt(5)) / 2
        for i in range(16):
            # Use Fibonacci-like spacing
            theta = math.acos(-1 + (2 * i) / 15.0)
            phi = math.sqrt(16 * math.pi) * i / golden_ratio
            
            # Convert to Cartesian coordinates on a sphere, then project to 2D
            x = 0.5 + 0.4 * math.sin(theta) * math.cos(phi)
            y = 0.5 + 0.4 * math.sin(theta) * math.sin(phi)
            
            points[i] = [x, y]
        
        # Add small random perturbations
        perturbation = 0.015 * np.random.rand(16, 2)
        points += perturbation
        points = np.clip(points, 0, 1)
        return points
    
    # Enhanced optimization with better parameters and fallback
    def enhanced_optimize(initial_points, max_iter=5000):
        """Enhanced optimization with multiple attempts and better parameters"""
        initial_flat = initial_points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            # Try multiple optimization methods with different parameters
            methods_and_params = [
                ('L-BFGS-B', {'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-16}),
                ('TNC', {'maxiter': max_iter//2, 'ftol': 1e-14, 'gtol': 1e-14}),
                ('SLSQP', {'maxiter': max_iter//2, 'ftol': 1e-14, 'gtol': 1e-14})
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
                        tol=1e-16
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
        grid_adaptive_initialization,
        mathematical_optimization_initialization,
        spherical_distribution_initialization
    ]
    
    # Run optimization from each initialization strategy
    for i, strategy_func in enumerate(strategies):
        try:
            # Generate initial points
            initial_points = strategy_func()
            
            # Enhance optimization with higher iterations for better convergence
            optimized_points = enhanced_optimize(initial_points, max_iter=4000)
            
            # Evaluate the result
            ratio = compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue  # Skip failed optimizations
    
    # Additional random restarts with very high iteration counts
    for seed in [100, 200, 300, 400, 500, 600, 700]:
        try:
            np.random.seed(seed)
            # Generate random points in unit square
            random_points = np.random.rand(16, 2)
            
            # Optimize with very high iteration count for thorough search
            optimized_random = enhanced_optimize(random_points, max_iter=5000)
            
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
