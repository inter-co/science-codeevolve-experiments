# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import math
import warnings
warnings.filterwarnings('ignore')

# Import for performance improvement
try:
    from numba import jit
except ImportError:
    def jit(func):
        return func

@jit(nopython=True)
def fast_compute_distances(points):
    """Fast computation of pairwise distances using Numba"""
    n = points.shape[0]
    distances = np.zeros((n*(n-1)//2,))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            distances[idx] = math.sqrt(dx*dx + dy*dy)
            idx += 1
    return distances

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
        # Use faster distance computation when available
        try:
            distances = fast_compute_distances(points)
        except:
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
    
    # Strategy 1: Optimized hexagonal packing with mathematical precision
    def hexagonal_packing_initialization():
        """Create points using optimized hexagonal packing with mathematical precision"""
        # Create a more precise hexagonal lattice arrangement
        points = []
        
        # Center point
        points.append([0.5, 0.5])
        
        # First ring: 6 points around center (hexagonal) with precise spacing
        radius = 0.35
        for i in range(6):
            angle = i * math.pi / 3
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        
        # Second ring: 9 points with better distribution
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
        perturbation = 0.01 * np.random.rand(16, 2)
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
        angle_step = 2 * math.pi / (phi * 1.3)  # Slightly adjusted for better spread
        
        # Create points along spiral with more even spacing
        center = np.array([0.5, 0.5])
        for i in range(16):
            angle = i * angle_step
            # Use a power law to create better distribution
            radius = 0.45 * (i / 15.0)**0.8  # Slight adjustment for better spacing
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points[i] = [x, y]
        
        # Add small random perturbations
        perturbation = 0.01 * np.random.rand(16, 2)
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
    
    # Strategy 4: Polar coordinate-based approach with optimized radial distribution
    def polar_coordinate_initialization():
        """Create points using polar coordinates with optimized radial distribution"""
        points = np.zeros((16, 2))
        np.random.seed(42)
        
        # Create points in a polar pattern with better radial distribution
        # Center point
        points[0] = [0.5, 0.5]
        
        # Radial layers with appropriate point counts
        radii = [0.15, 0.3, 0.45, 0.6]
        angles_per_layer = [4, 6, 4, 2]  # Number of points per layer
        
        idx = 1
        for r_idx, radius in enumerate(radii):
            num_points = angles_per_layer[r_idx]
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            for i in range(num_points):
                angle = angles[i]
                x = 0.5 + radius * math.cos(angle)
                y = 0.5 + radius * math.sin(angle)
                points[idx] = [x, y]
                idx += 1
                if idx >= 16:
                    break
            if idx >= 16:
                break
        
        # Fill remaining positions with random points if needed
        for i in range(idx, 16):
            points[i] = [0.1 + 0.8 * np.random.rand(), 0.1 + 0.8 * np.random.rand()]
        
        # Add small random perturbations
        perturbation = 0.01 * np.random.rand(16, 2)
        points += perturbation
        points = np.clip(points, 0, 1)
        return points
    
    # Enhanced optimization with better parameters and fallback
    def enhanced_optimize(initial_points, max_iter=6000):
        """Enhanced optimization with multiple attempts and better parameters"""
        initial_flat = initial_points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            # Try multiple optimization methods with different parameters
            methods_and_params = [
                ('L-BFGS-B', {'maxiter': max_iter, 'ftol': 1e-17, 'gtol': 1e-17}),
                ('TNC', {'maxiter': max_iter//2, 'ftol': 1e-15, 'gtol': 1e-15}),
                ('SLSQP', {'maxiter': max_iter//2, 'ftol': 1e-15, 'gtol': 1e-15}),
                ('trust-constr', {'maxiter': max_iter//2, 'ftol': 1e-15, 'gtol': 1e-15})
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
                        tol=1e-17
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
    
    # Global optimization using differential evolution for better exploration
    def global_optimization(initial_points):
        """Use differential evolution for global optimization"""
        def objective_wrapper(x_flat):
            points = x_flat.reshape(-1, 2)
            points = np.clip(points, 0, 1)
            ratio = compute_min_max_ratio(points)
            return -ratio  # Minimize negative to maximize ratio
        
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            # Use differential evolution for global search with reduced iterations
            result = differential_evolution(
                objective_wrapper,
                bounds,
                maxiter=100,
                popsize=10,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False,
                atol=1e-15,
                rtol=1e-15
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                return optimized_points
        except Exception:
            pass
        
        return initial_points
    
    # Multi-start approach with focused strategies
    best_ratio = -np.inf
    best_points = None
    
    # Try fewer, high-quality initialization strategies
    strategies = [
        hexagonal_packing_initialization,
        golden_spiral_initialization,
        concentric_rings_initialization,
        polar_coordinate_initialization
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
    
    # Additional random restarts with moderate iteration counts
    for seed in [100, 200, 300, 400, 500, 600]:
        try:
            np.random.seed(seed)
            # Generate random points in unit square
            random_points = np.random.rand(16, 2)
            
            # Optimize with moderate iteration count for efficiency
            optimized_random = enhanced_optimize(random_points, max_iter=2000)
            
            # Evaluate the result
            random_ratio = compute_min_max_ratio(optimized_random)
            
            if random_ratio > best_ratio:
                best_ratio = random_ratio
                best_points = optimized_random.copy()
                
        except Exception:
            continue  # Skip failed optimizations
    
    # Final refinement with global optimization using differential evolution
    if best_points is not None:
        try:
            # Try global optimization as final step to potentially escape local optima
            global_optimized = global_optimization(best_points)
            global_ratio = compute_min_max_ratio(global_optimized)
            
            if global_ratio > best_ratio:
                return global_optimized
        except Exception:
            pass
    
    # Final refinement with more aggressive optimization
    if best_points is not None:
        try:
            final_optimized = enhanced_optimize(best_points, max_iter=6000)
            final_ratio = compute_min_max_ratio(final_optimized)
            
            if final_ratio > best_ratio:
                return final_optimized
        except Exception:
            pass
    
    # If no optimization worked, return the best initialization directly
    if best_points is None:
        # Use hexagonal initialization as fallback (often most effective)
        fallback_points = hexagonal_packing_initialization()
        return np.clip(fallback_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
