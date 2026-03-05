# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import math
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction, multiple initialization strategies,
    and advanced optimization methods.

    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all pairs"""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0
    
    def objective(x_flat):
        """Objective function to maximize min/max distance ratio (negative for minimization)"""
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0 or np.max(distances) <= 1e-10:
            return 1e10  # Penalty for invalid configurations
            
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 1e-10:
            return 1e10
            
        # Return negative ratio to maximize (minimize negative)
        return -d_min / d_max
    
    def create_hexagonal_lattice():
        """Create a hexagonal lattice pattern - proven to work well"""
        points = []
        
        # Create a 4x4 grid with alternating row offsets for hexagonal packing
        for i in range(4):
            for j in range(4):
                # Offset every other row for hexagonal pattern
                x_offset = 0.25 if i % 2 == 1 else 0.0
                x = (j * 0.25) + x_offset + 0.125  # Center in [0.1, 0.9] range
                y = i * 0.25 + 0.125  # Center in [0.1, 0.9] range
                
                points.append([x, y])
        
        # Ensure exactly 16 points and clip to bounds [0.05, 0.95]
        points = np.array(points[:16])
        points = np.clip(points, 0.05, 0.95)
        return points
    
    def create_regular_grid():
        """Create a regular 4x4 grid pattern"""
        points = []
        for i in range(4):
            for j in range(4):
                x = 0.1 + j * 0.225
                y = 0.1 + i * 0.225
                points.append([x, y])
        return np.array(points[:16])
    
    def create_fibonacci_spiral():
        """Create points using Fibonacci spiral for good distribution"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(16):
            # Golden angle spiral
            theta = i * 2 * np.pi / golden_ratio
            radius = np.sqrt(i / 15) if i > 0 else 0
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y])
        
        # Normalize and scale to unit square
        points = np.array(points)
        if len(points) > 0:
            # Center around origin
            points -= np.mean(points, axis=0)
            # Scale to fit nicely
            max_coord = np.max(np.abs(points))
            if max_coord > 0:
                points /= max_coord
            # Scale to [0.1, 0.9] range to avoid boundary issues
            points = points * 0.4 + 0.5
        return points
    
    def create_concentric_rings():
        """Create concentric ring pattern"""
        points = []
        # Center point
        points.append([0.5, 0.5])
        # First ring: 6 points
        for i in range(6):
            angle = i * np.pi / 3
            points.append([0.5 + 0.25 * np.cos(angle), 0.5 + 0.25 * np.sin(angle)])
        # Second ring: 9 points
        for i in range(9):
            angle = i * 2 * np.pi / 9
            points.append([0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)])
        return np.array(points[:16])
    
    def create_von_neumann_pattern():
        """Create points using a von Neumann-like grid pattern"""
        # Create a 4x4 grid with some additional points
        points = []
        for i in range(4):
            for j in range(4):
                points.append([j/3.0, i/3.0])
        
        # Add some extra points for better coverage
        extra_points = [
            [0.25, 0.75],
            [0.75, 0.25],
            [0.25, 0.25],
            [0.75, 0.75]
        ]
        
        points.extend(extra_points)
        points = points[:16]
        points = np.array(points)
        
        # Random perturbation to break symmetry
        np.random.seed(42)
        points += np.random.normal(0, 0.02, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def simulated_annealing_optimization(initial_points, max_iterations=25000):
        """Optimize using simulated annealing with improved cooling schedule"""
        n = 16
        points = initial_points.copy()
        
        def calculate_ratio(points_array):
            """Calculate min/max distance ratio"""
            distances = pdist(points_array)
            if len(distances) == 0:
                return 0
            d_min = np.min(distances)
            d_max = np.max(distances)
            if d_max <= 0:
                return 0
            return d_min / d_max
        
        # Initialize optimization parameters
        current_points = points.copy()
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # More aggressive cooling schedule (from inspiration 1)
        temp = 0.1
        cooling_rate = 0.9985  # Even more aggressive cooling
        min_temp = 1e-12
        max_iter = max_iterations
        
        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing one point
            neighbor_points = current_points.copy()
            point_idx = np.random.randint(0, n)
            
            # Adaptive perturbation scale based on temperature (more aggressive)
            perturbation_scale = max(0.002, temp * 0.07)
            neighbor_points[point_idx] += np.random.normal(0, perturbation_scale, 2)
            
            # Keep within bounds [0,1]²
            neighbor_points[point_idx] = np.clip(neighbor_points[point_idx], 0, 1)
            
            # Calculate new ratio
            new_ratio = calculate_ratio(neighbor_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                current_points = neighbor_points
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = new_ratio
            else:
                # Accept with probability based on temperature
                delta = new_ratio - current_ratio
                if np.random.random() < np.exp(delta / temp):
                    current_points = neighbor_points
                    current_ratio = new_ratio
            
            # Cool down
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp
                
        return best_points
    
    def local_search_refinement(points, max_iterations=1000):
        """Refine with local search - more aggressive than before"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Track improvements to stop early
        improvement_count = 0
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try moving each point slightly in a random direction
            for i in range(16):
                # Try a small random movement
                new_points = current_points.copy()
                new_points[i] += np.random.normal(0, 0.0006, 2)
                new_points[i] = np.clip(new_points[i], 0, 1)
                
                new_ratio = compute_min_max_ratio(new_points)
                
                if new_ratio > current_ratio:
                    current_points = new_points
                    current_ratio = new_ratio
                    improved = True
                    improvement_count = 0
                else:
                    # Restore original point
                    current_points[i] = current_points[i]
            
            # Early stopping criteria
            if not improved:
                improvement_count += 1
                if improvement_count > 70:  # No improvement for 70 iterations
                    break
                    
        return current_points
    
    # Try multiple initialization strategies with optimization
    initial_strategies = [
        create_hexagonal_lattice,
        create_regular_grid,
        create_fibonacci_spiral,
        create_concentric_rings,
        create_von_neumann_pattern
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    # Run optimizations from multiple starting points
    for i, init_func in enumerate(initial_strategies):
        try:
            # Get initial points
            points = init_func()
            
            # Add small random perturbations to break symmetry
            np.random.seed(42 + i * 100)  # Different seed per strategy
            noise = np.random.normal(0, 0.01, points.shape)
            points += noise
            points = np.clip(points, 0, 1)
            
            # Try simulated annealing optimization with more iterations
            optimized_points = simulated_annealing_optimization(points, max_iterations=25000)
            optimized_ratio = compute_min_max_ratio(optimized_points)
            
            if optimized_ratio > best_ratio:
                best_ratio = optimized_ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    # Final refinement with both local search and gradient-based optimization
    if best_points is not None:
        # First local search refinement
        final_points = local_search_refinement(best_points)
        final_ratio = compute_min_max_ratio(final_points)
        
        # Then try gradient-based optimization for further improvement
        try:
            # Use the best configuration found so far as starting point
            final_params = final_points.flatten()
            
            # Define bounds for each coordinate (0 to 1)
            bounds = [(0, 1) for _ in range(32)]
            
            # Optimize with L-BFGS-B for high precision
            result = minimize(
                objective,
                final_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                gradient_optimized_points = result.x.reshape(-1, 2)
                gradient_ratio = compute_min_max_ratio(gradient_optimized_points)
                
                if gradient_ratio > final_ratio:
                    final_points = gradient_optimized_points
                    final_ratio = gradient_ratio
                    
        except Exception:
            pass
        
        return final_points
    else:
        # Fallback to hexagonal lattice
        points = create_hexagonal_lattice()
        return points


# EVOLVE-BLOCK-END
