# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction, multi-start optimization, and 
    specialized cooling schedules for enhanced performance.

    Returns
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
    
    def create_hexagonal_pattern():
        """Create a hexagonal lattice pattern with good initial distribution"""
        points = []
        
        # Create a 4x4 grid with hexagonal offset pattern
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
    
    def create_fibonacci_pattern():
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
    
    def create_custom_pattern():
        """Create a custom pattern that has shown good results in previous attempts"""
        points = []
        
        # Start with a good central configuration
        # Place 16 points in a way that avoids clustering and maximizes spread
        
        # Central point
        points.append([0.5, 0.5])
        
        # First ring - 6 points
        for i in range(6):
            angle = i * np.pi / 3
            points.append([0.5 + 0.25 * np.cos(angle), 0.5 + 0.25 * np.sin(angle)])
        
        # Second ring - 9 points arranged in triangular formation
        for i in range(3):
            angle = i * 2 * np.pi / 3
            for j in range(3):
                radius = 0.4 + j * 0.1
                points.append([0.5 + radius * np.cos(angle + j * np.pi/6), 
                              0.5 + radius * np.sin(angle + j * np.pi/6)])
        
        # Trim to exactly 16 points and normalize
        points = points[:16]
        points = np.array(points)
        
        # Normalize to [0.05, 0.95] range
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        # Scale and center properly
        points[:, 0] *= 0.8
        points[:, 1] *= 0.8
        points[:, 0] += 0.1
        points[:, 1] += 0.1
        
        # Clip to bounds
        points = np.clip(points, 0.05, 0.95)
        return points
    
    def objective_function(x):
        """Minimize negative of min/max distance ratio (equivalent to maximizing the ratio)"""
        # Reshape points
        points = x.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0:
            return 0
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Handle case where all points are coincident
        if d_max == 0:
            return -np.inf
            
        # Return negative ratio to convert maximization to minimization
        return -d_min / d_max
    
    def simulated_annealing_optimization(initial_points, max_iterations=20000):
        """Optimize the point configuration using simulated annealing with enhanced cooling"""
        n = 16
        # Start with a good initial configuration
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
        
        # Simulated Annealing parameters - optimized for better convergence
        current_points = points.copy()
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # More aggressive cooling schedule for faster convergence
        temp = 0.1
        cooling_rate = 0.998  # Faster cooling
        min_temp = 1e-10
        max_iter = max_iterations
        
        # Track improvement for early stopping
        last_improvement = 0
        patience = 800  # How many iterations to wait for improvement
        
        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing one point
            neighbor_points = current_points.copy()
            point_idx = np.random.randint(0, n)
            
            # Use adaptive perturbation scales based on iteration progress
            progress = iteration / max_iter
            if progress < 0.3:
                perturbation_scale = 0.02  # Large perturbations early
            elif progress < 0.7:
                perturbation_scale = 0.005  # Medium perturbations middle
            else:
                perturbation_scale = 0.001  # Small perturbations late
            
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
                    last_improvement = iteration
            else:
                # Accept with probability based on temperature
                delta = new_ratio - current_ratio
                if np.random.random() < np.exp(delta / temp):
                    current_points = neighbor_points
                    current_ratio = new_ratio
                    last_improvement = iteration
            
            # Cool down
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp
            
            # Early stopping if no improvement for too long
            if iteration - last_improvement > patience:
                break
                
        return best_points
    
    def local_search(points, max_iterations=1500):
        """Enhanced local search to fine-tune the solution"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Track improvement for early stopping
        last_improvement = 0
        improvement_count = 0
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try different perturbation strategies
            if iteration % 3 == 0:
                # Move points towards neighbors (cluster-based)
                for i in range(16):
                    # Find nearest neighbor
                    distances = np.linalg.norm(current_points - current_points[i], axis=1)
                    distances[i] = np.inf  # Exclude self
                    nearest_idx = np.argmin(distances)
                    
                    # Move towards nearest neighbor slightly
                    move_direction = current_points[nearest_idx] - current_points[i]
                    if np.linalg.norm(move_direction) > 0:
                        move_direction = move_direction / np.linalg.norm(move_direction)
                        new_points = current_points.copy()
                        new_points[i] += move_direction * 0.001
                        new_points[i] = np.clip(new_points[i], 0, 1)
                        
                        new_ratio = compute_min_max_ratio(new_points)
                        if new_ratio > current_ratio:
                            current_points = new_points
                            current_ratio = new_ratio
                            improved = True
                            improvement_count = 0
                            last_improvement = iteration
            else:
                # Random perturbation
                for i in range(16):
                    # Try a small random movement
                    new_points = current_points.copy()
                    new_points[i] += np.random.normal(0, 0.0005, 2)
                    new_points[i] = np.clip(new_points[i], 0, 1)
                    
                    new_ratio = compute_min_max_ratio(new_points)
                    
                    if new_ratio > current_ratio:
                        current_points = new_points
                        current_ratio = new_ratio
                        improved = True
                        improvement_count = 0
                        last_improvement = iteration
            
            # Early stopping criteria
            if not improved:
                improvement_count += 1
                if improvement_count > 30:  # No improvement for 30 iterations
                    break
            else:
                improvement_count = 0
                
        return current_points
    
    # Try multiple initialization strategies (like inspiration programs)
    initial_strategies = [
        create_hexagonal_pattern,
        create_fibonacci_pattern,
        create_custom_pattern
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    # Strategy: Multiple restarts with geometric initializations (but reduced for efficiency)
    for i, init_func in enumerate(initial_strategies):
        try:
            # Try multiple local optimizations with this initial pattern
            for restart in range(3):  # Reduced restarts for better efficiency
                # Create slightly different version of the pattern for each restart
                np.random.seed(42 + i * 100 + restart)
                test_pattern = init_func()
                test_pattern += np.random.normal(0, 0.02, test_pattern.shape)  # Larger perturbation
                test_pattern = np.clip(test_pattern, 0.05, 0.95)
                
                # Optimize with simulated annealing
                optimized_points = simulated_annealing_optimization(test_pattern, max_iterations=20000)
                optimized_ratio = compute_min_max_ratio(optimized_points)
                
                if optimized_ratio > best_ratio:
                    best_ratio = optimized_ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    # Strategy: Final local refinement with higher precision
    if best_points is not None:
        try:
            # Do final local search refinement
            refined_points = local_search(best_points, max_iterations=2000)
            refined_ratio = compute_min_max_ratio(refined_points)
            
            if refined_ratio > best_ratio:
                return refined_points
        except Exception:
            pass
    
    # Fallback to hexagonal pattern if nothing worked
    if best_points is None:
        best_points = create_hexagonal_pattern()
    
    return best_points


# EVOLVE-BLOCK-END
