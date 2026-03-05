# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, multiple optimization methods,
    and simulated annealing for robust convergence.

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
        """Objective function to minimize (negative of min/max ratio)"""
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0 or np.max(distances) <= 1e-10:
            return 1e10  # Large penalty for invalid configurations
            
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
        # Create a carefully optimized hexagonal arrangement
        points = []
        
        # Center point
        points.append([0.5, 0.5])
        
        # First ring of 6 points (radius 0.3)
        for i in range(6):
            angle = i * np.pi / 3
            points.append([0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)])
        
        # Second ring of 9 points (radius 0.6) in triangular arrangement
        for i in range(3):
            angle = i * 2 * np.pi / 3
            for j in range(3):
                radius = 0.6 + j * 0.15
                points.append([0.5 + radius * np.cos(angle + j * np.pi/6), 
                              0.5 + radius * np.sin(angle + j * np.pi/6)])
        
        # Trim to exactly 16 points and normalize
        points = points[:16]
        points = np.array(points)
        
        # Normalize to [0,1]² properly
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
    
    def create_alternating_grid():
        """Create an alternating grid pattern that works well for spacing"""
        points = []
        for i in range(4):
            for j in range(4):
                # Alternate offset for better spacing
                x = 0.1 + j * 0.225 + (i % 2) * 0.1125
                y = 0.1 + i * 0.225
                points.append([x, y])
        return np.array(points[:16])
    
    def create_custom_distribution():
        """Create a custom distribution pattern that balances uniformity and spacing"""
        # Start with a regular grid and adjust positions
        points = []
        for i in range(4):
            for j in range(4):
                points.append([j/3.0, i/3.0])
        
        points = np.array(points[:16])
        
        # Apply slight random adjustments to improve distribution
        np.random.seed(12345)  # Fixed seed for consistency
        adjustments = np.random.normal(0, 0.015, points.shape)
        points += adjustments
        points = np.clip(points, 0.05, 0.95)
        return points
    
    def create_better_hexagonal_pattern():
        """Create a more refined hexagonal pattern with better spacing"""
        # Inspired by hexagonal close packing but adapted for 16 points
        points = []
        
        # Place points in a hexagonal arrangement
        # 4 rows with 4 points each (but with offset rows)
        for i in range(4):
            for j in range(4):
                # Offset even rows
                offset = 0.25 if i % 2 == 1 else 0.0
                x = 0.1 + j * 0.25 + offset
                y = 0.1 + i * 0.25
                
                # Make sure we stay within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                
                points.append([x, y])
        
        # Ensure exactly 16 points
        points = np.array(points[:16])
        return points
    
    def create_symmetric_pattern():
        """Create a symmetric pattern inspired by optimal configurations"""
        points = []
        
        # Central cluster with radial symmetry
        center_x, center_y = 0.5, 0.5
        
        # Place points around center in concentric rings
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = [0.15, 0.25, 0.35, 0.45]  # Different ring radii
        
        # Distribute points in rings
        ring_idx = 0
        for i in range(16):
            angle = angles[i]
            radius = radii[ring_idx]
            
            # Add some randomness to avoid perfect regularity
            radius += np.random.normal(0, 0.015) * radius
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Keep within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            points.append([x, y])
            ring_idx = (ring_idx + 1) % len(radii)
        
        return np.array(points)
    
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
        
        # Aggressive cooling schedule for faster convergence
        temp = 0.1
        cooling_rate = 0.9985
        min_temp = 1e-12
        max_iter = max_iterations
        
        # Track improvement for early stopping
        last_improvement = 0
        patience = 1000
        
        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing one point
            neighbor_points = current_points.copy()
            point_idx = np.random.randint(0, n)
            
            # Adaptive perturbation scale based on temperature and iteration progress
            progress = iteration / max_iter
            if progress < 0.2:
                perturbation_scale = 0.03  # Large perturbations early
            elif progress < 0.5:
                perturbation_scale = 0.01  # Medium perturbations middle
            else:
                perturbation_scale = 0.002  # Small perturbations late
            
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
    
    def enhanced_local_search(points, max_iterations=2000):
        """Enhanced local search with more aggressive refinement"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Track improvements to stop early
        improvement_count = 0
        last_improvement = 0
        
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
                        new_points[i] += move_direction * 0.0015
                        new_points[i] = np.clip(new_points[i], 0, 1)
                        
                        new_ratio = compute_min_max_ratio(new_points)
                        if new_ratio > current_ratio:
                            current_points = new_points
                            current_ratio = new_ratio
                            improved = True
                            improvement_count = 0
                            last_improvement = iteration
            else:
                # Random perturbation with larger step size for better exploration
                for i in range(16):
                    # Try a small random movement
                    new_points = current_points.copy()
                    # Slightly larger perturbation to explore more
                    new_points[i] += np.random.normal(0, 0.0008, 2)
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
                if improvement_count > 50:  # No improvement for 50 iterations
                    break
            else:
                improvement_count = 0
                
        return current_points
    
    # Try multiple initialization strategies with optimization
    initial_strategies = [
        create_hexagonal_lattice,
        create_regular_grid,
        create_fibonacci_spiral,
        create_concentric_rings,
        create_von_neumann_pattern,
        create_alternating_grid,
        create_custom_distribution,
        create_better_hexagonal_pattern,
        create_symmetric_pattern
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    # Run optimizations from multiple starting points with more thorough exploration
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
            optimized_points = simulated_annealing_optimization(points, max_iterations=20000)
            optimized_ratio = compute_min_max_ratio(optimized_points)
            
            if optimized_ratio > best_ratio:
                best_ratio = optimized_ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    # Final refinement with enhanced local search
    if best_points is not None:
        final_points = enhanced_local_search(best_points)
        final_ratio = compute_min_max_ratio(final_points)
        
        # Return the best configuration found
        return final_points if final_ratio > best_ratio else best_points
    else:
        # Fallback to hexagonal lattice
        points = create_hexagonal_lattice()
        return points


# EVOLVE-BLOCK-END
