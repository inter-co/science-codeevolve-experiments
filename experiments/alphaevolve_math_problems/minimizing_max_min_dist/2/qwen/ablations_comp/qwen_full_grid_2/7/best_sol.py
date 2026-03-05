# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction and simulated annealing with enhanced strategies.
    
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
        """Create a hexagonal lattice pattern that often provides good starting points"""
        points = []
        rows = 4
        cols = 4
        
        # Hexagonal offset pattern
        for i in range(rows):
            for j in range(cols):
                # Base position
                x = j * 0.25 + (i % 2) * 0.125  # Offset every other row
                y = i * 0.25
                    
                # Add small random perturbation to avoid degeneracy
                x += np.random.normal(0, 0.015) * (0.25 if i < 3 else 0.1)
                y += np.random.normal(0, 0.015) * 0.25
                
                # Keep within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                
                points.append([x, y])
        
        return np.array(points)
    
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
    
    def create_fibonacci_pattern():
        """Create a Fibonacci spiral pattern for good distribution"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for i in range(16):
            # Fibonacci spiral approach
            theta = i * 2 * np.pi / phi
            r = np.sqrt(i / 15.0) * 0.4  # Scale to fit in unit square
            
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            
            # Add small random perturbation
            x += np.random.normal(0, 0.015) * 0.1
            y += np.random.normal(0, 0.015) * 0.1
            
            # Keep within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            points.append([x, y])
        
        return np.array(points)
    
    def create_grid_pattern():
        """Create a simple grid pattern"""
        points = []
        grid_size = 4
        x = np.linspace(0.1, 0.9, grid_size)
        y = np.linspace(0.1, 0.9, grid_size)
        
        for i in range(grid_size):
            for j in range(grid_size):
                x_pos = x[j]
                y_pos = y[i]
                points.append([x_pos, y_pos])
        
        return np.array(points[:16])
    
    def create_spiral_pattern():
        """Create a spiral pattern that distributes points well"""
        points = []
        # Create a spiral pattern with 4 arms
        n_points = 16
        arm_count = 4
        
        for arm in range(arm_count):
            for i in range(n_points // arm_count):
                angle = arm * np.pi / 2 + i * 0.3
                radius = 0.1 + i * 0.15
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                
                # Add small random perturbation
                x += np.random.normal(0, 0.01) * 0.1
                y += np.random.normal(0, 0.01) * 0.1
                
                # Keep within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                
                points.append([x, y])
                
        return np.array(points[:16])
    
    def simulated_annealing_optimization(initial_points, max_iterations=20000):
        """Optimize the point configuration using simulated annealing with proper cooling"""
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
        cooling_rate = 0.998  # Faster cooling from INSPIRATION 3
        min_temp = 1e-10
        max_iter = max_iterations
        
        # Track improvement for early stopping
        last_improvement = 0
        patience = 1000  # How many iterations to wait for improvement
        
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
                
        return best_points, best_ratio
    
    def local_search(points, max_iterations=2000):
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
                if improvement_count > 50:  # No improvement for 50 iterations
                    break
            else:
                improvement_count = 0
                
        return current_points
    
    # Try multiple geometric constructions as starting points
    initial_patterns = [
        create_hexagonal_pattern(),
        create_symmetric_pattern(), 
        create_fibonacci_pattern(),
        create_grid_pattern(),
        create_spiral_pattern()
    ]
    
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Multiple restarts with geometric initializations
    for i, initial_pattern in enumerate(initial_patterns):
        try:
            # Try multiple local optimizations with this initial pattern
            for restart in range(5):  # More restarts for better exploration (from INSPIRATION 3)
                # Create slightly different version of the pattern for each restart
                np.random.seed(42 + i * 100 + restart)
                test_pattern = initial_pattern.copy()
                test_pattern += np.random.normal(0, 0.02, test_pattern.shape)  # Larger perturbation
                test_pattern = np.clip(test_pattern, 0.05, 0.95)
                
                # Optimize with simulated annealing
                optimized_points, ratio = simulated_annealing_optimization(test_pattern, max_iterations=20000)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    # Strategy 2: Additional refinement with local search on best solution
    if best_points is not None:
        try:
            # Do final local search refinement
            refined_points = local_search(best_points, max_iterations=2000)
            refined_ratio = compute_min_max_ratio(refined_points)
            
            if refined_ratio > best_ratio:
                best_points = refined_points
                best_ratio = refined_ratio
        except Exception:
            pass
    
    # Fallback to hexagonal pattern if nothing worked
    if best_points is None:
        best_points = create_hexagonal_pattern()
    
    return best_points


# EVOLVE-BLOCK-END
