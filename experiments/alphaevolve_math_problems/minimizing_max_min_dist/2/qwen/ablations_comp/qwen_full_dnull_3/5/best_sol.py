# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and multiple optimization strategies.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance among all point pairs."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max
    
    # Strategy 1: Create mathematical configuration inspired by optimal point sets
    def create_mathematical_initial():
        """Create a mathematical configuration inspired by known good point distributions"""
        points = []
        
        # Outer ring: 12 points evenly distributed
        for i in range(12):
            angle = 2 * np.pi * i / 12
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        
        # Inner cluster: 4 points forming a square
        inner_radius = 0.15
        for i in range(4):
            angle = i * np.pi/2 + np.pi/4  # Square pattern
            x = 0.5 + inner_radius * np.cos(angle)
            y = 0.5 + inner_radius * np.sin(angle)
            points.append([x, y])
        
        return np.array(points[:16])
    
    # Strategy 2: Create hexagonal lattice initialization
    def create_hexagonal_initial():
        # Arrange points in a hexagonal pattern (4x4 grid with offset rows)
        points = []
        rows = 4
        cols = 4
        
        for i in range(rows):
            for j in range(cols):
                x = j + (i % 2) * 0.5  # Offset every other row
                y = i * math.sqrt(3) / 2
                points.append([x, y])
        
        # Normalize to unit square [0,1] x [0,1]
        points = np.array(points)
        
        # Find bounding box
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        # Avoid division by zero
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        # Scale to fit in [0,1] x [0,1]
        if x_range > 0:
            points[:, 0] = (points[:, 0] - x_min) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - y_min) / y_range
        
        # Center in unit square
        points[:, 0] -= np.mean(points[:, 0])
        points[:, 1] -= np.mean(points[:, 1])
        points[:, 0] += 0.5
        points[:, 1] += 0.5
        
        # Ensure all points are within [0,1] x [0,1]
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        
        return points
    
    # Strategy 3: Create circular pattern initialization
    def create_circular_initial():
        points = []
        for i in range(16):
            angle = 2 * math.pi * i / 16
            radius = 0.4  # Smaller radius to keep within bounds
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    # Strategy 4: Create random initialization with constraint
    def create_random_initial():
        np.random.seed(42)
        points = np.random.uniform(0, 1, (16, 2))
        return points
    
    # Strategy 5: Enhanced Simulated Annealing optimization
    def enhanced_simulated_annealing(initial_points, max_iter=12000, seed=42):
        np.random.seed(seed)
        
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Optimized parameters for better convergence with time constraints
        initial_temp = 0.15
        final_temp = 1e-8
        alpha = 0.999
        
        temp = initial_temp
        
        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing one point
            new_points = current_points.copy()
            point_idx = np.random.randint(0, len(current_points))
            
            # Perturb one point with adaptive displacement based on temperature
            displacement_magnitude = 0.02 + temp * 0.03
            displacement = np.random.normal(0, displacement_magnitude, 2)
            new_points[point_idx] = current_points[point_idx] + displacement
            
            # Keep points within [0,1] bounds
            new_points = np.clip(new_points, 0, 1)
            
            # Calculate new ratio
            new_ratio = compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < math.exp((new_ratio - current_ratio) / temp):
                current_points = new_points
                current_ratio = new_ratio
                
                # Update best solution if improved
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
            
            # Cool down temperature
            temp = max(final_temp, temp * alpha)
            
            # Early stopping if temperature gets too low
            if temp < final_temp:
                break
        
        return best_points, best_ratio
    
    # Strategy 6: Local optimization refinement with better parameters
    def local_optimization(initial_points, max_iter=1500):
        try:
            from scipy.optimize import minimize
            
            def objective(x_flat):
                points = x_flat.reshape(-1, 2)
                distances = pdist(points)
                if len(distances) == 0:
                    return 0.0
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max <= 1e-12:
                    return 0.0
                return -d_min / d_max  # Negative because we want to maximize
            
            # Use L-BFGS-B optimizer with bounds and better parameters
            bounds = [(0, 1)] * (len(initial_points) * 2)
            x0 = initial_points.flatten()
            
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                return optimized_points
        except Exception:
            # Fallback to original points if optimization fails
            pass
        return initial_points
    
    # Strategy 7: Optimized multi-start approach with better time management
    def optimized_multi_start():
        best_points = None
        best_ratio = 0.0
        
        # Test multiple initial configurations with optimized iterations
        initial_configs = [
            ("mathematical", create_mathematical_initial(), 12000),
            ("hexagonal", create_hexagonal_initial(), 12000),
            ("circular", create_circular_initial(), 8000),
            ("random", create_random_initial(), 8000)
        ]
        
        # Run optimization from each configuration with different strategies
        for config_name, initial_points, sa_iterations in initial_configs:
            try:
                # First SA optimization
                sa_points, sa_ratio = enhanced_simulated_annealing(
                    initial_points, max_iter=sa_iterations, seed=hash(config_name) % 1000
                )
                
                # Then local optimization
                refined_points = local_optimization(sa_points, max_iter=1500)
                refined_ratio = compute_min_max_ratio(refined_points)
                
                # Update best if improved
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
                    
            except Exception as e:
                continue
        
        # Single restart with perturbation for final improvement
        if best_points is not None:
            try:
                # Perturb the best points found so far
                perturbed = best_points + np.random.normal(0, 0.015, best_points.shape)
                perturbed = np.clip(perturbed, 0, 1)
                
                # Optimization with moderate settings
                restart_sa_points, _ = enhanced_simulated_annealing(
                    perturbed, max_iter=8000, seed=999
                )
                restart_refined_points = local_optimization(restart_sa_points, max_iter=1000)
                restart_ratio = compute_min_max_ratio(restart_refined_points)
                
                if restart_ratio > best_ratio:
                    best_ratio = restart_ratio
                    best_points = restart_refined_points.copy()
            except Exception:
                pass
        
        return best_points, best_ratio
    
    # Main optimization process with time management
    start_time = time.time()
    
    # Run optimized multi-start strategy
    best_points, best_ratio = optimized_multi_start()
    
    # Final refinement if we have a good solution and time permits
    if best_points is not None:
        # Try intensive refinement if time permits (with time check)
        elapsed = time.time() - start_time
        if elapsed < 55:  # Leave buffer for final processing
            final_refinement = local_optimization(best_points, max_iter=2000)
            final_ratio = compute_min_max_ratio(final_refinement)
            
            if final_ratio > best_ratio:
                return final_refinement
            else:
                return best_points
        else:
            return best_points
    else:
        # Fallback to mathematical construction
        return create_mathematical_initial()


# EVOLVE-BLOCK-END
