# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import random
from itertools import combinations
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, multiple optimization strategies,
    and energy-based refinement for better global optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_min_max_ratio(points: np.ndarray) -> float:
        """Calculate the min/max distance ratio for given points."""
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0
    
    def objective(x):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x.reshape(-1, 2)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return -1e10
        return -min_dist / max_dist
    
    def constraint_func(x):
        """Constraint to keep points within unit square"""
        points = x.reshape(-1, 2)
        # Keep all points in [0,1] x [0,1]
        return np.concatenate([
            points[:, 0],           # x coordinates >= 0
            1 - points[:, 0],       # x coordinates <= 1
            points[:, 1],           # y coordinates >= 0
            1 - points[:, 1]        # y coordinates <= 1
        ])
    
    def generate_diverse_initial_configs():
        """Generate diverse high-quality initial configurations"""
        configs = []
        
        # 1. Regular grid (4x4) - stable starting point
        grid_points = np.zeros((16, 2))
        grid_size = 4
        spacing = 1.0 / (grid_size - 1)
        for i in range(grid_size):
            for j in range(grid_size):
                grid_points[i * grid_size + j] = [i * spacing, j * spacing]
        configs.append(grid_points)
        
        # 2. Circle arrangement - good for spreading points
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        circle_points = np.column_stack([
            0.5 + 0.4 * np.cos(angles),
            0.5 + 0.4 * np.sin(angles)
        ])
        configs.append(circle_points)
        
        # 3. Fibonacci spiral with better parameterization
        golden_ratio = (1 + np.sqrt(5)) / 2
        fib_points = []
        for i in range(16):
            theta = i * 2 * np.pi / golden_ratio
            r = 0.4 * np.sqrt(i / 15.0) + 0.05
            fib_points.append([0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)])
        configs.append(np.array(fib_points))
        
        # 4. Hexagonal arrangement with offset
        hex_points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5 + (j % 2) * 0.25) / 4.0
                y = (j + 0.5) / 4.0
                hex_points.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
        configs.append(np.array(hex_points))
        
        # 5. Perturbed regular grid for better distribution
        np.random.seed(123)
        perturbed_grid = grid_points + np.random.normal(0, 0.03, grid_points.shape)
        perturbed_grid = np.clip(perturbed_grid, 0, 1)
        configs.append(perturbed_grid)
        
        # 6. Another hexagonal pattern with different offset
        hex_points2 = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.25 + (j % 2) * 0.5) / 4.0
                y = (j + 0.25) / 4.0
                hex_points2.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
        configs.append(np.array(hex_points2))
        
        # 7. Spiral pattern with different parameters
        spiral_points = []
        for i in range(16):
            angle = i * 0.7
            radius = 0.4 * (i / 15.0)
            spiral_points.append([0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)])
        configs.append(np.array(spiral_points))
        
        # 8. Square grid with perturbation
        square_points = np.array([[i/3, j/3] for i in range(4) for j in range(4)])
        np.random.seed(999)
        square_points += np.random.normal(0, 0.02, square_points.shape)
        square_points = np.clip(square_points, 0, 1)
        configs.append(square_points)
        
        # 9. Regular 16-gon (approximation) - mathematically elegant
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        regular_polygon = np.column_stack([
            0.5 + 0.4 * np.cos(angles),
            0.5 + 0.4 * np.sin(angles)
        ])
        configs.append(regular_polygon)
        
        # 10. Perturbed regular polygon - combines structure with randomness
        np.random.seed(42)
        perturbed_polygon = regular_polygon + np.random.normal(0, 0.02, regular_polygon.shape)
        perturbed_polygon = np.clip(perturbed_polygon, 0, 1)
        configs.append(perturbed_polygon)
        
        # 11. Lattice-based arrangement - structured yet flexible
        lattice_points = []
        for i in range(4):
            for j in range(4):
                lattice_points.append([i * 0.25 + 0.125, j * 0.25 + 0.125])
        configs.append(np.array(lattice_points))
        
        # 12. Random with controlled spread - ensures good coverage
        np.random.seed(555)
        random_spread = np.random.rand(16, 2) * 0.8 + 0.1  # Keep away from edges
        configs.append(random_spread)
        
        return configs
    
    def energy_based_refinement(points: np.ndarray, max_iter: int = 1000) -> np.ndarray:
        """Improved energy-based refinement using gradient descent with adaptive learning rate"""
        current_points = points.copy()
        current_ratio = calculate_min_max_ratio(current_points)
        
        # Adaptive gradient descent with momentum
        step_size = 0.01
        momentum = 0.9
        velocity = np.zeros_like(current_points)
        
        for iteration in range(max_iter):
            # Calculate gradients numerically using central differences
            gradients = np.zeros_like(current_points)
            eps = 1e-6
            
            for i in range(len(current_points)):
                # Forward difference approximation for each dimension
                test_points_plus = current_points.copy()
                test_points_plus[i] += eps
                test_points_plus = np.clip(test_points_plus, 0, 1)
                ratio_plus = calculate_min_max_ratio(test_points_plus)
                
                test_points_minus = current_points.copy()
                test_points_minus[i] -= eps
                test_points_minus = np.clip(test_points_minus, 0, 1)
                ratio_minus = calculate_min_max_ratio(test_points_minus)
                
                gradients[i] = (ratio_plus - ratio_minus) / (2 * eps)
            
            # Update with momentum
            velocity = momentum * velocity + step_size * gradients
            new_points = current_points + velocity
            new_points = np.clip(new_points, 0, 1)
            
            new_ratio = calculate_min_max_ratio(new_points)
            if new_ratio > current_ratio:
                current_points = new_points
                current_ratio = new_ratio
                # Increase learning rate when improving
                step_size = min(step_size * 1.05, 0.1)
            else:
                # Decrease learning rate when not improving
                step_size *= 0.95
                if step_size < 1e-8:
                    break
        
        return current_points
    
    def advanced_optimization(initial_points: np.ndarray, max_iter: int = 1000) -> tuple:
        """Advanced optimization combining multiple strategies"""
        best_points = initial_points.copy()
        best_ratio = calculate_min_max_ratio(best_points)
        
        # Try multiple optimization methods
        strategies = [
            ('SLSQP', {'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14}),
            ('L-BFGS-B', {'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14})
        ]
        
        for method, options in strategies:
            try:
                # Flatten for optimization
                x0 = best_points.flatten()
                
                # Define constraints
                cons = {'type': 'ineq', 'fun': constraint_func}
                
                # Optimize
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    constraints=cons,
                    options=options
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    ratio = calculate_min_max_ratio(optimized_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
            except Exception:
                continue
        
        return best_points, best_ratio
    
    # Generate diverse initial configurations
    initial_configs = generate_diverse_initial_configs()
    
    # Multi-start optimization with different strategies
    best_ratio = -float('inf')
    best_points = None
    
    # Try multiple optimization strategies with different starting points
    # Focus on the most promising configurations from our diverse set
    num_configs_to_try = min(8, len(initial_configs))  # Limit to prevent timeout
    
    for i in range(num_configs_to_try):
        start_points = initial_configs[i]
        
        # First optimization with different methods
        optimized_points, ratio = advanced_optimization(start_points, max_iter=500)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # If we didn't find anything, fallback to first configuration
    if best_points is None:
        best_points = initial_configs[0].copy()
    
    # Apply advanced energy refinement to the best solution
    refined_points = energy_based_refinement(best_points, max_iter=500)
    refined_ratio = calculate_min_max_ratio(refined_points)
    
    # If refinement helped, use it; otherwise stick with previous best
    if refined_ratio > best_ratio:
        best_points = refined_points
        best_ratio = refined_ratio
    
    # Final check - if we're close to the benchmark, do one last refinement
    if best_ratio > 0.27:
        final_refinement = energy_based_refinement(best_points, max_iter=300)
        final_ratio = calculate_min_max_ratio(final_refinement)
        if final_ratio > best_ratio:
            best_points = final_refinement
    
    return best_points


# EVOLVE-BLOCK-END
