# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and SLSQP optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return -1e10
            
        # Return negative ratio (since we're minimizing)
        return -d_min / d_max
    
    def constraint_func(x_flat):
        """Constraint function to keep points within unit square"""
        points = x_flat.reshape(-1, 2)
        # Ensure all points are within [0,1] x [0,1]
        return np.concatenate([
            points[:, 0],           # x coordinates >= 0
            1 - points[:, 0],       # x coordinates <= 1
            points[:, 1],           # y coordinates >= 0
            1 - points[:, 1]        # y coordinates <= 1
        ])
    
    # Strategy: Generate high-quality initial configurations
    def generate_initial_configs():
        """Generate high-quality initial configurations"""
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
        
        # 3. Random with symmetry breaking - for escaping local optima
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        configs.append(random_points)
        
        # 4. Perturbed regular grid - good balance of structure and randomness
        np.random.seed(123)
        perturbed_grid = grid_points + np.random.normal(0, 0.03, grid_points.shape)
        perturbed_grid = np.clip(perturbed_grid, 0, 1)
        configs.append(perturbed_grid)
        
        # 5. Fibonacci spiral - good for uniform distribution
        golden_ratio = (1 + np.sqrt(5)) / 2
        fib_points = []
        for i in range(16):
            theta = i * 2 * np.pi / golden_ratio
            r = 0.4 * np.sqrt(i / 15.0) + 0.05
            fib_points.append([0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)])
        configs.append(np.array(fib_points))
        
        # 6. Improved hexagonal arrangement - more evenly distributed (from inspiration 1)
        hex_points = []
        for i in range(4):
            for j in range(4):
                # Create a better hexagonal pattern with proper offset
                x = (i + 0.5 + (j % 2) * 0.25) / 4.0
                y = (j + 0.5) / 4.0
                hex_points.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
        configs.append(np.array(hex_points))
        
        # 7. Concentric circles with different radii - improved version
        concentric_points = []
        # Add points in concentric rings
        radii = [0.15, 0.3, 0.45, 0.6]  # Different radii
        points_per_ring = [4, 8, 12, 12]  # Points per ring (more points in outer rings)
        
        for i, (radius, num_points) in enumerate(zip(radii, points_per_ring)):
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            for angle in angles:
                concentric_points.append([0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)])
        
        # If we don't have enough points, fill with center points
        while len(concentric_points) < 16:
            concentric_points.append([0.5, 0.5])
        configs.append(np.array(concentric_points[:16]))
        
        return configs
    
    # Strategy: Efficient multi-start optimization with smart parameter tuning
    def efficient_multi_start():
        """Efficient multi-start optimization with strategic parameter selection"""
        best_ratio = -np.inf
        best_points = None
        
        # Define constraints
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        # Define bounds for optimization (points must remain in [0,1]x[0,1])
        bounds = [(0, 1) for _ in range(32)]
        
        # Get high-quality initial configurations
        initial_configs = generate_initial_configs()
        
        # Try multiple starting points with optimized parameters
        # Try fewer but more carefully selected configurations to save time
        for i, start_points in enumerate(initial_configs):
            try:
                # Flatten for optimization
                x0 = start_points.flatten()
                
                # Use SLSQP optimization with tuned parameters
                # Prioritize the most promising initial configurations
                if i < 3:  # First 3 configs get more iterations (they're typically better)
                    maxiter = 1500
                else:  # Remaining configs get fewer iterations
                    maxiter = 600
                    
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    constraints=cons,
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    distances = pdist(final_points)
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    
                    if d_max > 1e-12:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
            except Exception as e:
                continue
        
        # If we didn't find anything good, fall back to the first config
        if best_points is None:
            return initial_configs[0]
        
        return best_points
    
    # Perform the optimization
    optimized_points = efficient_multi_start()
    
    # Final refinement with modest optimization
    try:
        # Try one more optimization with the best result using moderate tolerances
        cons = {'type': 'ineq', 'fun': constraint_func}
        bounds = [(0, 1) for _ in range(32)]
        x0 = optimized_points.flatten()
        
        # Conservative optimization to avoid overfitting
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=cons,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            distances = pdist(final_points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 1e-12 and (d_min / d_max) > 0.999 * (best_ratio if 'best_ratio' in locals() else 0):
                return final_points
                
    except Exception as e:
        pass
    
    return optimized_points


# EVOLVE-BLOCK-END
