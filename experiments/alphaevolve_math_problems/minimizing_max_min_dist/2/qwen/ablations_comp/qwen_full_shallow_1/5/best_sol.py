# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


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
    
    # Generate diverse and high-quality initial configurations
    def generate_initial_configs():
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
        
        # 5. Hexagonal arrangement - maximizes uniformity
        hex_points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5 + (j % 2) * 0.25) / 4.0
                y = (j + 0.5) / 4.0
                hex_points.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
        configs.append(np.array(hex_points))
        
        # 6. Fibonacci spiral with better parameterization (inspired by top solutions)
        golden_ratio = (1 + np.sqrt(5)) / 2
        fib_points = []
        for i in range(16):
            theta = i * 2 * np.pi / golden_ratio
            # Use a slightly different scaling for better distribution
            r = 0.4 * np.sqrt(i / 15.0) + 0.05
            fib_points.append([0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)])
        configs.append(np.array(fib_points))
        
        # 7. Another hexagonal pattern with different offset
        hex_points2 = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.25 + (j % 2) * 0.5) / 4.0
                y = (j + 0.25) / 4.0
                hex_points2.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
        configs.append(np.array(hex_points2))
        
        # 8. Square spiral pattern - another way to distribute points evenly
        spiral_points = []
        for i in range(16):
            # Create a square spiral pattern
            layer = i // 4
            pos_in_layer = i % 4
            if pos_in_layer == 0:  # Right side
                x, y = 1 - 0.1 * layer, 0.1 * layer
            elif pos_in_layer == 1:  # Top side
                x, y = 1 - 0.1 * layer - 0.1, 0.1 * layer
            elif pos_in_layer == 2:  # Left side
                x, y = 1 - 0.1 * layer - 0.1, 0.1 * layer + 0.1
            else:  # Bottom side
                x, y = 1 - 0.1 * layer, 0.1 * layer + 0.1
            spiral_points.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
        configs.append(np.array(spiral_points))
        
        return configs
    
    # Multi-start optimization with SLSQP
    best_ratio = -np.inf
    best_points = None
    
    # Define constraints and bounds
    cons = {'type': 'ineq', 'fun': constraint_func}
    bounds = [(0, 1) for _ in range(32)]
    
    initial_configs = generate_initial_configs()
    
    # Use fewer restarts to save time while maintaining quality
    max_restarts = 6
    
    for i, start_points in enumerate(initial_configs[:max_restarts]):
        try:
            # Flatten for optimization
            x0 = start_points.flatten()
            
            # Use SLSQP optimization with tuned parameters for better speed and convergence
            # Different settings for early vs later iterations to balance speed and quality
            if i < 3:
                # Early iterations: less precise but faster
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    constraints=cons,
                    bounds=bounds,
                    options={'maxiter': 600, 'ftol': 1e-10, 'gtol': 1e-10}
                )
            else:
                # Later iterations: more precise
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    constraints=cons,
                    bounds=bounds,
                    options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12}
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
    
    # Final refinement with one additional optimization step
    if best_points is not None:
        try:
            x0 = best_points.flatten()
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
                
                if d_max > 1e-12 and (d_min / d_max) > best_ratio:
                    return final_points
                    
        except Exception as e:
            pass
    
    # If no good solution was found, return the best we have
    if best_points is None:
        # Fallback to regular grid
        grid_points = np.zeros((16, 2))
        grid_size = 4
        spacing = 1.0 / (grid_size - 1)
        for i in range(grid_size):
            for j in range(grid_size):
                grid_points[i * grid_size + j] = [i * spacing, j * spacing]
        return grid_points
    
    return best_points


# EVOLVE-BLOCK-END
