# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
from scipy.optimize import differential_evolution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initialization strategies, global optimization, 
    and targeted local refinement.

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
    
    # Strategy 1: Generate diverse and high-quality initial configurations (inspired by inspiration programs)
    def generate_diverse_configs():
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
        
        # 7. Concentric rings - good for coverage
        ring_points = []
        # Inner ring (4 points)
        for i in range(4):
            angle = i * np.pi/2
            ring_points.append([0.5 + 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle)])
        # Middle ring (8 points)
        for i in range(8):
            angle = i * np.pi/4
            ring_points.append([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
        # Outer ring (4 points)
        for i in range(4):
            angle = i * np.pi/2
            ring_points.append([0.5 + 0.6 * np.cos(angle), 0.5 + 0.6 * np.sin(angle)])
        configs.append(np.array(ring_points[:16]))
        
        return configs
    
    # Strategy 2: Enhanced multi-start optimization with better parameter selection
    def enhanced_multi_start():
        """Enhanced multi-start optimization with smarter strategy"""
        best_ratio = -np.inf
        best_points = None
        
        # Define constraints
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        # Define bounds for optimization (points must remain in [0,1]x[0,1])
        bounds = [(0, 1) for _ in range(32)]
        
        # Get diverse initial configurations
        initial_configs = generate_diverse_configs()
        
        # Try multiple starting points with different optimization parameters
        # Use fewer iterations to save time but still allow convergence
        for i, start_points in enumerate(initial_configs):
            try:
                # Flatten for optimization
                x0 = start_points.flatten()
                
                # Use multiple optimization methods with different parameters
                methods_to_try = [
                    ('L-BFGS-B', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                    ('SLSQP', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                    ('TNC', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
                ]
                
                for method, options in methods_to_try:
                    try:
                        result = minimize(
                            objective,
                            x0,
                            method=method,
                            constraints=cons,
                            bounds=bounds,
                            options=options
                        )
                        
                        if result.success:
                            final_points = result.x.reshape(-1, 2)
                            # Ensure points are within [0,1] bounds
                            final_points = np.clip(final_points, 0, 1)
                            
                            # Calculate final ratio
                            distances = pdist(final_points)
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            
                            if d_max > 1e-12:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = final_points.copy()
                                    
                    except Exception:
                        continue
                        
            except Exception as e:
                continue
        
        # If we didn't find anything good, fall back to the first config
        if best_points is None:
            return initial_configs[0]
        
        return best_points
    
    # Strategy 3: Global optimization for final refinement
    def global_refinement(points):
        """Use global optimization to refine the solution"""
        def global_objective(x_flat):
            points = x_flat.reshape(-1, 2)
            distances = pdist(points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max == 0:
                return -1e10
                
            ratio = d_min / d_max
            return -ratio
        
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            result = differential_evolution(
                global_objective,
                bounds,
                maxiter=100,  # Reduced iterations to stay within time limit
                popsize=15,   # Moderate population size
                tol=1e-8,     # Tighter tolerance
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                
                distances = pdist(optimized_points)
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > 0.95 * (compute_ratio_if_better(points) if 'compute_ratio_if_better' in globals() else 0):
                        return optimized_points
        except Exception:
            pass
        
        return points
    
    def compute_ratio_if_better(points):
        """Helper function to compute ratio for comparison"""
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max > 0:
            return d_min / d_max
        return 0
    
    # Main optimization process
    # First, get a good initial solution using multi-start optimization
    optimized_points = enhanced_multi_start()
    
    # Then perform global refinement
    refined_points = global_refinement(optimized_points)
    
    # Final local optimization to polish the result
    try:
        # Define constraints
        cons = {'type': 'ineq', 'fun': constraint_func}
        bounds = [(0, 1) for _ in range(32)]
        x0 = refined_points.flatten()
        
        # Use more aggressive local optimization
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            constraints=cons,
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            
            # Check if this improved the ratio
            distances = pdist(final_points)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 1e-12:
                ratio = d_min / d_max
                distances_orig = pdist(refined_points)
                d_min_orig = np.min(distances_orig)
                d_max_orig = np.max(distances_orig)
                orig_ratio = d_min_orig / d_max_orig if d_max_orig > 0 else 0
                
                if ratio > orig_ratio:
                    return final_points
                    
    except Exception:
        pass
    
    return refined_points


# EVOLVE-BLOCK-END
