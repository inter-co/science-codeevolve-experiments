# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random
from scipy.optimize import differential_evolution
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction, global optimization, and adaptive refinement.
    
    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    def calculate_ratio(points):
        """Calculate the min/max distance ratio"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0.0
    
    def objective(params):
        """Objective function to minimize (negative of ratio)"""
        points = params.reshape((16, 2))
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return -1.0
        return -min_dist / max_dist
    
    def create_known_optimal_config():
        """Create a configuration based on known optimal 16-point arrangements"""
        # This is inspired by known optimal configurations from sphere packing literature
        # and attempts to create a highly symmetric yet non-uniform distribution
        points = []
        
        # Create 4 concentric rings with different radii and angular spacing
        radii = [0.2, 0.3, 0.4, 0.45]
        points_per_ring = [4, 4, 4, 4]  # Equal distribution for simplicity
        
        for ring_idx, (radius, num_points) in enumerate(zip(radii, points_per_ring)):
            for i in range(num_points):
                # Use golden angle distribution for even spacing
                angle = 2 * np.pi * i / num_points + ring_idx * 0.2  # Add some phase variation
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        
        # Add a central cluster of points
        for i in range(4):
            angle = 2 * np.pi * i / 4
            x = 0.5 + 0.05 * np.cos(angle)
            y = 0.5 + 0.05 * np.sin(angle)
            points.append([x, y])
        
        # Trim to exactly 16 points
        return np.array(points[:16])
    
    def create_fibonacci_sphere_projection():
        """Create points using Fibonacci sphere projection - known to produce good uniform distributions"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(16):
            # Fibonacci-like distribution on sphere
            y = 1 - (i / 15.0) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y*y)  # radius at y
            
            theta = np.arccos(y)  # polar angle
            phi_angle = i * 2.399963229728653  # Golden angle increment
            
            # Convert to Cartesian and project to 2D
            x = radius * np.cos(phi_angle)
            z = radius * np.sin(phi_angle)
            
            # Project to 2D (take x,y coordinates) and scale appropriately
            points.append([0.5 + 0.4 * x, 0.5 + 0.4 * y])
        
        return np.array(points)
    
    def create_hexagonal_grid_with_variations():
        """Create a hexagonal grid with strategic variations to avoid symmetry issues"""
        points = []
        # Create a 4x4 grid with some hexagonal offsets
        for i in range(4):
            for j in range(4):
                x = 0.1 + 0.8 * j / 3
                y = 0.1 + 0.8 * i / 3
                
                # Add hexagonal offset for odd rows
                if i % 2 == 1:
                    x += 0.8 / 6
                
                # Add small random perturbations to break symmetry
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                
                points.append([x, y])
        
        return np.array(points)
    
    def create_perturbed_regular_polygon():
        """Create points in a regular polygon with perturbations"""
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4
        points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        # Add perturbations to spread out points
        noise = np.random.normal(0, 0.03, points.shape)
        points += noise
        return points
    
    def create_symmetric_star_pattern():
        """Create a symmetric star-like pattern with careful spacing"""
        points = []
        
        # Central point
        points.append([0.5, 0.5])
        
        # Outer points arranged in a more complex pattern
        angles = np.linspace(0, 2*np.pi, 15, endpoint=False)
        for i, angle in enumerate(angles):
            # Use varying radius to avoid clustering
            radius_factor = 0.3 + 0.3 * np.sin(3 * angle) * 0.5
            r = 0.4 * (0.5 + 0.5 * radius_factor)
            points.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
        
        return np.array(points[:16])
    
    def create_optimized_combination():
        """Create an optimized combination of different approaches"""
        # Start with fibonacci sphere projection for good uniformity
        fib_points = create_fibonacci_sphere_projection()
        
        # Perturb slightly to break any inherent symmetries
        fib_points += np.random.normal(0, 0.02, fib_points.shape)
        
        # Clip to valid range
        fib_points = np.clip(fib_points, 0, 1)
        
        return fib_points
    
    # Strategy 1: Global optimization with higher quality settings for better results
    best_points = None
    best_ratio = 0.0
    
    try:
        bounds = [(0, 1) for _ in range(32)]
        # Higher quality global optimization for potentially better results
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=100,   # More iterations for better search
            popsize=30,    # Larger population size
            mutation=(0.8, 1),  # Slightly higher mutation rate
            recombination=0.9,   # Higher recombination
            seed=42,
            disp=False,
            atol=1e-10,
            rtol=1e-10
        )
        
        if de_result.success:
            optimized_points = de_result.x.reshape((16, 2))
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = calculate_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 2: Multiple local optimizations with more thorough exploration
    initial_strategies = [
        create_fibonacci_sphere_projection,
        create_hexagonal_grid_with_variations,
        create_perturbed_regular_polygon,
        create_optimized_combination,
        create_known_optimal_config
    ]
    
    # Run optimization from multiple good starting points
    for strategy_idx, strategy in enumerate(initial_strategies):
        try:
            initial_points = strategy()
            initial_points = np.clip(initial_points, 0, 1)
            
            # Run optimization with multiple methods for robustness
            methods_and_options = [
                ('L-BFGS-B', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('SLSQP', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('TNC', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
            ]
            
            for method, options in methods_and_options:
                try:
                    x_flat = initial_points.flatten()
                    result = minimize(
                        objective,
                        x_flat,
                        method=method,
                        bounds=[(0, 1) for _ in range(32)],
                        options=options
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape((16, 2))
                        optimized_points = np.clip(optimized_points, 0, 1)
                        ratio = calculate_ratio(optimized_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
                except Exception:
                    continue
                    
        except Exception:
            continue
    
    # Strategy 3: Final refinement with highest precision
    if best_points is not None:
        # Try final high-precision optimization
        try:
            x_flat = best_points.flatten()
            result = minimize(
                objective,
                x_flat,
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success:
                final_points = result.x.reshape((16, 2))
                final_points = np.clip(final_points, 0, 1)
                final_ratio = calculate_ratio(final_points)
                
                if final_ratio > best_ratio:
                    best_points = final_points
                    best_ratio = final_ratio
        except Exception:
            pass
    
    # Final fallback if nothing works well
    if best_points is None:
        # Return a carefully constructed configuration
        points = []
        # Create a configuration that balances uniformity and spread
        for i in range(4):
            for j in range(4):
                # Create a pattern that spreads points well
                x = 0.1 + 0.8 * j / 3 + np.random.normal(0, 0.015)
                y = 0.1 + 0.8 * i / 3 + np.random.normal(0, 0.015)
                points.append([x, y])
        best_points = np.array(points)
        best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
