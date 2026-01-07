# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining mathematical constructions with advanced optimization.
    Focuses on achieving better performance than the AlphaEvolve benchmark of 0.2786.
    
    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape flat array back to 16x2 points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0:
            return float('inf')
            
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Handle edge case where all points are coincident
        if d_max == 0:
            return float('inf')
            
        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        return -d_min / d_max
    
    def generate_hexagonal_grid():
        """Generate a hexagonal grid pattern which is known to be good for point dispersion"""
        points = []
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 0.5
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 0.5
        
        # Use proper hexagonal spacing
        hex_spacing_y = spacing_y * np.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                x = j * spacing_x
                y = i * hex_spacing_y
                
                # Add hexagonal offset
                if i % 2 == 1:
                    x += spacing_x * 0.5
                
                # Add small random perturbation to avoid degeneracy
                x += (np.random.random() - 0.5) * 0.015
                y += (np.random.random() - 0.5) * 0.015
                
                # Ensure within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                
                points.append([x, y])
        
        return np.array(points[:16])
    
    def generate_fibonacci_like_pattern():
        """Generate a fibonacci-inspired distribution for even point distribution"""
        points = []
        # Use golden ratio distribution for 16 points
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(16):
            # Distribute points more evenly
            y = 1 - (i / 15.0) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y*y)
            
            # Use golden angle for better distribution
            theta = i * 2.399963229728653  # angle increment (close to 2π/φ²)
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Map to 2D square [0,1]x[0,1]
            px = (x + 1) / 2.0
            py = (z + 1) / 2.0
            
            # Add small random perturbation
            px += (np.random.random() - 0.5) * 0.02
            py += (np.random.random() - 0.5) * 0.02
            
            # Ensure within bounds
            px = max(0.01, min(0.99, px))
            py = max(0.01, min(0.99, py))
            
            points.append([px, py])
    
        return np.array(points)
    
    def generate_regular_grid():
        """Generate a regular 4x4 grid which is known to be good for point dispersion"""
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + 0.125
                y = i * 0.25 + 0.125
                points.append([x, y])
        return np.array(points[:16])
    
    def generate_optimal_configurations():
        """Generate several high-quality initial configurations"""
        configs = []
        
        # Hexagonal grid
        configs.append(('hex', generate_hexagonal_grid()))
        
        # Fibonacci-like pattern
        configs.append(('fib', generate_fibonacci_like_pattern()))
        
        # Regular grid
        configs.append(('reg', generate_regular_grid()))
        
        # Perturbed regular grid for variety
        reg_grid = generate_regular_grid()
        perturbed = reg_grid + np.random.normal(0, 0.01, reg_grid.shape)
        perturbed = np.clip(perturbed, 0, 1)
        configs.append(('pert_reg', perturbed))
        
        return configs
    
    def optimize_with_strategy(initial_points, strategy_name, maxiter=500):
        """Optimize a single initial configuration"""
        try:
            x0 = initial_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            if strategy_name.startswith('de'):
                # Use differential evolution for global optimization
                result = differential_evolution(
                    objective,
                    bounds,
                    seed=42,
                    maxiter=maxiter,
                    popsize=20,
                    tol=1e-12,
                    recombination=0.9,
                    mutation=(0.8, 1.0),
                    workers=1
                )
            else:
                # Use SLSQP for local optimization
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                
                # Evaluate the result
                distances = pdist(optimized_points)
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 0:
                    ratio = d_min / d_max
                    return optimized_points, ratio
        except Exception:
            pass
        return None, -float('inf')
    
    def optimize_with_multiple_strategies():
        """Try multiple optimization strategies to find the best solution"""
        best_points = None
        best_ratio = -float('inf')
        
        # Generate initial configurations
        configs = generate_optimal_configurations()
        
        # First, try differential evolution on top configurations
        for config_name, points in configs:
            optimized_points, ratio = optimize_with_strategy(points, 'de', maxiter=300)
            if optimized_points is not None and ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
        
        # If no good solution yet, try SLSQP with more aggressive restarts
        if best_points is None:
            best_points = configs[0][1]  # Start with first configuration
            best_ratio = -float('inf')
        
        # Try SLSQP with restarts on the best so far
        for restart in range(8):  # Reduced from 15 to reduce time
            try:
                # Perturb the current best
                points = best_points + np.random.normal(0, 0.015, best_points.shape)
                points = np.clip(points, 0, 1)
                
                optimized_points, ratio = optimize_with_strategy(points, 'slsqp', maxiter=400)
                if optimized_points is not None and ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
            except Exception:
                continue
        
        # Final refinement with more focused optimization
        if best_points is not None:
            # Try a few more targeted optimizations
            for _ in range(3):
                try:
                    # Slightly perturb and optimize
                    points = best_points + np.random.normal(0, 0.005, best_points.shape)
                    points = np.clip(points, 0, 1)
                    optimized_points, ratio = optimize_with_strategy(points, 'slsqp', maxiter=200)
                    if optimized_points is not None and ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                except Exception:
                    continue
        
        # Return the best solution found
        if best_points is not None:
            return best_points
        else:
            # Fallback to hexagonal grid if nothing worked
            return generate_hexagonal_grid()
    
    # Execute optimization
    return optimize_with_multiple_strategies()


# EVOLVE-BLOCK-END
