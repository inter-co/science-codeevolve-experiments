# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining mathematical constructions with advanced optimization.
    
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
    
    def optimize_with_multiple_strategies():
        """Try multiple optimization strategies to find the best solution"""
        best_points = None
        best_ratio = -float('inf')
        
        # Strategy 1: Start with better mathematical constructions (fewer but higher quality)
        initial_configs = [
            generate_hexagonal_grid(),  # Often gives good starting point
            generate_fibonacci_like_pattern(),  # Good for even distribution
        ]
        
        for i, points in enumerate(initial_configs):
            try:
                x0 = points.flatten()
                bounds = [(0, 1) for _ in range(32)]
                
                # Use differential evolution with parameters matching inspirations but more efficient
                result = differential_evolution(
                    objective,
                    bounds,
                    seed=42 + i,
                    maxiter=200,      # Reduced iterations to save time
                    popsize=20,       # Reduced population size for speed
                    tol=1e-11,        # Tolerate slightly looser convergence
                    recombination=0.9,
                    mutation=(0.8, 1.0),
                    workers=1
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
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
            except Exception:
                continue
        
        # Strategy 2: If DE didn't work, try SLSQP with fewer restarts to save time
        if best_points is None:
            best_points = generate_hexagonal_grid()
            best_ratio = -float('inf')
        
        # Try fewer restarts but with better quality starting points
        max_restarts = 8  # Reduced from 15 to 8 for time efficiency
        
        for restart in range(max_restarts):
            try:
                # Select different initialization strategies
                if restart < 3:
                    points = generate_hexagonal_grid()
                elif restart < 6:
                    points = generate_fibonacci_like_pattern()
                else:
                    # Random perturbation of the current best
                    points = best_points + np.random.normal(0, 0.015, best_points.shape)
                    points = np.clip(points, 0, 1)
                
                x0 = points.flatten()
                bounds = [(0, 1) for _ in range(32)]
                
                # Use SLSQP with moderate tolerances for faster convergence
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-11},
                    tol=1e-11
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
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
            except Exception:
                continue
        
        # Strategy 3: Final refinement with a few random restarts
        for restart in range(5):  # Reduced from 10 to 5 for time efficiency
            try:
                np.random.seed(restart * 1000 + 42)
                points = np.random.rand(16, 2)
                x0 = points.flatten()
                bounds = [(0, 1) for _ in range(32)]
                
                # Use SLSQP with moderate tolerances
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-11},
                    tol=1e-11
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
                        if ratio > best_ratio:
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
