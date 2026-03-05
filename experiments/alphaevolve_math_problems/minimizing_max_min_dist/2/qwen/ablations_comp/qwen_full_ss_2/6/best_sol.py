# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize, differential_evolution
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining multiple geometric initialization methods 
    with global and local optimization strategies.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    d = 2
    
    # Multiple initialization strategies inspired by good geometric configurations
    np.random.seed(42)
    
    # Strategy 1: Fibonacci spiral approach (good distribution)
    def generate_fibonacci_spiral():
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = i * 2 * np.pi / golden_ratio
            r = np.sqrt(i / (n - 1)) if i < n - 1 else 1.0
            x = 0.5 + r * np.cos(theta) * 0.4
            y = 0.5 + r * np.sin(theta) * 0.4
            points.append([x, y])
        return np.array(points)
    
    # Strategy 2: Regular polygon approach  
    def generate_regular_polygon():
        points = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            radius = 0.4  # Slightly inside the unit square
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    # Strategy 3: Hexagonal grid approach
    def generate_hexagonal_grid():
        points = []
        rows = 4
        cols = 4
        for row in range(rows):
            for col in range(cols):
                if len(points) >= n:
                    break
                x = col / (cols - 1.0) if cols > 1 else 0.5
                y = row / (rows - 1.0) if rows > 1 else 0.5
                if row % 2 == 1:
                    x += 1.0 / (2 * cols)  # Offset every other row
                points.append([x, y])
        return np.array(points[:n])
    
    # Strategy 4: Concentric circles approach
    def generate_concentric_circles():
        points = []
        # Two concentric rings
        radii = [0.25, 0.4]
        points_per_ring = [8, 8]  # Total 16 points
        
        for ring_idx, (radius, num_points) in enumerate(zip(radii, points_per_ring)):
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        return np.array(points[:n])
    
    # Strategy 5: Optimized grid with strategic outliers
    def generate_optimized_grid():
        # Start with a basic 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                points.append([(i + 0.5) / 4.0, (j + 0.5) / 4.0])
        
        points = np.array(points)
        
        # Add some carefully positioned outliers to improve minimum distance
        # Place some points near corners and edges
        outliers = [
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
            [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]
        ]
        
        # Replace some grid points with outliers
        for i, outlier in enumerate(outliers[:8]):
            if i < len(points):
                points[i] = outlier
        
        # Add small random noise for optimization robustness
        points += np.random.normal(0, 0.005, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    # Strategy 6: Golden angle spiral
    def generate_golden_spiral():
        points = []
        golden_angle = 2.399963229728653  # 2π(1 - 1/φ)
        
        for i in range(n):
            # Radial position (spiral pattern)
            r = np.sqrt(i / (n - 1)) if n > 1 else 0.5
            theta = i * golden_angle
            
            # Convert to Cartesian coordinates
            x = 0.5 + r * 0.4 * np.cos(theta)
            y = 0.5 + r * 0.4 * np.sin(theta)
            
            points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        
        return points
    
    # Strategy 7: Improved random distribution with better spacing
    def generate_improved_random():
        points = []
        # Generate points in a way that tries to avoid clustering
        for i in range(n):
            # Use rejection sampling to get good distribution
            attempts = 0
            while attempts < 100:
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                # Check if it's reasonably far from existing points
                valid = True
                if points:
                    dists = np.sqrt((np.array(points)[:, 0] - x)**2 + (np.array(points)[:, 1] - y)**2)
                    if np.min(dists) < 0.05:  # Minimum distance threshold
                        valid = False
                if valid:
                    points.append([x, y])
                    break
                attempts += 1
            if len(points) < i + 1:
                # Fallback to uniform random if we couldn't place properly
                points.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
        return np.array(points)
    
    # Generate multiple initial configurations
    configs = [
        generate_fibonacci_spiral(),
        generate_regular_polygon(), 
        generate_hexagonal_grid(),
        generate_concentric_circles(),
        generate_optimized_grid(),
        generate_golden_spiral(),
        generate_improved_random()
    ]
    
    # Add some random perturbations to break symmetry for all configurations
    for config in configs:
        # Add noise to break symmetry
        noise = np.random.normal(0, 0.02, config.shape)
        config += noise
        # Ensure within bounds
        config[:, 0] = np.clip(config[:, 0], 0.05, 0.95)
        config[:, 1] = np.clip(config[:, 1], 0.05, 0.95)
    
    # Objective function to maximize ratio (minimize negative ratio)
    def objective(params):
        points = params.reshape(n, d)
        distances = pdist(points)
        
        if len(distances) == 0:
            return float('inf')
            
        # Avoid division by zero
        distances = np.maximum(distances, 1e-12)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return float('inf')
            
        # Minimize negative of ratio (since we want to maximize ratio)
        ratio = d_min / d_max
        return -ratio  # Negative because we're minimizing
    
    # Compute ratio function for evaluation
    def compute_ratio(points_flat):
        points = points_flat.reshape(-1, 2)
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return d_min / d_max
    
    # Optimization bounds
    bounds = [(0, 1) for _ in range(2*n)]
    
    best_ratio = -1
    best_points = None
    
    # Time management - limit total optimization time
    start_time = time.time()
    max_time = 55  # seconds
    
    # Try multiple initialization strategies with focused optimization
    for i, initial_config in enumerate(configs):
        if time.time() - start_time > max_time:
            break
            
        current_points = initial_config.copy()
        
        try:
            # 1. First try dual annealing with high iteration count for global exploration
            if time.time() - start_time < max_time * 0.7:
                da_result = dual_annealing(
                    objective,
                    bounds,
                    maxiter=2000,  # Even more iterations for better convergence
                    initial_temp=2000,  # Higher temperature for better exploration
                    seed=42,
                    no_local_search=False  # Allow local search for better results
                )
                
                if da_result.success:
                    optimized_points = da_result.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            
            # 2. If we haven't found a good solution yet, try differential evolution
            if best_points is None and time.time() - start_time < max_time * 0.9:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=500,
                    popsize=30,
                    seed=42,
                    strategy='best1bin',
                    tol=1e-10
                )
                
                if de_result.success:
                    optimized_points = de_result.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            
            # 3. Local refinement with aggressive tolerance settings
            if best_points is not None and time.time() - start_time < max_time:
                # Try several local optimization methods with very tight tolerances
                # L-BFGS-B with very tight tolerances
                result1 = minimize(
                    objective,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18},
                    tol=1e-18
                )
                
                if result1.success:
                    optimized_points = result1.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
        except Exception:
            continue
    
    # If still no good solution, return the best configuration found so far
    if best_points is None:
        # Fallback to Fibonacci spiral as our best initial guess
        fallback = generate_fibonacci_spiral()
        # Add small random perturbations to break any remaining symmetries
        noise = np.random.normal(0, 0.01, fallback.shape)
        fallback += noise
        fallback[:, 0] = np.clip(fallback[:, 0], 0.05, 0.95)
        fallback[:, 1] = np.clip(fallback[:, 1], 0.05, 0.95)
        return fallback
    
    return best_points


# EVOLVE-BLOCK-END
