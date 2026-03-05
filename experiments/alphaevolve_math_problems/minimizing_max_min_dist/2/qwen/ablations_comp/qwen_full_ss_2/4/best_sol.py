# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing, differential_evolution, minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining multiple geometric initialization methods 
    with robust global and local optimization strategies.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # Strategy 1: Mathematical construction based on known optimal configurations
    # Using a 4x4 grid with strategic perturbations to avoid symmetry
    def generate_optimal_grid():
        points = []
        # Create a 4x4 grid
        for i in range(4):
            for j in range(4):
                x = i / 3.0 if i < 3 else 1.0
                y = j / 3.0 if j < 3 else 1.0
                points.append([x, y])
        
        points = np.array(points)
        
        # Apply strategic perturbations to break symmetry and improve distribution
        # Use a pattern that ensures points are spread out
        for i in range(len(points)):
            # Apply different perturbation patterns based on position
            row = i // 4
            col = i % 4
            
            # Add perturbation that increases distance between neighbors
            if row < 3 and col < 3:
                # Center points get more aggressive perturbation
                pert_x = 0.02 * np.sin(row * col * 0.5)
                pert_y = 0.02 * np.cos(row * col * 0.5)
            elif row == 0 or row == 3 or col == 0 or col == 3:
                # Edge points get moderate perturbation
                pert_x = 0.01 * np.sin(row + col)
                pert_y = 0.01 * np.cos(row + col)
            else:
                # Corner points get slight perturbation
                pert_x = 0.005 * np.sin(row * col)
                pert_y = 0.005 * np.cos(row * col)
                
            points[i][0] += pert_x
            points[i][1] += pert_y
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 2: Fibonacci spiral with better radial distribution
    def generate_fibonacci_spiral():
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = i * 2 * np.pi / golden_ratio
            # Improved radial distribution to avoid clustering at center
            r = 0.4 * np.sqrt(i / (n - 1)) if i < n - 1 else 0.4
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            points.append([x, y])
        return np.array(points)
    
    # Strategy 3: Concentric circles approach with more even distribution
    def generate_concentric_circles():
        points = []
        # Three concentric rings with different numbers of points
        radii = [0.2, 0.3, 0.4]
        points_per_ring = [4, 6, 6]  # Total 16 points
        
        for ring_idx, (radius, num_points) in enumerate(zip(radii, points_per_ring)):
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        return np.array(points[:n])
    
    # Strategy 4: Golden angle spiral with improved properties
    def generate_golden_spiral():
        points = []
        golden_angle = 2.399963229728653  # 2π(1 - 1/φ)
        
        for i in range(n):
            # Radial position with better distribution
            r = 0.4 * np.sqrt(i / (n - 1)) if n > 1 else 0.4
            theta = i * golden_angle
            
            # Convert to Cartesian coordinates
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            
            points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        
        return points
    
    # Strategy 5: Hexagonal grid approach
    def generate_hexagonal_grid():
        points = []
        # Create a hexagonal grid pattern
        for i in range(4):
            for j in range(4):
                if len(points) >= n:
                    break
                x = j + (i % 2) * 0.5
                y = i * np.sqrt(3)/2
                points.append([x, y])
        
        points = np.array(points[:n])
        
        # Normalize to fit nicely in [0,1] x [0,1]
        if points.shape[0] > 0:
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
            
            if x_max > x_min and y_max > y_min:
                scale_x = 1.0 / (x_max - x_min)
                scale_y = 1.0 / (y_max - y_min)
                
                points[:, 0] = (points[:, 0] - x_min) * scale_x * 0.9 + 0.05
                points[:, 1] = (points[:, 1] - y_min) * scale_y * 0.9 + 0.05
        
        return points
    
    # Generate multiple initial configurations
    configs = [
        generate_optimal_grid(),      # Main mathematical approach
        generate_fibonacci_spiral(),  # Good distribution
        generate_concentric_circles(), # Ring structure
        generate_golden_spiral(),     # Golden spiral
        generate_hexagonal_grid()     # Hexagonal pattern
    ]
    
    # Add some random perturbations to break symmetry and ensure diversity
    for config in configs:
        # Add noise to break symmetry
        noise = np.random.normal(0, 0.015, config.shape)
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
        distances = np.maximum(distances, 1e-15)
        
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
    
    # Try multiple initialization strategies with multiple optimization approaches
    for i, initial_config in enumerate(configs):
        if time.time() - start_time > max_time:
            break
            
        current_points = initial_config.copy()
        
        try:
            # Try multiple optimization approaches with different strategies
            # 1. Global optimization with dual annealing (primary approach)
            if time.time() - start_time < max_time * 0.6:
                da_result = dual_annealing(
                    objective,
                    bounds,
                    maxiter=1200,  # More iterations for better convergence
                    initial_temp=1000,
                    seed=42 + i,
                    no_local_search=True
                )
                
                if da_result.success:
                    optimized_points = da_result.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            
            # 2. Differential evolution for robust global search (backup)
            if time.time() - start_time < max_time * 0.8 and best_points is None:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=600,  # More iterations than before
                    popsize=30,   # Larger population
                    seed=42 + i,
                    strategy='best1bin'
                )
                
                if de_result.success:
                    optimized_points = de_result.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            
            # 3. Local refinement with multiple methods
            if time.time() - start_time < max_time * 0.9 and best_points is not None:
                # Try L-BFGS-B first with higher precision
                result1 = minimize(
                    objective,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-16},
                    tol=1e-16
                )
                
                if result1.success:
                    optimized_points = result1.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                
                # Also try SLSQP as backup with more iterations
                result2 = minimize(
                    objective,
                    best_points.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                
                if result2.success:
                    optimized_points = result2.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
        except Exception:
            continue
    
    # If still no good solution, return the best configuration found so far
    if best_points is None:
        # Fallback to the optimal grid as it has strong mathematical foundations
        fallback = generate_optimal_grid()
        # Add small random perturbations to break any remaining symmetries
        noise = np.random.normal(0, 0.01, fallback.shape)
        fallback += noise
        fallback[:, 0] = np.clip(fallback[:, 0], 0.05, 0.95)
        fallback[:, 1] = np.clip(fallback[:, 1], 0.05, 0.95)
        return fallback
    
    return best_points


# EVOLVE-BLOCK-END
