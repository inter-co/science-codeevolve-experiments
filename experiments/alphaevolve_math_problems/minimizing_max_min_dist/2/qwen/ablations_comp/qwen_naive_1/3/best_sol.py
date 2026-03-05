# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from scipy.spatial import ConvexHull
import random
from scipy.spatial import distance_matrix


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses advanced initialization and optimization techniques for improved results.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    d = 2
    
    # Advanced initialization strategies
    def initialize_fibonacci_sphere():
        """Initialize points using Fibonacci sphere algorithm adapted for 2D"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle in radians
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Map to 2D unit square
            points.append([(x + 1) / 2, (z + 1) / 2])
            
        return np.array(points)
    
    def initialize_hexagonal_grid():
        """Initialize points in a hexagonal grid pattern"""
        # For 16 points, create a 4x4 grid with offset rows
        points = []
        rows = 4
        cols = 4
        
        for i in range(rows):
            for j in range(cols):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.2165  # sqrt(3)/2 * 0.25
                points.append([x, y])
        
        # Take first n points and add jitter
        points = np.array(points[:n])
        # Add small random jitter to break symmetry
        jitter = np.random.normal(0, 0.01, (n, 2))
        points = points + jitter
        
        # Ensure points are within bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        
        return points
    
    def initialize_random():
        """Initialize points randomly"""
        return np.random.rand(n, 2)
    
    def initialize_polar():
        """Initialize points using polar coordinates"""
        points = []
        # Distribute points more evenly
        for i in range(n):
            angle = 2 * np.pi * i / n
            radius = 0.4 + 0.4 * np.random.random()  # Random radius between 0.4 and 0.8
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        
        return np.array(points)
    
    def initialize_regular_polygon():
        """Initialize points on a regular polygon with some randomness"""
        points = []
        # Place points on a circle, then add some noise
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        for angle in angles:
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        # Add small random noise to avoid perfect symmetry
        noise = np.random.normal(0, 0.02, (n, 2))
        points += noise
        points = np.clip(points, 0, 1)
        return points
    
    def initialize_voronoi_like():
        """Initialize points using a Voronoi-like distribution"""
        # Create a grid of points and then randomly perturb them
        points = []
        grid_size = 4
        spacing = 1.0 / (grid_size - 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * spacing
                y = i * spacing
                points.append([x, y])
        
        points = np.array(points[:n])
        # Add noise to break symmetry
        noise = np.random.normal(0, 0.01, (n, 2))
        points += noise
        points = np.clip(points, 0, 1)
        return points
    
    # Calculate min/max distance ratio
    def calculate_ratio(points):
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0
    
    # Enhanced objective function with better handling
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        # Apply boundary constraints by clipping
        points = np.clip(points, 0, 1)
        ratio = calculate_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    # More robust constraint handling
    def constraint_bounds(x_flat):
        points = x_flat.reshape(-1, 2)
        # Ensure all points are within [0,1] bounds
        constraints = []
        constraints.extend(points[:, 0])  # x >= 0
        constraints.extend(1 - points[:, 0])  # x <= 1
        constraints.extend(points[:, 1])  # y >= 0
        constraints.extend(1 - points[:, 1])  # y <= 1
        return np.array(constraints)
    
    # Try multiple initialization strategies and use the best one
    initial_strategies = [
        initialize_hexagonal_grid,
        initialize_fibonacci_sphere,
        initialize_random,
        initialize_polar,
        initialize_regular_polygon,
        initialize_voronoi_like
    ]
    
    best_initial_points = None
    best_ratio = 0
    
    for init_func in initial_strategies:
        try:
            points = init_func()
            ratio = calculate_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_initial_points = points.copy()
        except Exception as e:
            continue
    
    # If no good initialization found, fall back to hexagonal grid
    if best_initial_points is None:
        best_initial_points = initialize_hexagonal_grid()
    
    # Enhanced optimization with multiple attempts using different algorithms
    best_result_points = best_initial_points.copy()
    best_result_ratio = best_ratio
    
    # Try different optimization methods and parameters
    optimization_attempts = [
        {'method': 'L-BFGS-B', 'options': {'maxiter': 1000, 'ftol': 1e-12}},
        {'method': 'TNC', 'options': {'maxiter': 1000, 'ftol': 1e-12}},
        {'method': 'SLSQP', 'options': {'maxiter': 1000, 'ftol': 1e-12}}
    ]
    
    # Add a more sophisticated local search approach
    def enhanced_local_search(initial_points, max_iterations=200):
        current_points = initial_points.copy()
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Adaptive perturbation sizes
        perturbation_sizes = [0.005, 0.002, 0.001]
        
        for iteration in range(max_iterations):
            # Choose perturbation size based on iteration
            perturbation_size = perturbation_sizes[min(iteration // 50, len(perturbation_sizes)-1)]
            
            # Make random perturbations
            perturbed_points = current_points + np.random.normal(0, perturbation_size, (n, 2))
            # Keep within bounds
            perturbed_points = np.clip(perturbed_points, 0, 1)
            
            new_ratio = calculate_ratio(perturbed_points)
            
            if new_ratio > current_ratio:
                current_points = perturbed_points
                current_ratio = new_ratio
                
                if new_ratio > best_ratio:
                    best_points = perturbed_points.copy()
                    best_ratio = new_ratio
            elif np.random.random() < 0.05:  # Occasionally accept worse solutions
                current_points = perturbed_points
                current_ratio = new_ratio
        
        return best_points, best_ratio
    
    # Try the enhanced local search approach
    try:
        local_best_points, local_best_ratio = enhanced_local_search(best_initial_points, 150)
        if local_best_ratio > best_result_ratio:
            best_result_points = local_best_points
            best_result_ratio = local_best_ratio
    except Exception as e:
        pass
    
    # Try optimization with multiple starting points and different approaches
    for attempt in range(5):  # More attempts
        try:
            # Add more jitter to the initial points
            jittered_points = best_initial_points + np.random.normal(0, 0.03, (n, 2))
            jittered_points = np.clip(jittered_points, 0, 1)
            
            x0 = jittered_points.flatten()
            
            # Try different optimization methods
            for opt_config in optimization_attempts:
                try:
                    result = minimize(
                        objective,
                        x0,
                        method=opt_config['method'],
                        options=opt_config['options'],
                        tol=1e-12
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        ratio = calculate_ratio(final_points)
                        
                        if ratio > best_result_ratio:
                            best_result_points = final_points
                            best_result_ratio = ratio
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    # Final refinement with more intensive local search
    try:
        final_points, final_ratio = enhanced_local_search(best_result_points, 100)
        if final_ratio > best_result_ratio:
            best_result_points = final_points
            best_result_ratio = final_ratio
    except Exception as e:
        pass
    
    # Additional refinement using a simple gradient-free approach
    try:
        # Try a simple hill-climbing approach
        current_points = best_result_points.copy()
        current_ratio = best_result_ratio
        
        for _ in range(50):
            # Try to improve by moving individual points slightly
            for i in range(n):
                # Save current point
                original_point = current_points[i].copy()
                
                # Try small perturbations
                for _ in range(10):
                    perturbation = np.random.normal(0, 0.005, 2)
                    new_point = original_point + perturbation
                    new_point = np.clip(new_point, 0, 1)
                    
                    # Temporarily update point
                    current_points[i] = new_point
                    new_ratio = calculate_ratio(current_points)
                    
                    if new_ratio > current_ratio:
                        current_ratio = new_ratio
                    else:
                        # Revert if not better
                        current_points[i] = original_point
                        
        if current_ratio > best_result_ratio:
            best_result_points = current_points
            best_result_ratio = current_ratio
            
    except Exception as e:
        pass
    
    return best_result_points


# EVOLVE-BLOCK-END
