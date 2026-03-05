# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from scipy.spatial import distance_matrix
import random
from typing import Tuple
from numba import jit, prange
import warnings
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.spatial import SphericalVoronoi
import math


@jit(nopython=True)
def compute_min_max_distances(points):
    """Compute min and max distances more efficiently"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0
    
    # Only compute upper triangle to avoid redundant calculations
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    return min_dist, max_dist


@jit(nopython=True)
def compute_min_max_distances_vectorized(points):
    """Vectorized computation for better performance"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0
    
    # Use vectorized operations where possible
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    return min_dist, max_dist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def objective(x_flat):
        """Objective function to maximize the min/max distance ratio"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Use efficient distance computation
        try:
            min_dist, max_dist = compute_min_max_distances_vectorized(points)
        except:
            # Fallback to scipy version if numba fails
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return -np.inf
            
        # Return negative because we want to maximize
        return -min_dist / max_dist
    
    # Generate high-quality initial configurations
    np.random.seed(42)  # For reproducibility
    
    # Strategy 1: Optimized hexagonal packing pattern (more refined)
    def generate_hexagonal_layout():
        points = []
        # Create a more optimized hexagonal pattern
        row_offsets = [0, 0.5]  # Alternating row offsets for hexagonal packing
        spacing_x = 0.25
        spacing_y = 0.25 * np.sqrt(3) / 2  # Vertical spacing for hexagon
        
        for i in range(4):
            for j in range(4):
                if len(points) < 16:
                    x = j * spacing_x + row_offsets[i % 2] * spacing_x / 2
                    y = i * spacing_y
                    
                    # Add small jitter to avoid degeneracy
                    jitter = 0.01
                    x += (np.random.rand() - 0.5) * jitter
                    y += (np.random.rand() - 0.5) * jitter
                    
                    points.append([x, y])
        
        # Ensure exactly 16 points and clip to bounds
        points = points[:16]
        points = np.clip(points, 0, 1)
        return np.array(points)
    
    # Strategy 2: Optimized grid with better spacing and more randomness
    def generate_optimized_grid_layout():
        points = []
        # Create a 4x4 grid with strategic spacing
        for i in range(4):
            for j in range(4):
                if len(points) < 16:
                    # Better grid spacing with slight offsets
                    x = j * 0.25 + (i % 2) * 0.08
                    y = i * 0.25
                    points.append([x, y])
        
        # Add noise to make it non-degenerate and improve spread
        for i in range(len(points)):
            points[i][0] += (np.random.rand() - 0.5) * 0.03
            points[i][1] += (np.random.rand() - 0.5) * 0.03
            
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 3: Concentrated circular arrangement with better distribution
    def generate_circular_layout():
        points = []
        # Distribute points around a circle with radial variation
        for i in range(16):
            angle = 2 * np.pi * i / 16
            # Use radial distribution that avoids clustering
            radius = 0.35 + 0.15 * np.sin(angle * 2)  # Slight variation
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 4: Golden ratio based spiral with better distribution
    def generate_golden_spiral_layout():
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Use more points in spiral to get better distribution
        for i in range(20):  # Generate more points than needed
            angle = i * 2 * np.pi / phi
            radius = np.sqrt(i / 19.0) * 0.4  # Radial scaling
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            if len(points) < 16:
                points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 5: More advanced configuration inspired by sphere packing with improved jitter
    def generate_advanced_layout():
        # Start with a regular grid but apply more sophisticated perturbation
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                points.append([x, y])
        
        # Apply more strategic perturbation to improve spread
        for i in range(len(points)):
            # Add more substantial but controlled jitter
            points[i][0] += (np.random.rand() - 0.5) * 0.04
            points[i][1] += (np.random.rand() - 0.5) * 0.04
            
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 6: Improved random layout with better constraints and higher quality
    def generate_random_improved():
        # Generate points with better distribution properties
        points = []
        for _ in range(16):
            # Use a more uniform distribution with careful bounds
            x = np.random.uniform(0.08, 0.92)
            y = np.random.uniform(0.08, 0.92)
            points.append([x, y])
        
        points = np.array(points)
        return points
    
    # Strategy 7: Hybrid approach - combine several good patterns with better selection
    def generate_hybrid_layout():
        points = []
        
        # Start with hexagonal pattern
        row_offsets = [0, 0.5]
        spacing_x = 0.25
        spacing_y = 0.25 * np.sqrt(3) / 2
        
        for i in range(4):
            for j in range(4):
                if len(points) < 16:
                    x = j * spacing_x + row_offsets[i % 2] * spacing_x / 2
                    y = i * spacing_y
                    points.append([x, y])
        
        # Perturb slightly to avoid regularity with better values
        for i in range(len(points)):
            points[i][0] += (np.random.rand() - 0.5) * 0.03
            points[i][1] += (np.random.rand() - 0.5) * 0.03
            
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 8: Circle with perturbed points for better spread
    def generate_perturbed_circle():
        points = []
        # Place points in a circle pattern
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.ones(16) * 0.35  # All same radius for now
        
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        
        # Add small perturbations to avoid degeneracy
        for i in range(len(points)):
            points[i][0] += (np.random.rand() - 0.5) * 0.05
            points[i][1] += (np.random.rand() - 0.5) * 0.05
            
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 9: More sophisticated grid with adaptive spacing
    def generate_adaptive_grid():
        points = []
        # Create a more irregular grid pattern to avoid symmetry
        for i in range(4):
            for j in range(4):
                if len(points) < 16:
                    # Vary spacing slightly for each position
                    x_base = j * 0.25
                    y_base = i * 0.25
                    x = x_base + (np.random.rand() - 0.5) * 0.06
                    y = y_base + (np.random.rand() - 0.5) * 0.06
                    points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 10: Improved initial configuration based on known good arrangements
    def generate_better_initial():
        # Create a configuration inspired by optimal point distributions
        # This uses a more balanced approach with good spacing
        points = []
        # Create a more uniform distribution
        for i in range(4):
            for j in range(4):
                x = (j + 0.5) / 4.0
                y = (i + 0.5) / 4.0
                # Add subtle jitter to avoid regular patterns
                x += (np.random.rand() - 0.5) * 0.05
                y += (np.random.rand() - 0.5) * 0.05
                points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 11: Fibonacci-based spiral for even better distribution
    def generate_fibonacci_layout():
        points = []
        n = 16
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(n):
            # Fibonacci spiral placement
            theta = i * 2 * np.pi / golden_ratio
            r = np.sqrt(i / (n - 1)) * 0.4  # Scale to fit within unit square
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Try multiple initial configurations and pick the best one
    initial_configs = [
        generate_hexagonal_layout(),
        generate_optimized_grid_layout(), 
        generate_circular_layout(),
        generate_golden_spiral_layout(),
        generate_advanced_layout(),
        generate_random_improved(),
        generate_hybrid_layout(),
        generate_perturbed_circle(),
        generate_adaptive_grid(),
        generate_better_initial(),
        generate_fibonacci_layout()
    ]
    
    best_initial_points = None
    best_min_dist = 0
    
    for config in initial_configs:
        # Evaluate this configuration
        try:
            min_dist, max_dist = compute_min_max_distances_vectorized(config)
        except:
            distances = pdist(config)
            min_dist = np.min(distances)
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_initial_points = config.copy()
    
    # Use the best initial configuration
    points = best_initial_points
    
    # Flatten for optimization
    x0 = points.flatten()
    
    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(32)]
    
    # Use a more sophisticated optimization approach
    start_time = time.time()
    
    # First, try a hybrid approach: global then local optimization
    best_result = None
    best_ratio = -np.inf
    
    # Reduced restarts to speed up optimization while maintaining quality
    restart_strategies = [
        # Strategy 1: Use the best configuration from initial tries
        lambda: best_initial_points.flatten(),
        # Strategy 2: Random with better bounds
        lambda: np.random.uniform(0.1, 0.9, 32),
        # Strategy 3: More diverse random with tighter bounds
        lambda: np.random.uniform(0.05, 0.95, 32),
        # Strategy 4: Fibonacci layout
        lambda: generate_fibonacci_layout().flatten(),
        # Strategy 5: Better initial configuration
        lambda: generate_better_initial().flatten()
    ]
    
    # Use only 3 restarts to save time but make them more effective
    for restart_idx in range(3):
        try:
            # Randomize seed for better exploration
            np.random.seed(42 + restart_idx * 100)
            
            # Select restart strategy
            x0_restart = restart_strategies[restart_idx]()
            
            # Use different optimization methods for better results
            if restart_idx < 2:
                # Early restarts: use L-BFGS-B for fine-tuning
                method = 'L-BFGS-B'
                max_iter = 400
            else:
                # Later restarts: use SLSQP for robustness
                method = 'SLSQP'  
                max_iter = 300
                
            # Use a more robust optimization approach with better parameters
            result = minimize(
                objective,
                x0_restart,
                method=method,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                # Ensure points are within bounds
                final_points = np.clip(final_points, 0, 1)
                
                # Compute actual distances
                try:
                    final_min, final_max = compute_min_max_distances_vectorized(final_points)
                except:
                    distances = pdist(final_points)
                    final_min = np.min(distances)
                    final_max = np.max(distances)
                    
                if final_max > 0:
                    ratio = final_min / final_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_result = result
                        
        except Exception as e:
            continue
    
    # If no good result found, fall back to simple approach with more iterations
    if best_result is None:
        # Simple approach with L-BFGS-B with more iterations
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        best_result = result
    
    end_time = time.time()
    
    # Extract optimized points
    optimized_points = best_result.x.reshape(-1, 2)
    
    # Ensure all points are within bounds
    optimized_points = np.clip(optimized_points, 0, 1)
    
    # Additional refinement step: try local optimization on the best solution
    try:
        # Create a more refined optimization starting from the best result
        refined_result = minimize(
            objective,
            optimized_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14},
            tol=1e-14
        )
        
        if refined_result.success:
            final_points = refined_result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            
            # Check if refinement improved the result
            try:
                final_min, final_max = compute_min_max_distances_vectorized(final_points)
            except:
                distances = pdist(final_points)
                final_min = np.min(distances)
                final_max = np.max(distances)
                
            if final_max > 0:
                ratio = final_min / final_max
                if ratio > best_ratio:
                    optimized_points = final_points
    except:
        pass
    
    return optimized_points


# EVOLVE-BLOCK-END
