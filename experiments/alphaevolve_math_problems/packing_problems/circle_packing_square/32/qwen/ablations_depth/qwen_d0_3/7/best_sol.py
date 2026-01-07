# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time
from scipy.spatial.distance import cdist
import warnings
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: improved initialization + advanced optimization with multiple strategies.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Improved initialization using hexagonal packing principles
    def initialize_better_config():
        # Start with a hexagonal lattice pattern which typically gives better initial configurations
        # Hexagonal packing is known to be optimal for dense arrangements
        
        # Try different hexagonal grid sizes
        best_layout_score = -np.inf
        best_positions = None
        best_radii = None
        
        # Try different grid configurations that approximate hexagonal packing
        grid_configs = [
            (6, 6),   # 6x6 grid
            (5, 7),   # 5x7 grid  
            (7, 5),   # 7x5 grid
            (4, 8),   # 4x8 grid
            (8, 4),   # 8x4 grid
            (5, 6),   # 5x6 grid
            (6, 5),   # 6x5 grid
        ]
        
        for rows, cols in grid_configs:
            if rows * cols < n:
                continue
                
            # Create hexagonal grid pattern
            positions = []
            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= n:
                        break
                    # Offset every other row for hexagonal packing
                    x_offset = (j + 0.5) / cols
                    if i % 2 == 1:
                        x_offset += 0.5 / cols
                    y = (i + 0.5) / rows
                    x = x_offset
                    positions.append([x, y])
                    
            if len(positions) >= n:
                positions = np.array(positions[:n])
                
                # Calculate initial radii based on proximity
                radii = np.full(n, 0.05)
                
                # For each circle, compute minimum distance to neighbors
                tree = cKDTree(positions)
                for i in range(n):
                    distances, indices = tree.query(positions[i], k=min(8, n), p=2)
                    if len(distances) > 1:
                        min_dist = np.min(distances[1:])
                        radii[i] = min(0.2, min_dist / 2.0)
                
                # Compute layout score (sum of radii)
                layout_score = np.sum(radii)
                if layout_score > best_layout_score:
                    best_layout_score = layout_score
                    best_positions = positions.copy()
                    best_radii = radii.copy()
            
            if len(positions) >= n:
                break
        
        # If still no good configuration, fallback to random but with better distribution
        if best_positions is None:
            np.random.seed(42)
            # Generate points with better spatial distribution using latin hypercube sampling concept
            positions = np.zeros((n, 2))
            for i in range(n):
                # Distribute more evenly rather than purely random
                x = (i % 8 + np.random.random()) / 8.0
                y = (i // 8 + np.random.random()) / 4.0
                positions[i] = [x, y]
            
            radii = np.full(n, 0.05)
            tree = cKDTree(positions)
            for i in range(n):
                distances, indices = tree.query(positions[i], k=min(8, n), p=2)
                if len(distances) > 1:
                    min_dist = np.min(distances[1:])
                    radii[i] = min(0.2, min_dist / 2.0)
            best_positions = positions
            best_radii = radii
            
        return best_positions, best_radii
    
    # Optimized constraint functions with better vectorization and error handling
    def containment_constraints(circles_flat):
        """Ensure all circles are fully contained within unit square"""
        positions = circles_flat.reshape(-1, 3)[:, :2]
        radii = circles_flat.reshape(-1, 3)[:, 2]
        
        # Vectorized containment constraints
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        r_coords = radii
        
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        # This gives us four constraints per circle
        constraints = np.concatenate([
            x_coords - r_coords,           # x - r >= 0
            1 - x_coords - r_coords,       # 1 - x - r >= 0
            y_coords - r_coords,           # y - r >= 0
            1 - y_coords - r_coords        # 1 - y - r >= 0
        ])
        return constraints
    
    def non_overlap_constraints(circles_flat):
        """Ensure no two circles overlap - optimized version"""
        positions = circles_flat.reshape(-1, 3)[:, :2]
        radii = circles_flat.reshape(-1, 3)[:, 2]
        
        # More efficient implementation using vectorized operations
        n_circles = len(positions)
        
        # Precompute all pairwise differences efficiently
        # Use broadcasting to compute all distances at once
        diff_x = positions[:, 0][:, np.newaxis] - positions[:, 0][np.newaxis, :]
        diff_y = positions[:, 1][:, np.newaxis] - positions[:, 1][np.newaxis, :]
        dist_sq = diff_x**2 + diff_y**2
        
        # Create mask for upper triangle (avoid double counting)
        mask = np.triu(np.ones((n_circles, n_circles), dtype=bool), k=1)
        
        # Get distances for non-diagonal elements
        distances_sq = dist_sq[mask]
        
        # Get corresponding radii sums
        radii_sums = (radii[:, np.newaxis] + radii[np.newaxis, :])[mask]
        
        # Constraint: distance^2 >= (r_i + r_j)^2 (non-overlap)
        constraints = distances_sq - radii_sums**2
        
        return constraints
    
    # Objective function (negative because we minimize)
    def objective(circles_flat):
        return -np.sum(circles_flat.reshape(-1, 3)[:, 2])
    
    # Gradient of objective function
    def grad_objective(circles_flat):
        grad = np.zeros_like(circles_flat)
        grad[2::3] = -1.0  # gradient w.r.t. radii
        return grad
    
    # Initial configuration
    positions, radii = initialize_better_config()
    initial_circles = np.column_stack([positions, radii]).flatten()
    
    # Set up bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: containment_constraints(x)},
        {'type': 'ineq', 'fun': lambda x: non_overlap_constraints(x)}
    ]
    
    # Optimize using multiple strategies and restarts
    try:
        best_result = None
        best_sum = -np.inf
        
        # Strategy 1: Direct optimization with multiple restarts
        for restart in range(15):  # Increased restarts for better exploration
            np.random.seed(42 + restart)
            perturbed = initial_circles.copy()
            
            # Perturb positions and radii with different magnitudes
            for i in range(n):
                # Smaller perturbations to avoid large jumps
                perturbed[i*3] += np.random.uniform(-0.01, 0.01)
                perturbed[i*3 + 1] += np.random.uniform(-0.01, 0.01)
                # Even smaller perturbations for radii
                perturbed[i*3 + 2] += np.random.uniform(-0.005, 0.005)
            
            # Ensure bounds are respected
            for i in range(n):
                perturbed[i*3] = np.clip(perturbed[i*3], 0.001, 0.999)
                perturbed[i*3 + 1] = np.clip(perturbed[i*3 + 1], 0.001, 0.999)
                perturbed[i*3 + 2] = np.clip(perturbed[i*3 + 2], 0.001, 0.499)
            
            # Try different optimization methods
            methods_to_try = ['trust-constr', 'SLSQP']
            
            for method in methods_to_try:
                try:
                    result = minimize(
                        objective,
                        perturbed,
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 2000, 'ftol': 1e-8, 'gtol': 1e-8},
                        callback=lambda x: None
                    )
                    
                    if result.success:
                        current_sum = -result.fun
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_result = result
                except Exception:
                    continue
        
        # Strategy 2: Try with different optimization parameters
        if best_result is None:
            # Use a simpler approach first with more aggressive constraints
            simple_perturbed = initial_circles.copy()
            # Make even smaller perturbations
            for i in range(n):
                simple_perturbed[i*3] += np.random.normal(0, 0.003)
                simple_perturbed[i*3 + 1] += np.random.normal(0, 0.003)
                simple_perturbed[i*3 + 2] += np.random.normal(0, 0.001)
            
            # Ensure bounds
            for i in range(n):
                simple_perturbed[i*3] = np.clip(simple_perturbed[i*3], 0.001, 0.999)
                simple_perturbed[i*3 + 1] = np.clip(simple_perturbed[i*3 + 1], 0.001, 0.999)
                simple_perturbed[i*3 + 2] = np.clip(simple_perturbed[i*3 + 2], 0.001, 0.499)
            
            try:
                result = minimize(
                    objective,
                    simple_perturbed,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-9},
                    callback=lambda x: None
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception:
                pass
        
        if best_result is not None:
            final_circles = best_result.x.reshape(-1, 3)
        else:
            # Fallback to initial solution if optimization failed
            final_circles = initial_circles.reshape(-1, 3)
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        final_circles = initial_circles.reshape(-1, 3)
    
    # Final validation and cleanup
    validated_circles = []
    for i in range(n):
        x = max(0.001, min(0.999, final_circles[i, 0]))
        y = max(0.001, min(0.999, final_circles[i, 1]))
        r = max(0.001, min(0.499, final_circles[i, 2]))
        validated_circles.append([x, y, r])
    
    return np.array(validated_circles)


# EVOLVE-BLOCK-END
