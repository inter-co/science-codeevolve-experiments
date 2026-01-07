# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time
from scipy.spatial.distance import cdist
import warnings
from itertools import combinations
from scipy.optimize import differential_evolution
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: improved initialization + advanced optimization with mathematical programming.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Improved initialization using more systematic hexagonal packing
    def initialize_better_config():
        # Start with a hexagonal lattice pattern which typically gives better initial configurations
        # Hexagonal packing is known to be optimal for dense arrangements
        
        # Try different hexagonal grid sizes systematically
        best_layout_score = -np.inf
        best_positions = None
        best_radii = None
        
        # Try various grid configurations that approximate hexagonal packing
        # Focus on configurations that can accommodate 32 circles
        grid_configs = [
            (6, 6),   # 6x6 grid (36 positions)
            (5, 7),   # 5x7 grid (35 positions)  
            (7, 5),   # 7x5 grid (35 positions)
            (4, 9),   # 4x9 grid (36 positions)
            (9, 4),   # 9x4 grid (36 positions)
            (5, 8),   # 5x8 grid (40 positions)
            (8, 5),   # 8x5 grid (40 positions)
        ]
        
        for rows, cols in grid_configs:
            if rows * cols < n:
                continue
                
            # Create hexagonal grid pattern with proper spacing
            positions = []
            hex_radius = 1.0  # Will scale appropriately later
            
            # Generate points in hexagonal pattern
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
                
                # Calculate initial radii based on proximity using more careful computation
                radii = np.full(n, 0.05)
                
                # For each circle, compute minimum distance to neighbors with better tolerance
                tree = cKDTree(positions)
                for i in range(n):
                    # Query with more neighbors to get better estimate
                    distances, indices = tree.query(positions[i], k=min(12, n), p=2)
                    if len(distances) > 1:
                        min_dist = np.min(distances[1:])
                        # Use a more conservative estimate for radius
                        radii[i] = min(0.2, min_dist / 2.0 * 0.9)
                
                # Compute layout score (sum of radii)
                layout_score = np.sum(radii)
                if layout_score > best_layout_score:
                    best_layout_score = layout_score
                    best_positions = positions.copy()
                    best_radii = radii.copy()
            
            if len(positions) >= n:
                break
        
        # If still no good configuration, fallback to better distribution
        if best_positions is None:
            # Use a more structured approach: spiral-like distribution with better spacing
            positions = np.zeros((n, 2))
            for i in range(n):
                # Spiral distribution with good coverage
                angle = 2 * np.pi * i / n
                radius = 0.4 * (1 - 0.8 * (i / n))  # Gradually decrease from center
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                # Add some jitter to improve distribution
                x += np.random.normal(0, 0.02)
                y += np.random.normal(0, 0.02)
                positions[i] = [x, y]
            
            # Ensure positions are within bounds
            positions[:, 0] = np.clip(positions[:, 0], 0.01, 0.99)
            positions[:, 1] = np.clip(positions[:, 1], 0.01, 0.99)
            
            radii = np.full(n, 0.05)
            tree = cKDTree(positions)
            for i in range(n):
                distances, indices = tree.query(positions[i], k=min(10, n), p=2)
                if len(distances) > 1:
                    min_dist = np.min(distances[1:])
                    radii[i] = min(0.2, min_dist / 2.0 * 0.9)
            best_positions = positions
            best_radii = radii
            
        return best_positions, best_radii
    
    # Improved constraint functions with better vectorization and numerical stability
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
        """Ensure no two circles overlap - optimized version with better numerical handling"""
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
        # Add small epsilon to avoid numerical issues
        epsilon = 1e-12
        constraints = distances_sq - radii_sums**2 + epsilon
        
        return constraints
    
    # Objective function (negative because we minimize)
    def objective(circles_flat):
        return -np.sum(circles_flat.reshape(-1, 3)[:, 2])
    
    # Gradient of objective function
    def grad_objective(circles_flat):
        grad = np.zeros_like(circles_flat)
        grad[2::3] = -1.0  # gradient w.r.t. radii
        return grad
    
    # Enhanced optimization with better strategies
    def enhanced_optimization(initial_circles, bounds, cons):
        """Use a combination of optimization strategies for better results"""
        best_result = None
        best_sum = -np.inf
        
        # Strategy 1: Global optimization with differential evolution for initial search
        try:
            # First, do a coarse global search
            de_bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
            result_de = differential_evolution(
                objective,
                de_bounds,
                args=(initial_circles,),
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7
            )
            
            if result_de.success:
                # Refine with local optimization starting from DE solution
                refined_result = minimize(
                    objective,
                    result_de.x,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
                )
                
                if refined_result.success:
                    current_sum = -refined_result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = refined_result
                        
        except Exception:
            pass
        
        # Strategy 2: Multiple local optimizations with different starting points
        if best_result is None:
            # Try several restarts with different initial configurations
            for restart in range(10):
                np.random.seed(42 + restart)
                perturbed = initial_circles.copy()
                
                # Perturb more carefully with adaptive step sizes
                for i in range(n):
                    # Smaller perturbations for positions
                    perturbed[i*3] += np.random.normal(0, 0.005)
                    perturbed[i*3 + 1] += np.random.normal(0, 0.005)
                    # Even smaller perturbations for radii
                    perturbed[i*3 + 2] += np.random.normal(0, 0.002)
                
                # Ensure bounds are respected
                for i in range(n):
                    perturbed[i*3] = np.clip(perturbed[i*3], 0.001, 0.999)
                    perturbed[i*3 + 1] = np.clip(perturbed[i*3 + 1], 0.001, 0.999)
                    perturbed[i*3 + 2] = np.clip(perturbed[i*3 + 2], 0.001, 0.499)
                
                # Try different optimization methods with different tolerances
                methods_to_try = [
                    ('trust-constr', {'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-9}),
                    ('SLSQP', {'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6})
                ]
                
                for method, options in methods_to_try:
                    try:
                        result = minimize(
                            objective,
                            perturbed,
                            method=method,
                            bounds=bounds,
                            constraints=cons,
                            options=options,
                            callback=lambda x: None
                        )
                        
                        if result.success:
                            current_sum = -result.fun
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_result = result
                    except Exception:
                        continue
        
        # Strategy 3: If no good result yet, use direct refinement of initial solution
        if best_result is None:
            try:
                # Direct refinement with trust-constr
                result = minimize(
                    objective,
                    initial_circles,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception:
                pass
        
        return best_result, best_sum
    
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
    
    # Enhanced optimization process
    try:
        best_result, best_sum = enhanced_optimization(initial_circles, bounds, cons)
        
        if best_result is not None:
            final_circles = best_result.x.reshape(-1, 3)
        else:
            # Fallback to initial solution if optimization failed
            final_circles = initial_circles.reshape(-1, 3)
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        final_circles = initial_circles.reshape(-1, 3)
    
    # Final validation and cleanup with stricter bounds
    validated_circles = []
    for i in range(n):
        x = max(0.001, min(0.999, final_circles[i, 0]))
        y = max(0.001, min(0.999, final_circles[i, 1]))
        r = max(0.001, min(0.499, final_circles[i, 2]))
        validated_circles.append([x, y, r])
    
    return np.array(validated_circles)


# EVOLVE-BLOCK-END
