# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math
import random
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, advanced optimization, 
    and constraint validation for optimal results.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Generate initial configuration using hexagonal packing with boundary adjustments
    def generate_initial_config():
        """Generate a high-quality initial configuration using hexagonal packing principles"""
        # Create a hexagonal grid pattern
        rows = 6
        cols = 6
        if rows * cols < n:
            rows += 1
            
        # Create regular grid points with better spacing
        x_positions = np.linspace(0.05, 0.95, cols)
        y_positions = np.linspace(0.05, 0.95, rows)
        
        positions = []
        for i, y in enumerate(y_positions):
            for j, x in enumerate(x_positions):
                if len(positions) >= n:
                    break
                # Add slight offset for better packing (hexagonal pattern)
                offset_x = 0.03 * (i % 2) if i % 2 == 0 else 0.015
                offset_y = 0.03 * (j % 2) if j % 2 == 0 else 0.015
                positions.append([x + offset_x, y + offset_y])
        
        # Ensure we have exactly n points
        while len(positions) < n:
            positions.append([0.5, 0.5])
        
        positions = np.array(positions[:n])
        
        # Initialize with equal small radii
        radii = np.full(n, 0.04)
        
        # Adjust radii to respect boundary constraints
        for i in range(n):
            x, y = positions[i]
            # Maximum radius constrained by boundaries
            max_radius = min(x, y, 1-x, 1-y)
            radii[i] = min(radii[i], max_radius)
        
        return np.column_stack([positions, radii])
    
    # Fast constraint checking using vectorized operations and spatial indexing
    def check_constraints_fast(circles: np.ndarray) -> bool:
        """Efficient constraint checking using vectorized operations and spatial indexing"""
        if len(circles) == 0:
            return False
            
        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Check containment constraints (vectorized)
        containment_check = (
            (positions[:, 0] - radii >= 0) &
            (positions[:, 0] + radii <= 1) &
            (positions[:, 1] - radii >= 0) &
            (positions[:, 1] + radii <= 1)
        )
        
        if not np.all(containment_check):
            return False
        
        # Check overlap constraints efficiently using KDTree
        if len(circles) > 1:
            tree = cKDTree(positions)
            
            # For each circle, find neighbors within 2*max_radius distance
            max_radius = np.max(radii)
            for i in range(len(circles)):
                # Query nearby points (within 2*max_radius)
                neighbors = tree.query_ball_point(positions[i], 2 * max_radius)
                for j in neighbors:
                    if i != j:
                        dist_sq = np.sum((positions[i] - positions[j])**2)
                        r_sum = radii[i] + radii[j]
                        if dist_sq < r_sum * r_sum:
                            return False
        
        return True
    
    # Multi-step optimization approach inspired by INSPIRATION 2
    def optimize_solution(circles: np.ndarray) -> np.ndarray:
        """Apply multiple optimization steps to improve solution quality"""
        # Step 1: Local optimization using scipy minimize with constraints
        def optimize_with_slsqp(initial_circles):
            # Flatten for optimization
            initial_params = initial_circles.flatten()
            
            def objective(params):
                positions_and_radii = params.reshape(-1, 3)
                return -np.sum(positions_and_radii[:, 2])
            
            # Set bounds
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            # Constraint functions
            def constraint_containment(params):
                positions_and_radii = params.reshape(-1, 3)
                constraints = []
                for i in range(n):
                    x, y, r = positions_and_radii[i]
                    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
                    constraints.extend([
                        x - r,      # x - r >= 0
                        1 - x - r,  # 1 - x - r >= 0
                        y - r,      # y - r >= 0
                        1 - y - r   # 1 - y - r >= 0
                    ])
                return np.array(constraints)
            
            def constraint_nonoverlap(params):
                positions_and_radii = params.reshape(-1, 3)
                constraints = []
                positions = positions_and_radii[:, :2]
                radii = positions_and_radii[:, 2]
                
                # Vectorized non-overlap constraints
                if len(positions) > 1:
                    distances = cdist(positions, positions)
                    for i in range(len(positions)):
                        for j in range(i+1, len(positions)):
                            distance = distances[i, j]
                            r_sum = radii[i] + radii[j]
                            # Distance should be >= sum of radii (non-overlap)
                            constraints.append(distance - r_sum)
                
                return np.array(constraints)
            
            try:
                result = minimize(
                    objective,
                    initial_params,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[
                        {'type': 'ineq', 'fun': constraint_containment},
                        {'type': 'ineq', 'fun': constraint_nonoverlap}
                    ],
                    options={'maxiter': 200, 'ftol': 1e-6, 'eps': 1e-4}
                )
                
                if result.success:
                    return result.x.reshape(-1, 3)
            except:
                pass
            
            return initial_circles
        
        # Step 2: Differential evolution for global search
        def optimize_with_de(initial_circles):
            # Flatten parameters for optimization: [x0,y0,r0,x1,y1,r1,...]
            def flatten_params(circles_array):
                return circles_array.flatten()
            
            def unflatten_params(params):
                return params.reshape(-1, 3)
            
            # Define bounds for optimization
            bounds = []
            for i in range(n):
                # x bounds (slightly constrained to prevent extreme positions)
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            # Objective function (negative sum of radii for minimization)
            def objective(params):
                circles_flat = unflatten_params(params)
                return -np.sum(circles_flat[:, 2])
            
            try:
                # Use fewer iterations and higher population size for better exploration
                result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=100,
                    popsize=15,
                    mutation=(0.5, 1),
                    recombination=0.7,
                    seed=42,
                    disp=False,
                    atol=1e-6
                )
                
                if result.success:
                    optimized_circles = unflatten_params(result.x)
                    if check_constraints_fast(optimized_circles):
                        return optimized_circles
            except:
                pass
            
            return initial_circles
        
        # Apply sequential optimization
        circles = optimize_with_slsqp(circles)
        circles = optimize_with_de(circles)
        circles = optimize_with_slsqp(circles)
        
        return circles
    
    # Enhanced local refinement with better optimization
    def refine_solution(circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Apply enhanced local refinement to improve solution quality."""
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try to increase radii while maintaining constraints
            for i in range(n):
                old_radius = circles[i, 2]
                old_x, old_y = circles[i, 0], circles[i, 1]
                
                # Compute maximum possible radius
                max_radius = min(
                    old_x, old_y, 1-old_x, 1-old_y
                )
                
                # Find minimum distance to other circles
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                        min_dist = min(min_dist, dist)
                
                # Safe radius is limited by both boundary and overlap constraints
                safe_radius = min(max_radius, min_dist/2)
                
                # Try to increase radius (larger increments)
                if safe_radius > old_radius + 1e-5:
                    circles[i, 2] = safe_radius
                    improved = True
                    
                    # If we changed a radius, recheck constraints
                    if not check_constraints_fast(circles):
                        circles[i, 2] = old_radius
                        improved = False
            
            # Try more systematic position adjustments
            if improved:
                for i in range(n):
                    old_x, old_y, old_r = circles[i, 0], circles[i, 1], circles[i, 2]
                    
                    # Try more aggressive moves first, then smaller ones
                    moves = [
                        (-0.005, -0.005), (-0.005, 0), (-0.005, 0.005),
                        (0, -0.005), (0, 0.005),
                        (0.005, -0.005), (0.005, 0), (0.005, 0.005)
                    ]
                    
                    for dx, dy in moves:
                        test_x = old_x + dx
                        test_y = old_y + dy
                        
                        # Check bounds
                        if (test_x - old_r >= 0 and test_x + old_r <= 1 and 
                            test_y - old_r >= 0 and test_y + old_r <= 1):
                            
                            # Check overlap with others
                            valid_move = True
                            for j in range(n):
                                if i != j:
                                    dist = np.sqrt((test_x - circles[j, 0])**2 + (test_y - circles[j, 1])**2)
                                    if dist < (old_r + circles[j, 2]):
                                        valid_move = False
                                        break
                            
                            if valid_move:
                                circles[i, 0] = test_x
                                circles[i, 1] = test_y
                                improved = True
                                break
        
        return circles
    
    # Main execution flow with improved strategy
    # Step 1: Initialize with better hexagonal pattern from INSPIRATION 2
    circles = generate_initial_config()
    
    # Step 2: Apply advanced optimization with multiple techniques
    circles = optimize_solution(circles)
    
    # Step 3: Local refinement with more aggressive improvements
    circles = refine_solution(circles, max_iterations=50)
    
    # Step 4: Additional refinement passes
    for _ in range(2):
        circles = refine_solution(circles, max_iterations=30)
    
    # Step 5: Final constraint validation and correction
    if not check_constraints_fast(circles):
        # If constraints are violated, start over with better initialization
        circles = generate_initial_config()
        circles = optimize_solution(circles)
        circles = refine_solution(circles, max_iterations=50)
    
    return circles


# EVOLVE-BLOCK-END
