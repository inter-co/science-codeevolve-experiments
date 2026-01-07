# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math
import random
from typing import Tuple, List

# Fast constraint checking using spatial data structures
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

def generate_initial_config(n: int = 32) -> np.ndarray:
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

def optimize_local(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Perform local optimization on circle positions/radii using scipy"""
    n = len(circles)
    
    # Flatten for optimization
    initial_params = circles.flatten()
    
    def objective(params):
        positions_and_radii = params.reshape(-1, 3)
        return -np.sum(positions_and_radii[:, 2])
    
    def constraint_containment(params):
        positions_and_radii = params.reshape(-1, 3)
        constraints = []
        for i in range(n):
            x, y, r = positions_and_radii[i]
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
                    constraints.append(distance - r_sum)
        
        return np.array(constraints)
    
    # Set bounds
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
                {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
            ],
            options={'maxiter': max_iter, 'ftol': 1e-6, 'eps': 1e-4}
        )
        
        if result.success:
            return result.x.reshape(-1, 3)
    except:
        pass
    
    return circles

def refine_solution(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Apply enhanced local refinement to improve solution quality."""
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try to increase radii while maintaining constraints
        for i in range(len(circles)):
            old_radius = circles[i, 2]
            old_x, old_y = circles[i, 0], circles[i, 1]
            
            # Compute maximum possible radius
            max_radius = min(
                old_x, old_y, 1-old_x, 1-old_y
            )
            
            # Find minimum distance to other circles
            min_dist = float('inf')
            for j in range(len(circles)):
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
            for i in range(len(circles)):
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
                        for j in range(len(circles)):
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
    
    # Step 1: Generate initial configuration using hexagonal packing
    circles = generate_initial_config(n)
    
    # Step 2: Multiple rounds of optimization
    for _ in range(3):
        circles = optimize_local(circles, max_iter=200)
        circles = refine_solution(circles, max_iterations=30)
    
    # Step 3: Final constraint validation and correction
    if not check_constraints_fast(circles):
        # If constraints are violated, start over with better initialization
        circles = generate_initial_config(n)
        circles = optimize_local(circles, max_iter=300)
        circles = refine_solution(circles, max_iterations=50)
    
    # Final optimization pass
    circles = optimize_local(circles, max_iter=500)
    
    return circles


# EVOLVE-BLOCK-END
