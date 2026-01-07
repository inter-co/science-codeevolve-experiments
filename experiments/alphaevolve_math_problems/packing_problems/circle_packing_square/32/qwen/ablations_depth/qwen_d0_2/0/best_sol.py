# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import math
from typing import Tuple
import random
from numba import jit

@jit(nopython=True)
def compute_distance_squared(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

@jit(nopython=True)
def check_overlap_fast(circles, i, j):
    x1, y1, r1 = circles[i]
    x2, y2, r2 = circles[j]
    dist_sq = compute_distance_squared(x1, y1, x2, y2)
    min_dist_sq = (r1 + r2)**2
    return dist_sq < min_dist_sq

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, efficient constraint handling,
    and robust optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using hexagonal packing with optimization
    def initialize_hexagonal_layout():
        circles = []
        
        # Try a better hexagonal packing pattern
        # For 32 circles, we can arrange in roughly 5x6 grid with hexagonal offset
        rows = 6
        cols = 6
        
        # Calculate spacing for hexagonal packing
        # We'll use a denser hexagonal arrangement
        spacing_x = 0.9 / cols  # Leave some margin
        spacing_y = 0.9 / rows  # Leave some margin
        
        # Try to optimize the hexagonal pattern
        hex_radius = 0.12  # Starting radius
        
        # Place circles in a hexagonal pattern
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Offset every other row for hexagonal packing
                offset = (i % 2) * spacing_x / 2
                x = 0.05 + (j + 1) * spacing_x + offset  # Add margin
                y = 0.05 + (i + 1) * spacing_y  # Add margin
                # Ensure circles fit within bounds
                x = max(hex_radius, min(1 - hex_radius, x))
                y = max(hex_radius, min(1 - hex_radius, y))
                circles.append([x, y, hex_radius])
                count += 1
            if count >= n:
                break
        
        # Fill remaining positions with small circles
        while len(circles) < n:
            # Place in corners and edges with small radii
            pos_idx = len(circles)
            x = 0.05 + (pos_idx % 4) * 0.225
            y = 0.05 + (pos_idx // 4) * 0.225
            r = 0.015
            circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    # Generate initial configuration
    circles = initialize_hexagonal_layout()
    
    # More efficient constraint checking with numba acceleration
    def compute_violations_fast(circles_flat):
        """Compute constraint violations efficiently using numba"""
        circles = circles_flat.reshape(-1, 3)
        
        # Bounds violations
        bounds_violations = []
        for i in range(len(circles)):
            x, y, r = circles[i]
            bounds_violations.extend([
                x - r,           # x >= r
                1 - x - r,       # x <= 1 - r
                y - r,           # y >= r
                1 - y - r        # y <= 1 - r
            ])
        
        # Non-overlap violations using efficient pairwise checks
        nonoverlap_violations = []
        # Use KDTree for neighbor search but also do direct checks for better control
        centers = circles[:, :2]
        
        # Build KDTree for fast neighbor search
        tree = cKDTree(centers)
        
        # For each circle, find neighbors and check constraints
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            
            # Find nearby circles (using radius-based search)
            nearby_indices = tree.query_ball_point([x1, y1], 2*(r1 + 0.05))
            
            for j in nearby_indices:
                if i >= j:  # Avoid double counting and self-checking
                    continue
                x2, y2, r2 = circles[j]
                dist_sq = compute_distance_squared(x1, y1, x2, y2)
                min_dist_sq = (r1 + r2)**2
                # Violation is negative when circles overlap
                nonoverlap_violations.append(dist_sq - min_dist_sq)
        
        return np.array(bounds_violations), np.array(nonoverlap_violations)
    
    # Optimization objective: minimize negative sum of radii (maximize sum)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we're minimizing
    
    # Constraint functions
    def bounds_constraint(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Circle center must be within bounds
            constraints.extend([
                x - r,           # x >= r
                1 - x - r,       # x <= 1 - r
                y - r,           # y >= r
                1 - y - r        # y <= 1 - r
            ])
        return np.array(constraints)
    
    def nonoverlap_constraint(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        # Use more efficient constraint checking
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                # Distance squared constraint: (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
                # For constraint satisfaction, we want (x1-x2)^2 + (y1-y2)^2 - (r1+r2)^2 >= 0
                dist_sq = compute_distance_squared(x1, y1, x2, y2)
                constraints.append(dist_sq - (r1+r2)**2)
        return np.array(constraints)
    
    # Use a more robust optimization approach with improved constraints
    def improved_optimization(initial_circles):
        # Flatten initial circles
        initial_flat = initial_circles.flatten()
        
        # Create bounds for each parameter (x, y, r)
        bounds = [(0, 1), (0, 1), (0.001, 0.45)] * n  # r bounded appropriately
        
        # Define constraints with proper bounds checking
        cons = [
            {'type': 'ineq', 'fun': bounds_constraint},
            {'type': 'ineq', 'fun': nonoverlap_constraint}
        ]
        
        # Try multiple optimization methods with different starting points
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_value = float('-inf')
        
        # Try multiple random restarts to avoid local minima
        for restart in range(5):
            # Add some randomness to the initial solution
            if restart > 0:
                # Slightly perturb the solution
                perturbed = initial_flat.copy()
                for i in range(0, len(perturbed), 3):
                    # Perturb x and y slightly, keep radius mostly same
                    perturbed[i] = max(0.01, min(0.99, perturbed[i] + np.random.normal(0, 0.005)))
                    perturbed[i+1] = max(0.01, min(0.99, perturbed[i+1] + np.random.normal(0, 0.005)))
                    # Slightly adjust radius
                    perturbed[i+2] = max(0.001, min(0.45, perturbed[i+2] + np.random.normal(0, 0.002)))
                initial_flat = perturbed
            
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        initial_flat,
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
                    )
                    
                    if result.success:
                        current_sum = -result.fun  # Convert back to positive sum
                        if current_sum > best_value:
                            best_value = current_sum
                            best_result = result
                except Exception:
                    continue
        
        if best_result is not None and best_result.success:
            return best_result.x.reshape(-1, 3)
        else:
            return initial_circles
    
    # Apply optimization
    circles = improved_optimization(circles)
    
    # Enhanced refinement using a more sophisticated local search
    def advanced_local_search(circles):
        best_circles = circles.copy()
        best_sum = np.sum(circles[:, 2])
        
        # Perform multiple rounds of local search
        for round_num in range(3):
            # Generate candidate solutions through small perturbations
            candidates = []
            
            # Create 20 candidate solutions
            for _ in range(20):
                candidate = circles.copy()
                # Make small random changes
                for i in range(len(candidate)):
                    # Randomly decide whether to modify this circle
                    if np.random.random() < 0.3:
                        # Small perturbation
                        candidate[i, 0] = max(0.01, min(0.99, candidate[i, 0] + np.random.normal(0, 0.005)))
                        candidate[i, 1] = max(0.01, min(0.99, candidate[i, 1] + np.random.normal(0, 0.005)))
                        # Keep radius mostly same but allow small changes
                        candidate[i, 2] = max(0.001, min(0.45, candidate[i, 2] + np.random.normal(0, 0.002)))
                
                # Ensure constraints are satisfied
                if is_valid_solution(candidate):
                    candidates.append(candidate)
            
            # Evaluate candidates and keep the best
            for candidate in candidates:
                candidate_sum = np.sum(candidate[:, 2])
                if candidate_sum > best_sum:
                    best_sum = candidate_sum
                    best_circles = candidate.copy()
        
        return best_circles
    
    # Helper function to validate a solution
    def is_valid_solution(circles):
        # Check bounds
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlaps
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = compute_distance_squared(x1, y1, x2, y2)
                min_dist_sq = (r1 + r2)**2
                if dist_sq < min_dist_sq:
                    return False
        return True
    
    # Apply advanced local search
    circles = advanced_local_search(circles)
    
    # Final optimization using a different strategy
    def final_optimization_step(circles):
        # Try to improve by solving a simplified problem with fewer variables
        # This is a coarse-grained approach that focuses on the most promising adjustments
        
        # Get current sum
        current_sum = np.sum(circles[:, 2])
        
        # Try to increase individual radii one by one
        improved = True
        attempts = 0
        max_attempts = 50
        
        while improved and attempts < max_attempts:
            improved = False
            attempts += 1
            
            # Try increasing each circle's radius slightly
            for i in range(len(circles)):
                original_radius = circles[i, 2]
                # Try a small increase
                new_radius = min(0.45, original_radius + 0.005)
                
                if new_radius > original_radius:
                    # Test if this change works
                    test_circles = circles.copy()
                    test_circles[i, 2] = new_radius
                    
                    # Check all constraints
                    if is_valid_solution(test_circles):
                        new_sum = np.sum(test_circles[:, 2])
                        if new_sum > current_sum:
                            circles = test_circles
                            current_sum = new_sum
                            improved = True
                            break
        
        return circles
    
    circles = final_optimization_step(circles)
    
    # Final cleanup with constraint enforcement
    def enforce_final_constraints(circles):
        # First, make sure all circles are within bounds
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Ensure bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[i] = [x, y, r]
        
        # Then, apply a final constraint fixing approach
        # This is a greedy approach to fix overlaps
        changed = True
        iterations = 0
        while changed and iterations < 50:
            changed = False
            iterations += 1
            
            # Check all pairs for overlaps
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist_sq = compute_distance_squared(x1, y1, x2, y2)
                    min_dist_sq = (r1 + r2)**2
                    
                    if dist_sq < min_dist_sq:
                        # Resolve by reducing the larger radius
                        if r1 >= r2:
                            circles[i, 2] = max(0.001, r1 - 0.001)
                        else:
                            circles[j, 2] = max(0.001, r2 - 0.001)
                        changed = True
                        break
                if changed:
                    break
        
        return circles
    
    circles = enforce_final_constraints(circles)
    
    return circles


# EVOLVE-BLOCK-END
