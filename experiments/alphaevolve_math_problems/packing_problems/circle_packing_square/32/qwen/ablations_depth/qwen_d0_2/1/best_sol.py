# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math
from typing import Tuple

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
    
    # Better initialization using a more systematic approach
    def initialize_better_layout():
        # Start with a grid-like arrangement and then optimize
        circles = []
        
        # Grid parameters
        grid_size = int(math.ceil(math.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        radius = spacing * 0.4  # Initial radius
        
        # Place circles in a grid pattern
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n:
                    break
                x = (j + 1) * spacing
                y = (i + 1) * spacing
                circles.append([x, y, radius])
                count += 1
            if count >= n:
                break
        
        # Fill remaining positions with smaller circles near boundaries
        while len(circles) < n:
            # Place in corners with very small radii
            x = 0.05 + (len(circles) % 3) * 0.3
            y = 0.05 + (len(circles) // 3) * 0.3
            r = 0.02
            circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    # Generate initial configuration
    circles = initialize_better_layout()
    
    # More efficient constraint checking using spatial data structures
    def compute_constraints(circles_flat):
        """Compute all constraint violations efficiently"""
        circles = circles_flat.reshape(-1, 3)
        
        # Bounds constraints (each circle must be fully within square)
        bounds_violations = []
        for i in range(len(circles)):
            x, y, r = circles[i]
            bounds_violations.extend([
                x - r,           # x >= r
                1 - x - r,       # x <= 1 - r
                y - r,           # y >= r
                1 - y - r        # y <= 1 - r
            ])
        
        # Non-overlap constraints using KDTree for efficiency
        nonoverlap_violations = []
        centers = circles[:, :2]
        
        # Build KDTree for fast neighbor search
        tree = cKDTree(centers)
        
        # For each pair, check if they violate overlap constraint
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            
            # Find nearby circles (using radius-based search)
            nearby_indices = tree.query_ball_point([x1, y1], 2*(r1 + 0.1))
            
            for j in nearby_indices:
                if i >= j:  # Avoid double counting and self-checking
                    continue
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                nonoverlap_violations.append(dist_sq - min_dist_sq)
        
        return np.array(bounds_violations), np.array(nonoverlap_violations)
    
    # Optimization objective: minimize negative sum of radii (maximize sum)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we're minimizing
    
    # Constraints for optimization - more efficient version
    def constraint_bounds(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        # Each circle must be fully inside unit square
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
    
    def constraint_nonoverlap(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        # Check all pairs of circles for overlap
        constraints = []
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                # Distance squared constraint: (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                constraints.append(dist_sq - (r1+r2)**2)
        return np.array(constraints)
    
    # Set up optimization
    initial_flat = circles.flatten()
    
    # Use scipy.optimize with constraints - more robust approach
    try:
        # Create bounds for each parameter (x, y, r)
        bounds = [(0, 1), (0, 1), (0, 0.5)] * n  # r bounded by 0.5 to prevent overflow
        
        # Define constraints
        cons = [
            {'type': 'ineq', 'fun': lambda x: constraint_bounds(x)},
            {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
        ]
        
        # Try different optimization methods
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_value = float('-inf')
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_flat,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
                )
                
                if result.success:
                    current_sum = -result.fun  # Convert back to positive sum
                    if current_sum > best_value:
                        best_value = current_sum
                        best_result = result
            except Exception:
                continue
        
        if best_result is not None and best_result.success:
            circles = best_result.x.reshape(-1, 3)
            
    except Exception as e:
        # If optimization fails, use initial configuration
        pass
    
    # Enhanced refinement using simulated annealing approach
    def refine_with_local_search(circles):
        # More sophisticated local search
        best_circles = circles.copy()
        best_sum = np.sum(circles[:, 2])
        
        # Try various local improvements
        for iteration in range(500):
            improved = False
            # Randomly select a circle to modify
            i = np.random.randint(0, len(circles))
            
            # Save current state
            old_circles = circles.copy()
            old_sum = np.sum(circles[:, 2])
            
            # Try to increase radius of circle i
            x, y, r = circles[i]
            new_r = min(r + 0.005, 0.45)  # Limit max radius
            
            if new_r > r:
                # Test if this change is valid
                test_circles = circles.copy()
                test_circles[i, 2] = new_r
                
                # Check all constraints
                valid = True
                for j in range(len(test_circles)):
                    if j != i:
                        x1, y1, r1 = test_circles[i]
                        x2, y2, r2 = test_circles[j]
                        if (x1-x2)**2 + (y1-y2)**2 < (r1+r2)**2:
                            valid = False
                            break
                        # Check bounds
                        x, y, r = test_circles[j]
                        if x-r < 0 or x+r > 1 or y-r < 0 or y+r > 1:
                            valid = False
                            break
                
                if valid and np.sum(test_circles[:, 2]) > best_sum:
                    circles = test_circles
                    best_sum = np.sum(test_circles[:, 2])
                    improved = True
            
            # Try moving circle i to a better location
            if not improved:
                # Try small random moves
                dx = np.random.uniform(-0.01, 0.01)
                dy = np.random.uniform(-0.01, 0.01)
                
                test_circles = circles.copy()
                test_circles[i, 0] = max(0.01, min(0.99, test_circles[i, 0] + dx))
                test_circles[i, 1] = max(0.01, min(0.99, test_circles[i, 1] + dy))
                
                # Check constraints
                valid = True
                for j in range(len(test_circles)):
                    if j != i:
                        x1, y1, r1 = test_circles[i]
                        x2, y2, r2 = test_circles[j]
                        if (x1-x2)**2 + (y1-y2)**2 < (r1+r2)**2:
                            valid = False
                            break
                        # Check bounds
                        x, y, r = test_circles[j]
                        if x-r < 0 or x+r > 1 or y-r < 0 or y+r > 1:
                            valid = False
                            break
                
                if valid and np.sum(test_circles[:, 2]) > best_sum:
                    circles = test_circles
                    best_sum = np.sum(test_circles[:, 2])
                    improved = True
            
            # If we improved, update the best solution
            if improved:
                best_circles = circles.copy()
                best_sum = np.sum(circles[:, 2])
        
        return best_circles
    
    circles = refine_with_local_search(circles)
    
    # Final constraint enforcement
    for i in range(len(circles)):
        # Ensure circles stay within bounds
        x, y, r = circles[i]
        # Clamp coordinates to valid range
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
