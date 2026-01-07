# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, multi-start optimization, and 
    constraint satisfaction techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    random.seed(42)
    np.random.seed(42)
    
    # Improved initialization strategies
    def initialize_hexagonal():
        """Better hexagonal initialization with optimized spacing"""
        circles = []
        
        # Use a more systematic approach with optimal spacing for 32 circles
        max_rows = 6
        max_cols = 6
        
        # Calculate optimal spacing for maximum packing density
        spacing_x = 0.15  # Adjusted spacing
        spacing_y = 0.15 * math.sqrt(3)  # Vertical spacing for hexagonal
        
        # Place circles in a hexagonal pattern
        for i in range(max_rows):
            for j in range(max_cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Adjust for hexagonal offset
                if i % 2 == 1:
                    x += spacing_x / 2
                # Ensure we're within bounds
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, 0.05])
        
        # Fill remaining slots with carefully placed circles
        while len(circles) < n:
            # Try placing near edges but with better distribution
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Try to place at the boundary of other circles to encourage growth
            circles.append([x, y, 0.05])
            
        return np.array(circles[:n])
    
    def initialize_grid():
        """Grid-based initialization for better coverage"""
        circles = []
        # 5x7 grid approach
        rows, cols = 5, 7
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles.append([x, y, 0.05])
        
        # Fill remaining with random
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        return np.array(circles[:n])
    
    def initialize_random():
        """Pure random initialization"""
        circles = []
        for i in range(n):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
        return np.array(circles)
    
    # Enhanced constraint checking with early termination
    def is_valid(circles_array):
        if len(circles_array) == 0:
            return False
            
        # Check containment constraints
        for x, y, r in circles_array:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlap constraints efficiently with spatial indexing
        try:
            distances = cdist(circles_array[:, :2], circles_array[:, :2])
            for i in range(len(circles_array)):
                for j in range(i+1, len(circles_array)):
                    dist = distances[i, j]
                    if dist < circles_array[i, 2] + circles_array[j, 2]:
                        return False
        except:
            # Fallback if cdist fails
            for i, j in combinations(range(len(circles_array)), 2):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dx = x1 - x2
                dy = y1 - y2
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < r1 + r2:
                    return False
                    
        return True
    
    # Objective function with better numerical stability
    def objective(radii_and_positions):
        # Extract radii and positions
        radii = radii_and_positions[:n]
        positions = radii_and_positions[n:].reshape(-1, 2)
        
        # Objective: maximize sum of radii (negative for minimization)
        return -np.sum(radii)
    
    # More robust constraint function
    def constraints_func(radii_and_positions):
        # Extract data
        radii = radii_and_positions[:n]
        positions = radii_and_positions[n:].reshape(-1, 2)
        
        # Constraint 1: containment within unit square
        contain_constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # x >= r and 1-x >= r and y >= r and 1-y >= r
            contain_constraints.extend([
                x - r,  # x >= r
                1 - x - r,  # 1 - x >= r
                y - r,  # y >= r
                1 - y - r   # 1 - y >= r
            ])
        
        # Constraint 2: non-overlap (distance >= sum of radii)
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                min_distance = radii[i] + radii[j]
                # We want distance >= min_distance, so we add: distance - min_distance >= 0
                overlap_constraints.append(distance - min_distance)
        
        return np.array(contain_constraints + overlap_constraints)
    
    # Multi-start optimization approach - enhanced version
    best_result = None
    best_sum = 0
    
    # Try multiple starting points with different initialization strategies
    initial_strategies = [
        ("hexagonal", initialize_hexagonal),
        ("grid", initialize_grid),
        ("random", initialize_random)
    ]
    
    # Try multiple restarts with increased exploration and better convergence
    for start_iter in range(20):  # Even more restarts for better exploration
        # Select initialization strategy
        strategy_name, init_func = initial_strategies[start_iter % len(initial_strategies)]
        
        # Generate initial configuration
        initial_circles = init_func()
        
        # Add more randomness for later iterations to escape local optima
        if start_iter > 0:
            for i in range(n):
                # Larger perturbations for later iterations
                if start_iter < 10:
                    perturbation = 0.01
                else:
                    perturbation = 0.02
                initial_circles[i, 0] += np.random.normal(0, perturbation)
                initial_circles[i, 1] += np.random.normal(0, perturbation)
                # For radii, allow more variation for later iterations
                if start_iter < 10:
                    scale_factor = 0.1
                else:
                    scale_factor = 0.15
                initial_circles[i, 2] = max(0.01, min(0.45, initial_circles[i, 2] * np.random.normal(1, scale_factor)))
        
        # Set up optimization variables: [r1, r2, ..., rn, x1, y1, x2, y2, ...]
        initial_guess = np.concatenate([
            initial_circles[:, 2],  # initial radii
            initial_circles[:, :2].flatten()  # initial positions
        ])
        
        # Bounds for radii (positive) and positions (within unit square)
        bounds = []
        for i in range(n):
            bounds.append((0.001, 0.45))  # Radii bounds - tighter upper bound to avoid numerical issues
        for i in range(n):
            bounds.append((0.001, 0.999))  # x bounds
        for i in range(n):
            bounds.append((0.001, 0.999))  # y bounds
        
        # Optimize using SLSQP method with more robust settings
        try:
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints_func},
                options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-8, 'iprint': -1}  # Even tighter tolerances
            )
            
            if result.success:
                final_radii = result.x[:n]
                final_positions = result.x[n:].reshape(-1, 2)
                
                # Create final circles array
                circles = np.column_stack([
                    final_positions,
                    final_radii
                ])
                
                # Validate and refine
                if is_valid(circles):
                    sum_radii = np.sum(final_radii)
                    if sum_radii > best_sum:
                        best_sum = sum_radii
                        best_result = circles.copy()
            else:
                # Even if optimization fails, check the initial configuration
                if is_valid(initial_circles):
                    sum_radii = np.sum(initial_circles[:, 2])
                    if sum_radii > best_sum:
                        best_sum = sum_radii
                        best_result = initial_circles.copy()
                        
        except Exception as e:
            # If optimization fails, use the initial configuration
            if is_valid(initial_circles):
                sum_radii = np.sum(initial_circles[:, 2])
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = initial_circles.copy()
    
    # If we still don't have a good solution, fall back to initial configuration
    if best_result is None:
        initial_circles = initialize_hexagonal()
        # Final validation and refinement
        if is_valid(initial_circles):
            best_result = initial_circles
        else:
            # Last resort: fix any violations
            circles = initial_circles.copy()
            for i in range(n):
                x, y, r = circles[i]
                # Ensure containment
                r = min(r, x, 1-x, y, 1-y)
                circles[i] = [x, y, r]
            best_result = circles
    
    # Final refinement step - ensure final solution is valid
    final_circles = best_result.copy()
    
    # Make sure all circles are valid after optimization
    for i in range(n):
        x, y, r = final_circles[i]
        # Ensure containment constraints
        r = min(r, x, 1-x, y, 1-y)
        final_circles[i] = [x, y, r]
    
    # Double-check for overlaps and resolve if needed
    for _ in range(100):
        valid = True
        try:
            distances = cdist(final_circles[:, :2], final_circles[:, :2])
            for i in range(len(final_circles)):
                for j in range(i+1, len(final_circles)):
                    dist = distances[i, j]
                    if dist < final_circles[i, 2] + final_circles[j, 2]:
                        valid = False
                        # Reduce radii to resolve overlap
                        avg_radius = (final_circles[i, 2] + final_circles[j, 2]) / 2
                        final_circles[i, 2] = avg_radius * 0.95
                        final_circles[j, 2] = avg_radius * 0.95
        except:
            valid = False
            break
            
        if valid:
            break
    
    # Final boundary checks
    for i in range(n):
        x, y, r = final_circles[i]
        x = max(0.01, min(0.99, x))
        y = max(0.01, min(0.99, y))
        r = min(r, x, 1-x, y, 1-y)
        final_circles[i] = [x, y, r]
    
    return final_circles


# EVOLVE-BLOCK-END
