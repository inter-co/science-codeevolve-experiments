# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from numba import jit
from itertools import combinations
import random

@jit(nopython=True)
def distance_squared(x1, y1, x2, y2):
    """Fast squared distance calculation"""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy

@jit(nopython=True)
def check_overlap_fast(circles, i, j):
    """Fast overlap checking between two circles"""
    x1, y1, r1 = circles[i]
    x2, y2, r2 = circles[j]
    return distance_squared(x1, y1, x2, y2) < (r1 + r2) * (r1 + r2)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with mathematical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better geometric initialization using a more effective packing strategy
    def initialize_better_layout():
        # Use a hexagonal packing approach for better initial configuration
        # This is based on the fact that hexagonal packing is optimal for circles in 2D
        
        circles = []
        
        # Start with a regular hexagonal lattice pattern
        # Calculate parameters for hexagonal packing
        # For a hexagonal arrangement, the packing density is ~0.9069
        
        # Estimate radius based on total area needed
        # Area of all circles = n * pi * r^2
        # We want this to fit in unit square, so we need to estimate r
        target_area = 0.9 * 1.0  # 90% of square area for circles
        estimated_r = np.sqrt(target_area / (n * np.pi))
        
        # Create hexagonal grid with proper spacing
        hex_radius = estimated_r * 1.1  # slightly larger than just touching
        hex_spacing_x = hex_radius * 2
        hex_spacing_y = hex_radius * np.sqrt(3)
        
        # Create a grid that covers the square
        rows = int(np.ceil(1.0 / hex_spacing_y)) + 2
        cols = int(np.ceil(1.0 / hex_spacing_x)) + 2
        
        # Generate points in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                # Offset odd rows
                x_offset = (i % 2) * (hex_spacing_x / 2)
                x = x_offset + j * hex_spacing_x
                y = i * hex_spacing_y
                
                # Only consider points within the unit square
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Check if we have enough circles
                    if len(circles) < n:
                        # Ensure circle fits within bounds
                        r = min(hex_radius, x, 1-x, y, 1-y) * 0.95
                        if r > 0:
                            circles.append([x, y, r])
        
        # Fill remaining spots with random placements but with better initial radii
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Use a more informed radius selection
            r = np.random.uniform(0.01, 0.1)
            
            # Check if this circle would overlap with any existing circles
            valid = True
            for cx, cy, cr in circles:
                if distance_squared(x, y, cx, cy) < (r + cr) * (r + cr):
                    valid = False
                    break
            
            if valid:
                circles.append([x, y, r])
                
        return np.array(circles)
    
    # More efficient constraint functions
    def create_constraint_functions():
        # Create constraint functions that are more numerically stable
        def boundary_constraints(x):
            # For each circle, enforce: x-r >= 0, y-r >= 0, 1-x-r >= 0, 1-y-r >= 0
            constraints = []
            for i in range(n):
                x_i = x[3*i]
                y_i = x[3*i+1]
                r_i = x[3*i+2]
                constraints.extend([
                    x_i - r_i,           # x - r >= 0
                    y_i - r_i,           # y - r >= 0
                    1 - x_i - r_i,       # 1 - x - r >= 0
                    1 - y_i - r_i        # 1 - y - r >= 0
                ])
            return np.array(constraints)
        
        def overlap_constraints(x):
            # Non-overlap constraints: sqrt((x_i-x_j)^2 + (y_i-y_j)^2) >= r_i + r_j
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    x_i = x[3*i]
                    y_i = x[3*i+1]
                    r_i = x[3*i+2]
                    x_j = x[3*j]
                    y_j = x[3*j+1]
                    r_j = x[3*j+2]
                    
                    # This constraint should be >= 0 for non-overlap
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    radii_sum = r_i + r_j
                    
                    # We want sqrt(dist_sq) >= radii_sum, so dist_sq >= radii_sum^2
                    # Therefore: dist_sq - radii_sum^2 >= 0
                    constraints.append(dist_sq - radii_sum * radii_sum)
            return np.array(constraints)
        
        return boundary_constraints, overlap_constraints
    
    # Objective function to maximize (negative because minimize)
    def objective(x):
        return -np.sum(x[2::3])  # Sum of all radii (indices 2, 5, 8, ...)

    # Constraints wrapper that works better with scipy
    def get_constraints():
        cons = []
        
        # Boundary constraints
        for i in range(n):
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
            # y - r >= 0  
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
            # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
            # 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
        
        # Non-overlap constraints - use full set for better accuracy
        # But limit to reasonable number to keep computation manageable
        def nonoverlap_constraint(x, i, j):
            x_i = x[3*i]
            y_i = x[3*i+1]
            r_i = x[3*i+2]
            x_j = x[3*j]
            y_j = x[3*j+1]
            r_j = x[3*j+2]
            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            radii_sum = r_i + r_j
            return dist_sq - radii_sum * radii_sum
        
        # Create all pairwise constraints but limit the total number for performance
        # Use a smart approach: first compute distances, then only add constraints for close pairs
        # But for now, let's try with a more complete constraint set to improve quality
        constraint_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                constraint_pairs.append((i, j))
        
        # Limit to a reasonable number of constraints to avoid blowup
        # But still maintain good coverage
        max_constraints = min(len(constraint_pairs), 500)
        selected_pairs = constraint_pairs[:max_constraints]
        
        for i, j in selected_pairs:
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: nonoverlap_constraint(x, i, j)
            })
        
        return cons
    
    # Multi-start optimization approach
    def multi_start_optimization(initial_guesses):
        best_result = None
        best_sum = -np.inf
        
        # Try multiple starting points
        for i, initial_guess in enumerate(initial_guesses):
            try:
                # Set bounds for variables (x, y, r) for each circle
                bounds = []
                for j in range(n):
                    bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
                
                # Get constraints
                constraints = get_constraints()
                
                # Run optimization
                result = minimize(
                    objective,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-6}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
                        
            except Exception as e:
                continue  # Skip failed optimizations
        
        return best_result
    
    # Generate multiple initial guesses
    initial_guesses = []
    for _ in range(5):  # Try 5 different initializations
        initial_circles = initialize_better_layout()
        initial_guesses.append(initial_circles.flatten())
    
    # Run multi-start optimization
    try:
        result = multi_start_optimization(initial_guesses)
        
        if result and result.success:
            final_circles = result.x.reshape(-1, 3)
            return final_circles
        else:
            # Fallback to best initial configuration if optimization fails
            initial_circles = initialize_better_layout()
            return initial_circles
            
    except Exception as e:
        # Return initial configuration if anything goes wrong
        initial_circles = initialize_better_layout()
        return initial_circles


# EVOLVE-BLOCK-END
