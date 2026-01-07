# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math
from numba import jit
from joblib import Parallel, delayed
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
    
    # Improved geometric initialization using a more effective packing strategy
    def initialize_better_layout():
        # Start with a hexagonal packing pattern for better density
        circles = []
        
        # Hexagonal lattice parameters
        # For hexagonal packing, we need to calculate appropriate spacing
        # Let's start with a more systematic approach
        rows = 6
        cols = 6
        
        # Calculate spacing for hexagonal packing
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Hexagonal packing with offset rows
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset odd rows
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + 1) * spacing_x + x_offset * spacing_x * 0.5
                y = (i + 1) * spacing_y
                
                # Initial radius - smaller than spacing to allow for better packing
                r = spacing_x * 0.25
                
                # Ensure it's within bounds
                if x + r <= 1 and y + r <= 1 and x - r >= 0 and y - r >= 0:
                    circles.append([x, y, r])
        
        # Fill remaining spots with strategic random placements
        while len(circles) < n:
            # Try to place in less crowded areas
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            
            # Estimate good radius based on proximity to existing circles
            min_dist = float('inf')
            for cx, cy, _ in circles:
                dist = distance_squared(x, y, cx, cy)
                if dist < min_dist:
                    min_dist = dist
            
            # Radius should be inversely proportional to density
            if min_dist > 0:
                r = min(0.1, 0.05 * np.sqrt(min_dist))  # Cap max radius
            else:
                r = np.random.uniform(0.01, 0.05)
            
            # Ensure reasonable minimum radius
            r = max(r, 0.005)
            
            # Check if this circle would overlap with any existing circles
            valid = True
            for cx, cy, cr in circles:
                if distance_squared(x, y, cx, cy) < (r + cr) * (r + cr):
                    valid = False
                    break
            
            if valid:
                circles.append([x, y, r])
                
        return np.array(circles)
    
    # More efficient constraint functions with spatial indexing
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
        
        # Boundary constraints - more robust version
        for i in range(n):
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
            # y - r >= 0  
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
            # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
            # 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
        
        # Non-overlap constraints using spatial indexing for efficiency
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
        
        # Use a smarter constraint selection approach
        # Create a KDTree for efficient neighbor search
        def get_relevant_constraints(x):
            # Convert to array format for tree construction
            coords = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
            tree = cKDTree(coords)
            
            # Find nearby points (within a reasonable distance)
            pairs = tree.query_pairs(0.2, predicate=lambda u, v: u != v)
            relevant_cons = []
            
            for i, j in pairs:
                relevant_cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: nonoverlap_constraint(x, i, j)
                })
            
            return relevant_cons
        
        # Add a subset of constraints instead of all pairs
        # This makes optimization more tractable
        constraint_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                if j < i + 15:  # Limit to nearby circles
                    constraint_pairs.append((i, j))
        
        for i, j in constraint_pairs:
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: nonoverlap_constraint(x, i, j)
            })
        
        return cons
    
    # Multi-start optimization to avoid local optima
    def multi_start_optimization(initial_guesses):
        best_result = None
        best_sum = 0
        
        for i, initial_guess in enumerate(initial_guesses):
            try:
                # Set bounds for variables (x, y, r) for each circle
                bounds = []
                for j in range(n):
                    bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                
                # Get constraints for this specific initial guess
                constraints = get_constraints()
                
                # Try optimization with different methods
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
                continue
        
        return best_result
    
    # Generate multiple initial guesses for multi-start approach
    def generate_initial_guesses():
        initial_guesses = []
        for _ in range(5):  # 5 different starting points
            initial_circles = initialize_better_layout()
            initial_guesses.append(initial_circles.flatten())
        return initial_guesses
    
    # Initial guess generation
    initial_guesses = generate_initial_guesses()
    
    # Run multi-start optimization
    try:
        result = multi_start_optimization(initial_guesses)
        
        if result is not None and result.success:
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
