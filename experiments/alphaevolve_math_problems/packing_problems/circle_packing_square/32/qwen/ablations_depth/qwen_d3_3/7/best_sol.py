# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math
from typing import Tuple
import random
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining multiple strategies: geometric initialization,
    evolutionary algorithm, and local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using hexagonal packing with refinement
    def initialize_circles_hexagonal():
        circles = []
        
        # Create a hexagonal lattice pattern
        rows = 6
        cols = 6
        spacing = 0.15
        
        # Generate hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset odd rows for hexagonal packing
                x = 0.05 + spacing * j
                y = 0.05 + spacing * (i + 0.5 * (j % 2))
                    
                # Only place if within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append([x, y, 0])
        
        # Fill remaining slots with strategic random positions
        while len(circles) < n:
            # Use more uniform distribution near edges and center
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            circles.append([x, y, 0])
            
        circles_array = np.array(circles[:n])
        
        # Set initial radii using a more sophisticated approach
        for i in range(n):
            # Find closest neighbors using KDTree for efficiency
            distances = []
            for j in range(n):
                if i != j:
                    dx = circles_array[i, 0] - circles_array[j, 0]
                    dy = circles_array[i, 1] - circles_array[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    distances.append((dist, j))
            
            # Sort by distance
            distances.sort()
            
            # Set radius based on boundary constraints and neighbor distances
            boundary_constraint = min(
                circles_array[i, 0], 1 - circles_array[i, 0],
                circles_array[i, 1], 1 - circles_array[i, 1]
            )
            
            # Conservative approach for initial radii
            if distances:
                min_neighbor_dist = distances[0][0]
                # Allow for more radius but keep it reasonable
                max_radius = min(boundary_constraint, min_neighbor_dist * 0.4)
            else:
                max_radius = boundary_constraint
                
            # Start with a reasonable initial radius
            circles_array[i, 2] = max(0.02, min(0.18, max_radius))
            
        return circles_array
    
    # Alternative initialization: more evenly distributed with better spacing
    def initialize_circles_alternative():
        circles = []
        
        # Create a more uniform distribution
        # First, create a grid of potential centers
        grid_size = 5
        spacing = 1.0 / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(circles) >= n:
                    break
                x = spacing * (j + 1)
                y = spacing * (i + 1)
                circles.append([x, y, 0])
        
        # Add some random positions to fill gaps
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            circles.append([x, y, 0])
            
        circles_array = np.array(circles[:n])
        
        # Set initial radii with focus on maximizing total area
        for i in range(n):
            # Find nearest neighbors
            distances = []
            for j in range(n):
                if i != j:
                    dx = circles_array[i, 0] - circles_array[j, 0]
                    dy = circles_array[i, 1] - circles_array[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    distances.append((dist, j))
            
            distances.sort()
            
            # Set radius based on boundary and neighbor constraints
            boundary_constraint = min(
                circles_array[i, 0], 1 - circles_array[i, 0],
                circles_array[i, 1], 1 - circles_array[i, 1]
            )
            
            # Try to set larger initial radii for better performance
            if distances:
                min_neighbor_dist = distances[0][0]
                max_radius = min(boundary_constraint, min_neighbor_dist * 0.35)
            else:
                max_radius = boundary_constraint
                
            # Use a more aggressive initial radius
            circles_array[i, 2] = max(0.03, min(0.2, max_radius))
            
        return circles_array
    
    # Multi-start optimization approach
    def multi_start_optimization(initial_circles):
        best_circles = initial_circles.copy()
        best_sum = np.sum(initial_circles[:, 2])
        
        # Try multiple optimization approaches
        for start_idx in range(3):  # Three different starting configurations
            if start_idx == 0:
                circles = initialize_circles_hexagonal()
            elif start_idx == 1:
                circles = initialize_circles_alternative()
            else:
                # Random initialization
                circles = np.random.rand(n, 3)
                circles[:, 0] = circles[:, 0] * 0.9 + 0.05  # Keep away from boundaries
                circles[:, 1] = circles[:, 1] * 0.9 + 0.05
                circles[:, 2] = np.minimum(0.2, circles[:, 2] * 0.1 + 0.02)  # Small initial radii
            
            # Run optimization with multiple methods
            optimized = optimize_with_multiple_methods(circles)
            
            current_sum = np.sum(optimized[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized.copy()
        
        return best_circles
    
    # Enhanced optimization with multiple methods
    def optimize_with_multiple_methods(circles_array):
        # Method 1: Trust-constr optimization
        try:
            optimized = optimize_trust_constr(circles_array)
            if np.sum(optimized[:, 2]) > np.sum(circles_array[:, 2]):
                circles_array = optimized
        except:
            pass
        
        # Method 2: SLSQP optimization  
        try:
            optimized = optimize_slsqp(circles_array)
            if np.sum(optimized[:, 2]) > np.sum(circles_array[:, 2]):
                circles_array = optimized
        except:
            pass
            
        # Method 3: Local refinement with binary search
        circles_array = local_refinement_binary_search(circles_array)
        
        return circles_array
    
    def optimize_trust_constr(circles_array):
        x0 = circles_array.flatten()
        
        # Define bounds more carefully
        bounds = []
        for i in range(n):
            bounds.append((0.001, 1 - 0.001))  # x bounds
            bounds.append((0.001, 1 - 0.001))  # y bounds
            bounds.append((0.001, 0.4))  # r bounds (reasonable upper bound)
        
        # Objective function
        def obj_func(x_flat):
            circles_reconstructed = x_flat.reshape(-1, 3)
            return -np.sum(circles_reconstructed[:, 2])
        
        # Constraint function - more efficient implementation
        def constraint_func(x_flat):
            circles_reconstructed = x_flat.reshape(-1, 3)
            constraints = []
            
            # Non-overlap constraints - only check close pairs using spatial indexing
            tree = cKDTree(circles_reconstructed[:, :2])
            pairs = tree.query_pairs(0.8)  # Only check nearby pairs
            
            for i, j in pairs:
                if i < j:  # Avoid duplicates
                    x1, y1, r1 = circles_reconstructed[i]
                    x2, y2, r2 = circles_reconstructed[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    constraints.append(distance - (r1 + r2))
            
            # Boundary constraints
            for i in range(n):
                x, y, r = circles_reconstructed[i]
                constraints.extend([
                    x - r,           # left constraint
                    1 - x - r,       # right constraint
                    y - r,           # bottom constraint
                    1 - y - r        # top constraint
                ])
            
            return np.array(constraints)
        
        # Constraints dict
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        result = minimize(
            obj_func,
            x0,
            method='trust-constr',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1500, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=lambda x: None
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            return circles_array
    
    def optimize_slsqp(circles_array):
        x0 = circles_array.flatten()
        
        # Define bounds
        bounds = []
        for i in range(n):
            bounds.append((0.001, 1 - 0.001))  # x bounds
            bounds.append((0.001, 1 - 0.001))  # y bounds
            bounds.append((0.001, 0.4))  # r bounds
        
        # Objective function
        def obj_func(x_flat):
            circles_reconstructed = x_flat.reshape(-1, 3)
            return -np.sum(circles_reconstructed[:, 2])
        
        # Constraint function
        def constraint_func(x_flat):
            circles_reconstructed = x_flat.reshape(-1, 3)
            constraints = []
            
            # Non-overlap constraints using spatial indexing
            tree = cKDTree(circles_reconstructed[:, :2])
            pairs = tree.query_pairs(0.8)  # Only check nearby pairs
            
            for i, j in pairs:
                if i < j:  # Avoid duplicates
                    x1, y1, r1 = circles_reconstructed[i]
                    x2, y2, r2 = circles_reconstructed[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    constraints.append(distance - (r1 + r2))
            
            # Boundary constraints
            for i in range(n):
                x, y, r = circles_reconstructed[i]
                constraints.extend([
                    x - r,
                    1 - x - r,
                    y - r,
                    1 - y - r
                ])
            
            return np.array(constraints)
        
        # Constraints dict
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        result = minimize(
            obj_func,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-7},
            callback=lambda x: None
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            return circles_array
    
    # Local refinement with binary search for maximum radius
    def local_refinement_binary_search(circles_array):
        max_iterations = 100
        for iteration in range(max_iterations):
            improved = False
            # Shuffle for better exploration
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                old_x, old_y, old_r = circles_array[i]
                
                # Compute maximum possible radius
                max_radius = min(
                    circles_array[i, 0], 1 - circles_array[i, 0],
                    circles_array[i, 1], 1 - circles_array[i, 1]
                )
                
                # Binary search for optimal radius
                low = old_r
                high = max_radius
                best_radius = old_r
                
                # Check for valid radius with neighbors
                def is_valid_radius(radius):
                    # Check against all neighbors
                    for j in range(n):
                        if i != j:
                            dx = circles_array[i, 0] - circles_array[j, 0]
                            dy = circles_array[i, 1] - circles_array[j, 1]
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance < (radius + circles_array[j, 2]):
                                return False
                    return True
                
                # Binary search
                while high - low > 0.00001:
                    mid = (low + high) / 2
                    if is_valid_radius(mid):
                        best_radius = mid
                        low = mid
                    else:
                        high = mid
                
                # Update if improvement
                if best_radius > old_r + 0.0001:
                    circles_array[i, 2] = best_radius
                    improved = True
            
            if not improved:
                break
                
        return circles_array
    
    # Validation and refinement
    def validate_and_refine(circles_array):
        # Ensure all constraints are satisfied
        tree = cKDTree(circles_array[:, :2])
        max_radius = np.max(circles_array[:, 2])
        pairs = tree.query_pairs(2 * max_radius)
        
        # Fix overlaps by reducing radii
        for i, j in pairs:
            if i < j:  # Avoid duplicate processing
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < (r1 + r2):
                    # Reduce both radii proportionally to fix overlap
                    overlap = (r1 + r2) - distance
                    reduction = overlap * 0.4
                    circles_array[i, 2] = max(0.001, circles_array[i, 2] - reduction)
                    circles_array[j, 2] = max(0.001, circles_array[j, 2] - reduction)
        
        # Ensure all circles respect boundary constraints
        for i in range(n):
            x, y, r = circles_array[i]
            # Clamp to valid range
            circles_array[i, 0] = np.clip(x, r, 1 - r)
            circles_array[i, 1] = np.clip(y, r, 1 - r)
            
        return circles_array
    
    # Try multiple initializations and select the best
    best_circles = None
    best_sum = 0
    
    for trial in range(3):
        if trial == 0:
            circles = initialize_circles_hexagonal()
        elif trial == 1:
            circles = initialize_circles_alternative()
        else:
            # Random initialization
            circles = np.random.rand(n, 3)
            circles[:, 0] = circles[:, 0] * 0.9 + 0.05
            circles[:, 1] = circles[:, 1] * 0.9 + 0.05
            circles[:, 2] = np.minimum(0.2, circles[:, 2] * 0.1 + 0.02)
        
        # Apply optimization
        optimized_circles = multi_start_optimization(circles)
        
        # Validate and refine
        final_circles = validate_and_refine(optimized_circles)
        
        current_sum = np.sum(final_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = final_circles.copy()
    
    # Final refinement pass
    if best_circles is not None:
        best_circles = local_refinement_binary_search(best_circles)
        best_circles = validate_and_refine(best_circles)
    
    return best_circles if best_circles is not None else initialize_circles_hexagonal()


# EVOLVE-BLOCK-END
