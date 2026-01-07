# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from scipy.spatial import KDTree
import random
from itertools import combinations

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with advanced optimization techniques.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using hexagonal packing approximation for better density
    circles = np.zeros((n, 3))
    
    # Create a hexagonal lattice pattern as initial configuration
    # This provides a good starting point for dense packing
    rows = 6
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            offset = spacing_x * 0.5 if i % 2 == 1 else 0
            x = (j + 1) * spacing_x + offset + random.uniform(-spacing_x*0.05, spacing_x*0.05)
            y = (i + 1) * spacing_y + random.uniform(-spacing_y*0.05, spacing_y*0.05)
            circles[idx] = [x, y, 0.0]  # Initialize with zero radius
            idx += 1
        if idx >= n:
            break
    
    # Set initial radii based on available space
    min_radius = 0.02
    for i in range(n):
        circles[i][2] = min_radius
    
    # Improved optimization approach using a more sophisticated method
    # Use differential evolution for global optimization, then local refinement
    
    def compute_constraints(circles_array):
        """Compute all constraint violations"""
        violations = []
        n = len(circles_array)
        
        # Containment constraints (should be positive for valid constraints)
        for i in range(n):
            x, y, r = circles_array[i]
            violations.append(x - r)  # Should be >= 0
            violations.append(1 - x - r)  # Should be >= 0
            violations.append(y - r)  # Should be >= 0
            violations.append(1 - y - r)  # Should be >= 0
            
        # Overlap constraints (should be positive for valid constraints)
        for i, j in combinations(range(n), 2):
            x1, y1, r1 = circles_array[i]
            x2, y2, r2 = circles_array[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            violations.append(dist - (r1 + r2))  # Should be >= 0
            
        return np.array(violations)
    
    def objective_function(circles_array):
        """Objective function to maximize sum of radii (negative for minimization)"""
        return -np.sum(circles_array[:, 2])
    
    def constraint_function(circles_array):
        """Constraint function returning positive values when satisfied"""
        constraints = compute_constraints(circles_array)
        # Return negative because scipy expects constraints to be >= 0
        return -constraints
    
    def optimize_with_bounds(circles_array, max_iter=100):
        """Optimize using SLSQP with proper bounds and constraints"""
        n = len(circles_array)
        
        # Flatten for optimization
        initial_flat = circles_array.flatten()
        
        def obj_flat(flat_params):
            circles_reconstructed = flat_params.reshape((n, 3))
            return -np.sum(circles_reconstructed[:, 2])
        
        # Constraints for SLSQP
        def containment_constraint(flat_params):
            circles_reconstructed = flat_params.reshape((n, 3))
            violations = []
            for i in range(n):
                x, y, r = circles_reconstructed[i]
                violations.extend([x - r, 1 - x - r, y - r, 1 - y - r])
            return np.array(violations)
        
        def overlap_constraint(flat_params):
            circles_reconstructed = flat_params.reshape((n, 3))
            violations = []
            for i, j in combinations(range(n), 2):
                x1, y1, r1 = circles_reconstructed[i]
                x2, y2, r2 = circles_reconstructed[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                violations.append(dist - (r1 + r2))
            return np.array(violations)
        
        # Combine constraints
        def combined_constraint(flat_params):
            return np.concatenate([
                containment_constraint(flat_params),
                overlap_constraint(flat_params)
            ])
        
        # Bounds: x, y in [0.001, 0.999], r in [0.001, 0.5]
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])
        
        # Constraint definitions
        cons = [
            {'type': 'ineq', 'fun': lambda x: containment_constraint(x)},
            {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
        ]
        
        try:
            result = minimize(
                obj_flat,
                initial_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': 1e-6, 'eps': 1e-6},
                callback=None
            )
            
            if result.success:
                return result.x.reshape((n, 3))
            else:
                return circles_array
                
        except Exception as e:
            return circles_array
    
    # Phase 1: Global optimization with differential evolution
    # Use a simpler approach with better constraint handling
    
    # Phase 2: Local optimization with better constraint management
    circles = optimize_with_bounds(circles, max_iter=500)
    
    # Phase 3: Iterative refinement with better spatial indexing
    # Create KDTree for efficient neighbor queries
    tree = KDTree(circles[:, :2])
    
    # Refinement loop with improved logic
    max_iterations = 50
    improvement_threshold = 1e-5
    
    for iteration in range(max_iterations):
        old_sum = np.sum(circles[:, 2])
        improved = False
        
        # Try to improve each circle individually
        for i in range(n):
            # Find neighbors within a reasonable distance
            neighbors = tree.query_ball_point(circles[i, :2], 0.3)
            neighbors = [idx for idx in neighbors if idx != i]
            
            # Compute maximum possible radius for this circle
            max_radius = float('inf')
            
            # Check containment constraints
            x, y = circles[i, 0], circles[i, 1]
            containment_radius = min(x, 1-x, y, 1-y)
            max_radius = min(max_radius, containment_radius)
            
            # Check overlap constraints with neighbors
            for j in neighbors:
                x2, y2 = circles[j, 0], circles[j, 1]
                r2 = circles[j, 2]
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                if dist > 0:  # Avoid division by zero
                    max_radius = min(max_radius, dist - r2)
            
            # If we can increase the radius, do so
            if max_radius > circles[i, 2] and max_radius > 0:
                # Try to increase radius more aggressively
                new_radius = min(max_radius, circles[i, 2] * 1.5)
                if new_radius > circles[i, 2] + improvement_threshold:
                    circles[i, 2] = new_radius
                    improved = True
        
        # Early termination if no significant improvement
        new_sum = np.sum(circles[:, 2])
        if abs(new_sum - old_sum) < improvement_threshold:
            break
            
        # Update the tree after changes
        try:
            tree = KDTree(circles[:, :2])
        except:
            pass
    
    # Final constraint validation and correction
    for i in range(n):
        x, y, r = circles[i, 0], circles[i, 1], circles[i, 2]
        # Ensure containment constraints
        r = min(r, x, 1-x, y, 1-y)
        # Ensure non-negative radius
        r = max(r, 0.001)
        circles[i, 2] = r
    
    # Final optimization step with better convergence
    circles = optimize_with_bounds(circles, max_iter=200)
    
    return circles


# EVOLVE-BLOCK-END
