# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math
from typing import Tuple
import random
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a combination of geometric initialization and gradient-based optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Even better initialization using a proven pattern from literature
    def initialize_circles():
        # Create a better initial configuration based on known good packings
        # Using a pattern that's closer to optimal arrangements
        circles = []
        
        # Create a more refined hexagonal-like arrangement
        # This creates a pattern similar to what's known to work well for circle packing
        rows = 6
        cols = 6
        spacing_x = 0.16  # Slightly adjusted spacing
        spacing_y = 0.16
        
        # Create hexagonal grid with better distribution
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset odd rows for hexagonal packing
                x = 0.08 + spacing_x * (j + 0.5)
                y = 0.08 + spacing_y * (i + 0.5)
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                # Only place if within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append([x, y, 0])
        
        # Fill remaining slots with carefully placed points
        remaining_slots = n - len(circles)
        if remaining_slots > 0:
            # Place remaining circles in strategic positions
            for i in range(remaining_slots):
                # Place in a more central region with varied positions
                x = 0.2 + random.random() * 0.6
                y = 0.2 + random.random() * 0.6
                circles.append([x, y, 0])
            
        circles_array = np.array(circles[:n])
        
        # Set initial radii more aggressively
        for i in range(n):
            # Find closest neighbors to estimate minimum safe distance
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
            
            # Use a more aggressive approach for initial radii
            if distances:
                min_neighbor_dist = distances[0][0]
                # Allow for more aggressive radius assignment
                max_radius = min(boundary_constraint, min_neighbor_dist * 0.35)
            else:
                max_radius = boundary_constraint
                
            circles_array[i, 2] = max(0.01, min(0.18, max_radius))
            
        return circles_array
    
    # Initialize with better starting configuration
    circles = initialize_circles()
    
    # Enhanced optimization approach with better constraint handling
    def optimize_circles(circles_array):
        # Flatten the array for scipy optimization
        x0 = circles_array.flatten()
        
        # Define bounds for variables (x, y, r for each circle)
        bounds = []
        for i in range(n):
            # x bounds: [r, 1-r] 
            bounds.append((circles_array[i, 2], 1 - circles_array[i, 2]))
            # y bounds: [r, 1-r]
            bounds.append((circles_array[i, 2], 1 - circles_array[i, 2]))
            # r bounds: [0.001, 0.5] (reasonable range)
            bounds.append((0.001, 0.5))
        
        # Define the optimization problem
        def obj_func(x_flat):
            circles_reconstructed = x_flat.reshape(-1, 3)
            return -np.sum(circles_reconstructed[:, 2])  # Negative because we want to maximize
        
        def constraint_func(x_flat):
            circles_reconstructed = x_flat.reshape(-1, 3)
            constraints = []
            
            # Non-overlap constraints
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles_reconstructed[i]
                    x2, y2, r2 = circles_reconstructed[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    constraints.append(distance - (r1 + r2))  # Should be >= 0
            
            # Boundary constraints
            for i in range(n):
                x, y, r = circles_reconstructed[i]
                constraints.extend([
                    x - r,  # x >= r
                    1 - x - r,  # 1-x >= r
                    y - r,  # y >= r
                    1 - y - r   # 1-y >= r
                ])
            
            return np.array(constraints)
        
        # Create constraints dictionary
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        # Try different optimization methods to find the best solution
        methods_to_try = ['trust-constr', 'SLSQP']
        best_result = None
        best_value = float('inf')
        
        for method in methods_to_try:
            try:
                result = minimize(
                    obj_func,
                    x0,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-10},
                    callback=lambda x: None
                )
                
                if result.success:
                    # Check if this is better
                    current_sum = -result.fun
                    if current_sum < best_value:
                        best_value = current_sum
                        best_result = result
            except Exception:
                continue
        
        if best_result is not None and best_result.success:
            optimized_circles = best_result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Return original if optimization fails
            return circles_array
    
    # Run the optimization
    circles = optimize_circles(circles)
    
    # Final validation and refinement with more careful approach
    def validate_and_refine(circles_array):
        # Ensure all constraints are satisfied
        tree = cKDTree(circles_array[:, :2])
        max_radius = np.max(circles_array[:, 2])
        pairs = tree.query_pairs(2 * max_radius)
        
        # Fix overlaps by reducing radii carefully
        for i, j in pairs:
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
    
    circles = validate_and_refine(circles)
    
    # Enhanced aggressive refinement with multiple passes
    def enhanced_aggressive_refinement(circles_array):
        # Multiple passes of refinement for maximum improvement
        for pass_num in range(3):
            improved = True
            iteration_count = 0
            
            while improved and iteration_count < 150:
                improved = False
                iteration_count += 1
                
                # Randomly shuffle circle order for better exploration
                indices = list(range(n))
                random.shuffle(indices)
                
                for i in indices:
                    old_x, old_y, old_r = circles_array[i]
                    
                    # Try to increase radius while maintaining constraints
                    max_radius = min(
                        circles_array[i, 0], 1 - circles_array[i, 0],
                        circles_array[i, 1], 1 - circles_array[i, 1]
                    )
                    
                    # Binary search for maximum radius with very fine granularity
                    low = old_r
                    high = max_radius
                    best_radius = old_r
                    
                    while high - low > 0.000001:
                        mid = (low + high) / 2
                        test_radius = mid
                        
                        # Check if this radius works with neighbors
                        valid = True
                        for j in range(n):
                            if i != j:
                                dx = circles_array[i, 0] - circles_array[j, 0]
                                dy = circles_array[i, 1] - circles_array[j, 1]
                                distance = math.sqrt(dx*dx + dy*dy)
                                if distance < (test_radius + circles_array[j, 2]):
                                    valid = False
                                    break
                        
                        if valid:
                            best_radius = test_radius
                            low = test_radius
                        else:
                            high = test_radius
                    
                    # Update if we found a better radius
                    if best_radius > old_r + 0.00001:
                        circles_array[i, 2] = best_radius
                        improved = True
                        
                # Small position adjustments every few iterations
                if iteration_count % 10 == 0:
                    for i in range(n):
                        old_x, old_y, old_r = circles_array[i]
                        
                        # Try to slightly adjust position to improve packing
                        best_pos = [old_x, old_y]
                        best_sum = np.sum(circles_array[:, 2])
                        
                        # Test small movements in a more comprehensive way
                        movements = [
                            (-0.002, -0.002), (-0.002, 0), (-0.002, 0.002),
                            (0, -0.002), (0, 0.002),
                            (0.002, -0.002), (0.002, 0), (0.002, 0.002)
                        ]
                        
                        for dx, dy in movements:
                            test_x = old_x + dx
                            test_y = old_y + dy
                            
                            # Check if position is valid
                            if (test_x >= circles_array[i, 2] and 
                                test_x <= 1 - circles_array[i, 2] and
                                test_y >= circles_array[i, 2] and 
                                test_y <= 1 - circles_array[i, 2]):
                                
                                # Check if this improves the configuration
                                temp_circles = circles_array.copy()
                                temp_circles[i, 0] = test_x
                                temp_circles[i, 1] = test_y
                                
                                # Check constraints
                                valid = True
                                for j in range(n):
                                    if i != j:
                                        dx_test = temp_circles[i, 0] - temp_circles[j, 0]
                                        dy_test = temp_circles[i, 1] - temp_circles[j, 1]
                                        distance = math.sqrt(dx_test*dx_test + dy_test*dy_test)
                                        if distance < (temp_circles[i, 2] + temp_circles[j, 2]):
                                            valid = False
                                            break
                                
                                if valid:
                                    # Calculate new sum
                                    new_sum = np.sum(temp_circles[:, 2])
                                    if new_sum > best_sum:
                                        best_sum = new_sum
                                        best_pos = [test_x, test_y]
                        
                        circles_array[i, 0] = best_pos[0]
                        circles_array[i, 1] = best_pos[1]
                        
                        # Check if we made an improvement
                        if best_pos != [old_x, old_y]:
                            improved = True
        
        return circles_array
    
    circles = enhanced_aggressive_refinement(circles)
    
    # Final validation
    circles = validate_and_refine(circles)
    
    return circles


# EVOLVE-BLOCK-END
