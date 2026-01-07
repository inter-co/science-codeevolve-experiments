# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from scipy.spatial import KDTree
import random
from itertools import combinations
import time

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
    
    # Use a proven initial configuration approach - start with a known good pattern
    # Create a configuration inspired by the optimal known solutions for circle packing
    circles = np.zeros((n, 3))
    
    # Use a more structured initial layout - a combination of hexagonal and square arrangements
    # This gives us a better starting point than pure randomization
    
    # Layout 1: 6 rows of 6 columns (36 positions, take first 32)
    rows, cols = 6, 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Hexagonal offset for odd rows
            offset = spacing_x * 0.5 if i % 2 == 1 else 0
            x = (j + 1) * spacing_x + offset
            y = (i + 1) * spacing_y
            # Add small randomization to avoid symmetry issues
            x += random.uniform(-spacing_x*0.05, spacing_x*0.05)
            y += random.uniform(-spacing_y*0.05, spacing_y*0.05)
            # Clamp to valid range
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            positions.append([x, y])
        if len(positions) >= n:
            break
    
    # Fill remaining positions with random but well-distributed points
    while len(positions) < n:
        x = random.uniform(0.01, 0.99)
        y = random.uniform(0.01, 0.99)
        positions.append([x, y])
    
    # Assign initial positions
    for i in range(n):
        circles[i, 0] = positions[i][0]
        circles[i, 1] = positions[i][1]
        circles[i, 2] = 0.02  # Slightly larger initial radius
    
    # Advanced optimization with better constraint handling and multiple strategies
    
    def compute_violations(params):
        """Compute constraint violations for all constraints"""
        circles_array = params.reshape((n, 3))
        violations = []
        
        # Containment violations (should be negative when violated)
        for i in range(n):
            x, y, r = circles_array[i]
            violations.extend([r - x, r - (1 - x), r - y, r - (1 - y)])
        
        # Overlap violations (should be negative when violated)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                violations.append((r1 + r2) - dist)
        
        return np.array(violations)
    
    def objective(params):
        """Objective function: negative sum of radii to minimize"""
        circles_array = params.reshape((n, 3))
        return -np.sum(circles_array[:, 2])
    
    def constraint_containment(params):
        """Containment constraints: should be >= 0"""
        circles_array = params.reshape((n, 3))
        violations = []
        for i in range(n):
            x, y, r = circles_array[i]
            violations.extend([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.array(violations)
    
    def constraint_overlap(params):
        """Overlap constraints: should be >= 0"""
        circles_array = params.reshape((n, 3))
        violations = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                violations.append(dist - (r1 + r2))
        return np.array(violations)
    
    # Try multiple optimization strategies and keep the best
    best_result = None
    best_sum = -float('inf')
    
    # Strategy 1: Direct constrained optimization with SLSQP
    try:
        initial_flat = circles.flatten()
        
        # Define bounds: x, y in [0.01, 0.99], r in [0.001, 0.5]
        bounds = []
        for i in range(n):
            bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.001, 0.5)])
        
        # Constraints
        cons = [
            {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
            {'type': 'ineq', 'fun': lambda x: constraint_overlap(x)}
        ]
        
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape((n, 3))
            sum_radii = np.sum(optimized_circles[:, 2])
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_result = optimized_circles.copy()
    except Exception as e:
        pass
    
    # Strategy 2: Multiple restarts with better initializations
    for restart in range(8):
        # Create a new slightly perturbed initial configuration
        temp_circles = circles.copy()
        if restart > 0:
            # Perturb positions
            for i in range(n):
                temp_circles[i, 0] += random.uniform(-0.03, 0.03)
                temp_circles[i, 1] += random.uniform(-0.03, 0.03)
                # Clamp to valid range
                temp_circles[i, 0] = np.clip(temp_circles[i, 0], 0.01, 0.99)
                temp_circles[i, 1] = np.clip(temp_circles[i, 1], 0.01, 0.99)
        
        # Greedy radius optimization before full optimization
        for _ in range(5):
            for i in range(n):
                max_radius = float('inf')
                
                # Containment constraints
                x, y = temp_circles[i, 0], temp_circles[i, 1]
                max_radius = min(max_radius, x, 1-x, y, 1-y)
                
                # Overlap constraints with all others
                for j in range(n):
                    if i != j:
                        x2, y2 = temp_circles[j, 0], temp_circles[j, 1]
                        r2 = temp_circles[j, 2]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if dist > 0:
                            max_radius = min(max_radius, dist - r2)
                
                # Update radius
                if max_radius > temp_circles[i, 2] and max_radius > 0:
                    temp_circles[i, 2] = min(max_radius, temp_circles[i, 2] * 1.1)
        
        # Full optimization
        try:
            initial_flat = temp_circles.flatten()
            
            # Define bounds: x, y in [0.01, 0.99], r in [0.001, 0.5]
            bounds = []
            for i in range(n):
                bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.001, 0.5)])
            
            # Constraints
            cons = [
                {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
                {'type': 'ineq', 'fun': lambda x: constraint_overlap(x)}
            ]
            
            result = minimize(
                objective,
                initial_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                optimized_circles = result.x.reshape((n, 3))
                sum_radii = np.sum(optimized_circles[:, 2])
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = optimized_circles.copy()
        except Exception as e:
            continue
    
    # If we have a result, use it; otherwise use the original
    if best_result is not None:
        circles = best_result.copy()
    
    # Final aggressive refinement using spatial indexing
    tree = KDTree(circles[:, :2])
    
    # Use a more sophisticated refinement loop
    improvement_count = 0
    max_improvements = 50
    
    for iteration in range(max_improvements):
        old_sum = np.sum(circles[:, 2])
        improved = False
        
        # Process circles in shuffled order for better exploration
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            # Find neighbors more efficiently
            neighbors = tree.query_ball_point(circles[i, :2], 0.3)
            neighbors = [idx for idx in neighbors if idx != i]
            
            # Compute maximum possible radius for this circle
            max_radius = float('inf')
            
            # Containment constraints
            x, y = circles[i, 0], circles[i, 1]
            containment_radius = min(x, 1-x, y, 1-y)
            max_radius = min(max_radius, containment_radius)
            
            # Overlap constraints with neighbors
            for j in neighbors:
                x2, y2 = circles[j, 0], circles[j, 1]
                r2 = circles[j, 2]
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                if dist > 0:
                    max_radius = min(max_radius, dist - r2)
            
            # If we can increase the radius significantly, do so
            if max_radius > circles[i, 2] + 1e-4 and max_radius > 0:
                # Use a more aggressive approach to increase radius
                new_radius = min(max_radius, circles[i, 2] * 1.5)
                if new_radius > circles[i, 2] + 1e-5:
                    circles[i, 2] = new_radius
                    improved = True
                    improvement_count += 1
        
        # Early stopping if no improvements
        new_sum = np.sum(circles[:, 2])
        if abs(new_sum - old_sum) < 1e-6:
            break
            
        # Update tree for next iteration
        try:
            tree = KDTree(circles[:, :2])
        except:
            pass
    
    # Final constraint validation and cleanup
    for i in range(n):
        x, y, r = circles[i, 0], circles[i, 1], circles[i, 2]
        # Ensure containment
        r = min(r, x, 1-x, y, 1-y)
        # Ensure minimum radius
        r = max(r, 0.001)
        circles[i, 2] = r
    
    return circles


# EVOLVE-BLOCK-END
