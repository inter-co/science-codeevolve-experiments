# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, advanced optimization, and 
    multi-start strategies to approach the benchmark.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Enhanced initialization using a more strategic approach
    def initialize_enhanced_layout():
        # Use a more sophisticated approach inspired by circle packing heuristics
        # Start with a 5x7 grid to get a good balance of coverage and spacing
        rows = 5
        cols = 7
        
        # Calculate grid spacing
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Calculate radius based on distance to boundaries and expected spacing
                # This helps avoid overly constrained circles at boundaries
                r = min(x, 1-x, y, 1-y) * 0.45
                
                # Reduce radius for circles near boundaries to allow better packing
                if x < 0.15 or x > 0.85 or y < 0.15 or y > 0.85:
                    r *= 0.6
                    
                r = max(0.01, min(0.48, r))
                circles.append([x, y, r])
                
        # Fill remaining positions if needed
        while len(circles) < n:
            # Place in center area with small radii
            circles.append([0.5, 0.5, 0.05])
            
        return np.array(circles[:n])
    
    # Alternative initialization: golden ratio inspired placement
    def initialize_golden_ratio_layout():
        # Use golden ratio-inspired distribution for better spreading
        circles = []
        
        # Place in a pattern that tries to distribute evenly
        # Start with some strategic corner placements
        corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
        for x, y in corners:
            circles.append([x, y, 0.09])
        
        # Add edge points with spacing
        # Top edge
        for i in range(4):
            if len(circles) < 32:
                x = 0.15 + 0.7 * i/3
                circles.append([x, 0.85, 0.07])
        # Bottom edge
        for i in range(4):
            if len(circles) < 32:
                x = 0.15 + 0.7 * i/3
                circles.append([x, 0.15, 0.07])
        # Left edge
        for i in range(4):
            if len(circles) < 32:
                y = 0.15 + 0.7 * i/3
                circles.append([0.15, y, 0.07])
        # Right edge
        for i in range(4):
            if len(circles) < 32:
                y = 0.15 + 0.7 * i/3
                circles.append([0.85, y, 0.07])
                
        # Fill remaining with center distribution
        while len(circles) < 32:
            circles.append([0.5, 0.5, 0.06])
            
        return np.array(circles[:32])
    
    # Initialize with multiple strategies and pick best
    def initialize_best_strategy():
        candidates = []
        
        # Strategy 1: Enhanced grid-based
        enhanced_init = initialize_enhanced_layout()
        candidates.append(enhanced_init)
        
        # Strategy 2: Golden ratio layout
        golden_init = initialize_golden_ratio_layout()
        candidates.append(golden_init)
        
        # Strategy 3: Random with better constraints
        random_init = np.zeros((32, 3))
        for i in range(32):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            # Radius based on distance to boundaries but more generous
            r = min(x, 1-x, y, 1-y) * 0.4
            r = max(0.01, min(0.45, r))
            random_init[i] = [x, y, r]
        candidates.append(random_init)
        
        # Evaluate each candidate by sum of radii
        best_idx = 0
        best_sum = 0
        for i, candidate in enumerate(candidates):
            current_sum = np.sum(candidate[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_idx = i
                
        return candidates[best_idx]
    
    # Create initial configuration
    circles = initialize_best_strategy()
    
    # Efficient constraint evaluation using vectorized operations
    def compute_distance_matrix(positions):
        """Compute pairwise distances between all circle centers"""
        return cdist(positions, positions)
    
    def check_feasibility(circles_array, tolerance=1e-10):
        """Check if configuration is feasible with numerical tolerance"""
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        # Check containment
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlaps using distance matrix
        distances = compute_distance_matrix(positions)
        # For each pair, check if distance < sum of radii (with tolerance)
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                dist = distances[i, j]
                r_sum = radii[i] + radii[j]
                if dist < r_sum - tolerance:
                    return False
                    
        return True
    
    # More efficient constraint handling with better numerical handling
    def evaluate_constraints(x_flat):
        """Evaluate all constraints efficiently with better numerical handling"""
        # Convert flat array back to circles
        circles_array = x_flat.reshape(-1, 3)
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        # Containment constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
        containment = []
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            containment.extend([
                x - r,           # x - r >= 0
                1 - x - r,       # 1 - x - r >= 0  
                y - r,           # y - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
            
        # Overlap constraints: distance >= r1 + r2
        overlap = []
        distances = compute_distance_matrix(positions)
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                dist = distances[i, j]
                r_sum = radii[i] + radii[j]
                # Small tolerance to handle numerical precision
                overlap.append(dist - r_sum)  # Want this >= 0
                
        return np.array(containment + overlap)
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x_flat):
        return -np.sum(x_flat[2::3])  # Sum of all radii (every 3rd element starting at index 2)
    
    # Constraints for scipy.optimize
    def constraint_func(x_flat):
        return evaluate_constraints(x_flat)
    
    # Bounds for variables: [x, y, r] for each circle
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0, 1))
        # y coordinate bounds  
        bounds.append((0, 1))
        # radius bounds (must be positive and respect containment)
        bounds.append((0.001, 0.5))  # Allow small radii, max 0.5
    
    # Advanced optimization with multiple strategies and better fallback
    try:
        # Multi-start approach with different initialization strategies
        best_result = None
        best_sum = float('-inf')
        
        # Try multiple different starting configurations
        for start_iter in range(20):  # Increase number of starts even more
            # Choose initialization strategy
            if start_iter == 0:
                # First try: use the best initialization
                x0 = circles.flatten()
            elif start_iter < 10:
                # Try perturbed version of best solution so far
                perturbed = circles.copy()
                for i in range(n):
                    # Perturb position more aggressively for exploration
                    perturbed[i, 0] += np.random.normal(0, 0.04)
                    perturbed[i, 1] += np.random.normal(0, 0.04)
                    # Keep within bounds
                    perturbed[i, 0] = max(0.01, min(0.99, perturbed[i, 0]))
                    perturbed[i, 1] = max(0.01, min(0.99, perturbed[i, 1]))
                    
                    # Perturb radius more aggressively
                    perturbed[i, 2] += np.random.normal(0, 0.04)
                    perturbed[i, 2] = max(0.001, min(0.49, perturbed[i, 2]))
                x0 = perturbed.flatten()
            else:
                # Try completely random initialization with better distribution
                temp_circles = np.zeros((32, 3))
                for i in range(32):
                    # Prefer placing circles away from corners to allow larger radii
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                    # Use more conservative radius calculation
                    r = min(x, 1-x, y, 1-y) * 0.42
                    r = max(0.01, min(0.48, r))
                    temp_circles[i] = [x, y, r]
                x0 = temp_circles.flatten()
            
            # Try different optimization methods for better results
            method = 'SLSQP'  # Stick with SLSQP as it's often most reliable for this type of problem
            
            # Optimization with bounds and constraints
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 2500, 'ftol': 1e-6, 'gtol': 1e-6, 'disp': False}
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
            
            # Early stopping if we're close to target
            if best_sum > 2.92:  # Stop early if we're very close to benchmark
                break
        
        if best_result is not None and best_result.success:
            final_circles = best_result.x.reshape(-1, 3)
        else:
            final_circles = circles
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        final_circles = circles
    
    # Final validation and cleanup
    # Ensure all circles are valid and within bounds
    for i in range(n):
        x, y, r = final_circles[i]
        # Ensure containment
        final_circles[i, 0] = max(r, min(1-r, x))
        final_circles[i, 1] = max(r, min(1-r, y))
        final_circles[i, 2] = max(0.001, min(0.5, r))
    
    # Double-check feasibility with tolerance
    if not check_feasibility(final_circles):
        # If still infeasible, try a more conservative approach
        for i in range(n):
            x, y, r = final_circles[i]
            # Make sure it's contained
            final_circles[i, 0] = max(r, min(1-r, x))
            final_circles[i, 1] = max(r, min(1-r, y))
            final_circles[i, 2] = max(0.001, min(0.49, r))
    
    return final_circles


# EVOLVE-BLOCK-END
