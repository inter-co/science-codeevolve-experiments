# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
from itertools import combinations
import time
from scipy.optimize import differential_evolution
import warnings
from scipy.spatial import distance
import copy
from scipy.optimize import dual_annealing
import math
from scipy.optimize import shgo
import heapq
from scipy.optimize import Bounds
import numba
from numba import jit
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_distances_numba(circles):
    """Fast distance computation for circle packing"""
    n = len(circles)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric initialization and global optimization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    n = 21
    
    # Objective function to maximize sum of radii
    def objective(params):
        # Reshape parameters: [x1,y1,r1,x2,y2,r2,...,x21,y21,r21]
        circles = params.reshape(-1, 3)
        # Minimize negative sum of radii (maximize sum)
        return -np.sum(circles[:, 2])
    
    # More efficient constraint checking
    def constraint_distance(params):
        circles = params.reshape(-1, 3)
        # Use fast Numba-based distance computation
        distances = compute_distances_numba(circles)
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                radii_sum = circles[i, 2] + circles[j, 2]
                # Constraint: distance >= radii_sum (for non-overlap)
                constraints.append(dist - radii_sum)
        return np.array(constraints)
    
    # Constraints for boundary conditions
    def constraint_bounds(params, width, height):
        circles = params.reshape(-1, 3)
        constraints = []
        for i in range(n):
            # x - r >= 0 (left boundary)
            constraints.append(circles[i, 0] - circles[i, 2])
            # width - x - r >= 0 (right boundary)  
            constraints.append(width - circles[i, 0] - circles[i, 2])
            # y - r >= 0 (bottom boundary)
            constraints.append(circles[i, 1] - circles[i, 2])
            # height - y - r >= 0 (top boundary)
            constraints.append(height - circles[i, 1] - circles[i, 2])
        return np.array(constraints)
    
    # Better initialization approach inspired by known optimal patterns
    def generate_initial_solution():
        # Try a more systematic approach with focus on good aspect ratios
        # Based on research, aspect ratios around 1.2-1.5 tend to work well for circle packing
        
        aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2]
        
        best_config = None
        best_sum = 0
        
        for ratio in aspect_ratios:
            width = 2.0 / (1.0 + ratio)
            height = width * ratio
            
            # Create a more sophisticated initial arrangement using hexagonal packing
            circles = np.zeros((n, 3))
            
            # Use a more structured approach for better initial packing
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            
            # Adjust for exact count
            actual_cols = min(cols, n)
            actual_rows = min(rows, n)
            if actual_cols * actual_rows < n:
                actual_rows = int(np.ceil(n / actual_cols))
            
            # Grid spacing
            cell_width = width / actual_cols
            cell_height = height / actual_rows
            
            # Maximum possible radius based on grid spacing
            max_radius = min(cell_width, cell_height) / 2.0
            
            # Place circles in a staggered pattern for better packing
            idx = 0
            for row in range(actual_rows):
                for col in range(actual_cols):
                    if idx >= n:
                        break
                        
                    # Staggered pattern to improve packing
                    x_offset = 0.5 * (row % 2)  # Offset every other row
                    x = (col + 0.5 + x_offset) * cell_width
                    y = (row + 0.5) * cell_height
                    
                    # Ensure within bounds
                    x = max(max_radius, min(width - max_radius, x))
                    y = max(max_radius, min(height - max_radius, y))
                    
                    # Use more conservative initial radius for better optimization
                    r = max_radius * random.uniform(0.6, 0.85)
                    
                    circles[idx] = [x, y, r]
                    idx += 1
                    
            if idx < n:
                # Fill remaining circles with some randomness
                for i in range(idx, n):
                    # Random placement within bounds
                    x = random.uniform(max_radius, width - max_radius)
                    y = random.uniform(max_radius, height - max_radius)
                    r = random.uniform(max_radius * 0.4, max_radius * 0.7)
                    circles[i] = [x, y, r]
            
            # Evaluate this configuration
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_config = (circles.copy(), width, height)
        
        if best_config is not None:
            return best_config[0], best_config[1], best_config[2]
        else:
            # Fallback to default
            width = 1.0
            height = 1.0
            circles = np.zeros((n, 3))
            max_radius = 0.2
            for i in range(n):
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                r = random.uniform(max_radius * 0.5, max_radius * 0.9)
                circles[i] = [x, y, r]
            return circles, width, height
    
    # Use a more aggressive optimization strategy
    def optimize_with_improved_strategy(initial_circles, width, height):
        # Use a combination of different optimization approaches
        bounds = []
        for i in range(n):
            # x bounds
            bounds.extend([(0, width), (0, height), (1e-6, width/2)])
        
        # Create constraints dictionary
        def combined_constraint(params):
            # Combine both constraint types
            dist_constraints = constraint_distance(params)
            bound_constraints = constraint_bounds(params, width, height)
            return np.concatenate([dist_constraints, bound_constraints])
        
        # Use a more robust optimization approach
        try:
            # First try L-BFGS-B which is usually faster for smooth problems
            result = minimize(
                objective,
                initial_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                refined_circles = result.x.reshape(-1, 3)
                # Verify constraints are satisfied
                if np.all(constraint_distance(result.x) >= -1e-8) and \
                   np.all(constraint_bounds(result.x, width, height) >= -1e-8):
                    return refined_circles, True
        except Exception as e:
            pass
        
        # If that fails, use SLSQP with better settings
        try:
            result = minimize(
                objective,
                initial_circles.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'eps': 1e-12},
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: constraint_distance(x)},
                    {'type': 'ineq', 'fun': lambda x: constraint_bounds(x, width, height)}
                ]
            )
            
            if result.success:
                refined_circles = result.x.reshape(-1, 3)
                return refined_circles, True
        except Exception as e:
            pass
        
        return initial_circles, False
    
    # Try multiple rectangle dimensions for better results
    best_circles = None
    best_sum = 0
    best_width = 1.0
    best_height = 1.0
    
    # Focus on the most promising aspect ratios for circle packing
    # These are based on known good solutions for similar problems
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5]
    
    # Try fewer aspect ratios to save time but focus on high-quality ones
    for ratio in aspect_ratios[:8]:  # Reduced number for speed
        width = 2.0 / (1.0 + ratio)
        height = width * ratio
        
        # Generate initial solution for this ratio
        circles, _, _ = generate_initial_solution()
        
        # Optimize for this configuration
        optimized_circles, success = optimize_with_improved_strategy(circles, width, height)
        
        # Check if this is better
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
            best_width = width
            best_height = height
    
    # Try a few more aspect ratios with different strategies
    for ratio in aspect_ratios[8:]:
        width = 2.0 / (1.0 + ratio)
        height = width * ratio
        
        # Generate initial solution for this ratio
        circles, _, _ = generate_initial_solution()
        
        # Optimize for this configuration
        optimized_circles, success = optimize_with_improved_strategy(circles, width, height)
        
        # Check if this is better
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
            best_width = width
            best_height = height
    
    # Final optimization with the best configuration using multiple passes
    if best_circles is not None:
        # Create final bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0, best_width), (0, best_height), (1e-6, best_width/2)])
        
        # More aggressive final optimization
        try:
            # Try multiple optimization methods
            methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
            
            for method in methods_to_try:
                try:
                    result = minimize(
                        objective,
                        best_circles.flatten(),
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
                    )
                    
                    if result.success:
                        final_circles = result.x.reshape(-1, 3)
                        # Verify final constraints
                        if (np.all(constraint_distance(result.x) >= -1e-10) and 
                            np.all(constraint_bounds(result.x, best_width, best_height) >= -1e-10)):
                            best_circles = final_circles
                            break
                except Exception as e:
                    continue
        except Exception as e:
            pass
    
    # Ensure all circles are valid
    if best_circles is not None:
        # Validate constraints
        circles = best_circles.copy()
        for i in range(len(circles)):
            # Ensure radii are positive
            circles[i, 2] = max(1e-6, circles[i, 2])
            # Ensure positions are valid
            circles[i, 0] = max(circles[i, 2], min(best_width - circles[i, 2], circles[i, 0]))
            circles[i, 1] = max(circles[i, 2], min(best_height - circles[i, 2], circles[i, 1]))
        
        return circles
    
    # Fallback to initial solution
    circles, _, _ = generate_initial_solution()
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
