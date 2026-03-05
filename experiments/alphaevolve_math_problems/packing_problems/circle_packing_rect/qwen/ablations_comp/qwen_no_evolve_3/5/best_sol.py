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
from deap import base, creator, tools, algorithms
import multiprocessing as mp
import os
from sklearn.cluster import KMeans
import cvxpy as cp
from numba import jit

warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_distances_numba(positions, radii):
    """Efficiently compute distances between circles using numba"""
    n = len(positions)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses an improved hybrid optimization approach with better initialization and constraint handling.

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
    
    # Constraints for non-overlapping
    def constraint_distance(params):
        circles = params.reshape(-1, 3)
        # Use more efficient distance calculation
        distances = cdist(circles[:, :2], circles[:, :2])
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
    
    # Improved initialization approach using hexagonal packing
    def generate_initial_solution():
        # Try several aspect ratios based on literature and testing
        aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]
        
        best_config = None
        best_sum = 0
        
        for ratio in aspect_ratios:
            width = 2.0 / (1.0 + ratio)
            height = width * ratio
            
            # Create initial configuration using hexagonal packing approach
            circles = np.zeros((n, 3))
            
            # For 21 circles, try to use a 5x5 grid with some adjustments
            rows = 5
            cols = 5
            
            # Adjust grid to fit exactly 21 circles
            actual_cols = min(cols, n)
            actual_rows = min(rows, n)
            if actual_cols * actual_rows < n:
                actual_rows = int(np.ceil(n / actual_cols))
            
            # Grid spacing
            cell_width = width / actual_cols
            cell_height = height / actual_rows
            
            # Maximum possible radius based on grid spacing
            max_radius = min(cell_width, cell_height) / 2.0
            
            # Place circles in a staggered hexagonal pattern
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
                    
                    # Use a more conservative radius assignment to start with
                    # This gives room for optimization later
                    r = max_radius * random.uniform(0.7, 0.9)
                    
                    circles[idx] = [x, y, r]
                    idx += 1
                    
            if idx < n:
                # Fill remaining circles with better placement strategy
                for i in range(idx, n):
                    # Try to place in regions that are likely to have space
                    # Use a more intelligent placement: place near edges with proper spacing
                    attempt = 0
                    placed = False
                    while not placed and attempt < 50:
                        # Place near edge or in corners for better utilization
                        if random.random() < 0.4:
                            # Place near edge
                            side = random.choice(['top', 'bottom', 'left', 'right'])
                            if side == 'top' or side == 'bottom':
                                x = random.uniform(max_radius, width - max_radius)
                                y = max_radius if side == 'bottom' else height - max_radius
                            else:
                                x = max_radius if side == 'left' else width - max_radius
                                y = random.uniform(max_radius, height - max_radius)
                        else:
                            # Place randomly in valid region
                            x = random.uniform(max_radius, width - max_radius)
                            y = random.uniform(max_radius, height - max_radius)
                        
                        # Check if position is valid with existing circles
                        valid_position = True
                        for j in range(i):
                            dx = circles[j, 0] - x
                            dy = circles[j, 1] - y
                            distance_squared = dx*dx + dy*dy
                            min_distance = circles[j, 2] + max_radius * 0.7
                            if distance_squared < min_distance * min_distance:
                                valid_position = False
                                break
                        
                        if valid_position:
                            # Use a more appropriate radius
                            r = random.uniform(max_radius * 0.6, max_radius * 0.9)
                            circles[i] = [x, y, r]
                            placed = True
                        attempt += 1
                    
                    if not placed:
                        # Fallback to simple random placement
                        x = random.uniform(max_radius, width - max_radius)
                        y = random.uniform(max_radius, height - max_radius)
                        r = random.uniform(max_radius * 0.5, max_radius * 0.8)
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
    
    # Enhanced optimization using simpler but more effective approach
    def enhanced_optimization(initial_circles, width, height):
        # Simplified optimization approach focusing on effectiveness over complexity
        bounds = []
        for i in range(n):
            bounds.extend([(0, width), (0, height), (1e-6, width/2)])
        
        # Create constraints dictionaries
        def distance_constraint(params):
            circles = params.reshape(-1, 3)
            distances = cdist(circles[:, :2], circles[:, :2])
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    radii_sum = circles[i, 2] + circles[j, 2]
                    constraints.append(dist - radii_sum)
            return np.array(constraints)
        
        def bound_constraint(params):
            circles = params.reshape(-1, 3)
            constraints = []
            for i in range(n):
                constraints.append(circles[i, 0] - circles[i, 2])
                constraints.append(width - circles[i, 0] - circles[i, 2])
                constraints.append(circles[i, 1] - circles[i, 2])
                constraints.append(height - circles[i, 1] - circles[i, 2])
            return np.array(constraints)
        
        distance_cons = {
            'type': 'ineq',
            'fun': lambda x: distance_constraint(x)
        }
        
        bound_cons = {
            'type': 'ineq', 
            'fun': lambda x: bound_constraint(x)
        }
        
        # Use only SLSQP for faster convergence with fewer iterations
        try:
            result = minimize(
                objective,
                initial_circles.flatten(),
                method='SLSQP',
                constraints=[distance_cons, bound_cons],
                options={'maxiter': 200, 'ftol': 1e-10, 'eps': 1e-10},
                bounds=bounds
            )
            
            if result.success:
                return result.x.reshape(-1, 3)
        except:
            pass
        
        # If optimization fails, return initial solution
        return initial_circles
    
    # Try multiple rectangle dimensions for better results
    best_circles = None
    best_sum = 0
    best_width = 1.0
    best_height = 1.0
    
    # Focus on aspect ratios that tend to work well for circle packing
    # Based on literature and previous experiments, ratios around 1.0-2.0 work well
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5]
    
    # Try a few key aspect ratios first for quick wins
    for ratio in aspect_ratios[:8]:
        width = 2.0 / (1.0 + ratio)
        height = width * ratio
        
        # Generate initial solution for this ratio
        circles, _, _ = generate_initial_solution()
        
        # Optimize for this configuration
        optimized_circles = enhanced_optimization(circles, width, height)
        
        # Check if this is better
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
            best_width = width
            best_height = height
    
    # If we don't have a good solution yet, try more aggressive optimization
    if best_circles is None:
        # Try a more aggressive optimization approach
        circles, width, height = generate_initial_solution()
        # Try with a more focused optimization approach
        bounds = []
        for i in range(n):
            bounds.extend([(0, width), (0, height), (1e-6, width/2)])
        
        def distance_constraint(params):
            circles = params.reshape(-1, 3)
            distances = cdist(circles[:, :2], circles[:, :2])
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    radii_sum = circles[i, 2] + circles[j, 2]
                    constraints.append(dist - radii_sum)
            return np.array(constraints)
        
        def bound_constraint(params):
            circles = params.reshape(-1, 3)
            constraints = []
            for i in range(n):
                constraints.append(circles[i, 0] - circles[i, 2])
                constraints.append(width - circles[i, 0] - circles[i, 2])
                constraints.append(circles[i, 1] - circles[i, 2])
                constraints.append(height - circles[i, 1] - circles[i, 2])
            return np.array(constraints)
        
        distance_cons = {
            'type': 'ineq',
            'fun': lambda x: distance_constraint(x)
        }
        
        bound_cons = {
            'type': 'ineq', 
            'fun': lambda x: bound_constraint(x)
        }
        
        # Try optimization with a different solver
        try:
            result = minimize(
                objective,
                circles.flatten(),
                method='L-BFGS-B',
                constraints=[distance_cons, bound_cons],
                options={'maxiter': 100, 'ftol': 1e-12},
                bounds=bounds
            )
            
            if result.success:
                best_circles = result.x.reshape(-1, 3)
                best_sum = np.sum(best_circles[:, 2])
        except:
            pass
    
    # Final validation and return
    if best_circles is not None:
        # Validate constraints and clean up
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
