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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from numba import jit
import math
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from functools import partial
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True)
def compute_distances_numba(circles):
    """Fast distance computation using numba"""
    n = len(circles)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid evolutionary approach combining genetic algorithms with local optimization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    n = 21
    max_width = 2.0
    max_height = 2.0
    
    # Objective function to maximize sum of radii
    def objective(params):
        # Reshape parameters: [x1,y1,r1,x2,y2,r2,...,x21,y21,r21]
        circles = params.reshape(-1, 3)
        # Minimize negative sum of radii (maximize sum)
        return -np.sum(circles[:, 2])
    
    # Constraint checking function
    def check_constraints(params, width, height):
        circles = params.reshape(-1, 3)
        
        # Check boundary constraints
        for i in range(n):
            if (circles[i, 0] - circles[i, 2] < 0 or 
                circles[i, 0] + circles[i, 2] > width or
                circles[i, 1] - circles[i, 2] < 0 or
                circles[i, 1] + circles[i, 2] > height):
                return False
        
        # Check overlap constraints
        distances = cdist(circles[:, :2], circles[:, :2])
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                radii_sum = circles[i, 2] + circles[j, 2]
                if dist < radii_sum:
                    return False
        return True
    
    # Fitness function for evolutionary algorithm
    def fitness_function(params):
        circles = params.reshape(-1, 3)
        sum_radii = np.sum(circles[:, 2])
        
        # Penalize invalid solutions heavily
        if not check_constraints(params, 1.0, 1.0):  # Default rectangle
            return -1e10,  # Very bad fitness
            
        return sum_radii,
    
    # Better initialization using hexagonal packing with adaptive sizing
    def generate_hexagonal_initialization(width=1.0, height=1.0):
        # Create hexagonal arrangement
        circles = np.zeros((n, 3))
        
        # Calculate grid size for hexagonal packing
        rows = int(np.ceil(np.sqrt(n * 1.2)))
        cols = int(np.ceil(n / rows))
        
        if cols * rows < n:
            rows = int(np.ceil(n / cols))
            
        # Grid spacing
        cell_width = width / cols
        cell_height = height / rows
        
        # Maximum possible radius based on grid spacing
        max_radius = min(cell_width, cell_height) / 2.0
        
        # Place circles in a staggered pattern for better packing
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                    
                # Staggered pattern to improve packing
                x_offset = 0.5 * (row % 2)  # Offset every other row
                x = (col + 0.5 + x_offset) * cell_width
                y = (row + 0.5) * cell_height
                
                # Ensure within bounds
                x = max(max_radius, min(width - max_radius, x))
                y = max(max_radius, min(height - max_radius, y))
                
                # Use a more aggressive radius assignment to start with
                r = max_radius * random.uniform(0.7, 0.95)
                
                circles[idx] = [x, y, r]
                idx += 1
                
        # Fill remaining circles with strategic placement
        if idx < n:
            for i in range(idx, n):
                # Place in a more scattered pattern with better distribution
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                r = random.uniform(max_radius * 0.6, max_radius * 0.9)
                circles[i] = [x, y, r]
        
        return circles
    
    # Advanced initialization using spiral pattern
    def generate_spiral_initialization(width=1.0, height=1.0):
        circles = np.zeros((n, 3))
        center_x, center_y = width/2, height/2
        max_radius = min(width, height) / 4
        
        # Place circles in spiral pattern
        for i in range(n):
            # Spiral layout
            angle = i * 0.8
            radius = (i + 1) * max_radius / n * 0.8
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Ensure within bounds
            x = max(max_radius, min(width - max_radius, x))
            y = max(max_radius, min(height - max_radius, y))
            
            # Radius decreases with distance from center
            r = max_radius * 0.7 * (1 - i / n) * random.uniform(0.8, 1.0)
            r = max(0.01, min(max_radius * 0.5, r))
            
            circles[i] = [x, y, r]
        
        return circles
    
    # Multi-objective optimization approach with better constraint handling
    def multi_objective_optimization():
        # Define the problem as a multi-objective optimization
        # We'll use a weighted approach to balance objectives
        
        # Initialize population
        def create_individual():
            # Random rectangle aspect ratio
            ratio = random.uniform(0.5, 2.0)
            width = 2.0 / (1.0 + ratio)
            height = width * ratio
            
            # Generate initial circles
            circles = generate_hexagonal_initialization(width, height)
            
            # Flatten for GA representation
            individual = circles.flatten()
            return individual
        
        # Create toolbox for DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            # Convert back to circles array
            circles = np.array(individual).reshape(-1, 3)
            
            # Extract width and height from first few parameters (approximate)
            # This is a simplified approach - in practice we'd need to track rectangle size separately
            width = 1.0
            height = 1.0
            
            # Check constraints
            if not check_constraints(individual, width, height):
                return -1e10,  # Invalid solution penalty
            
            # Return sum of radii
            return np.sum(circles[:, 2]),
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution
        pop = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                             ngen=30, stats=stats, halloffame=hof, verbose=False)
            return hof[0]
        except Exception:
            # Fallback to local optimization
            return create_individual()
    
    # Improved optimization with better constraint handling
    def optimize_with_improved_strategies(initial_circles, width, height):
        # Strategy: Use a two-phase approach
        # Phase 1: Global optimization with bounds
        bounds = []
        for i in range(n):
            # x bounds
            bounds.extend([(0, width), (0, height), (1e-6, width/2)])
        
        # Create constraint function that returns positive values when satisfied
        def constraint_distance(params):
            circles = params.reshape(-1, 3)
            distances = cdist(circles[:, :2], circles[:, :2])
            constraints = []
            # Only compute upper triangle to avoid duplicates
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    radii_sum = circles[i, 2] + circles[j, 2]
                    # Positive value means constraint satisfied (distance >= radii_sum)
                    constraints.append(dist - radii_sum)
            return np.array(constraints)
        
        def constraint_bounds(params):
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
        
        # Optimization with better parameter settings
        try:
            # Try different optimization approaches
            result1 = minimize(
                objective,
                initial_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result1.success:
                # Try SLSQP with constraints
                distance_cons = {
                    'type': 'ineq',
                    'fun': lambda x: constraint_distance(x)
                }
                
                bound_cons = {
                    'type': 'ineq', 
                    'fun': lambda x: constraint_bounds(x)
                }
                
                result2 = minimize(
                    objective,
                    result1.x,
                    method='SLSQP',
                    constraints=[distance_cons, bound_cons],
                    options={'maxiter': 200, 'ftol': 1e-8, 'eps': 1e-8},
                    bounds=bounds
                )
                
                if result2.success:
                    refined_circles = result2.x.reshape(-1, 3)
                    return refined_circles, True
        except Exception as e:
            pass
        
        return initial_circles, False
    
    # Try different strategies and aspect ratios
    best_circles = None
    best_sum = 0
    best_width = 1.0
    best_height = 1.0
    
    # Test different aspect ratios systematically
    aspect_ratios = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
    
    # Try multiple initialization methods
    init_methods = [generate_hexagonal_initialization, generate_spiral_initialization]
    
    for ratio in aspect_ratios:
        width = 2.0 / (1.0 + ratio)
        height = width * ratio
        
        # Try different initialization methods
        for init_method in init_methods:
            try:
                # Generate initial solution
                circles = init_method(width, height)
                
                # Optimize this configuration
                optimized_circles, success = optimize_with_improved_strategies(circles, width, height)
                
                # Check if this is better
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized_circles.copy()
                    best_width = width
                    best_height = height
            except Exception as e:
                continue
    
    # Try evolutionary approach as fallback
    if best_sum < 2.0:  # If not good enough, try evolutionary approach
        try:
            # Run evolutionary optimization
            evolved_individual = multi_objective_optimization()
            evolved_circles = np.array(evolved_individual).reshape(-1, 3)
            
            # Evaluate evolved solution
            current_sum = np.sum(evolved_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = evolved_circles.copy()
        except Exception as e:
            pass
    
    # Final refinement with improved algorithm
    if best_circles is not None:
        # Create final bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0, best_width), (0, best_height), (1e-6, best_width/2)])
        
        # Recreate constraints with more efficient implementation
        def final_constraint_distance(params):
            circles = params.reshape(-1, 3)
            distances = cdist(circles[:, :2], circles[:, :2])
            constraints = []
            # Only compute upper triangle to avoid duplicates
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    radii_sum = circles[i, 2] + circles[j, 2]
                    constraints.append(dist - radii_sum)
            return np.array(constraints)
        
        def final_constraint_bounds(params):
            circles = params.reshape(-1, 3)
            constraints = []
            for i in range(n):
                constraints.append(circles[i, 0] - circles[i, 2])
                constraints.append(best_width - circles[i, 0] - circles[i, 2])
                constraints.append(circles[i, 1] - circles[i, 2])
                constraints.append(best_height - circles[i, 1] - circles[i, 2])
            return np.array(constraints)
        
        # Try with multiple optimization approaches
        try:
            # Method 1: L-BFGS-B first for global search
            result1 = minimize(
                objective,
                best_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result1.success:
                # Then SLSQP with constraints
                distance_cons = {
                    'type': 'ineq',
                    'fun': lambda x: final_constraint_distance(x)
                }
                
                bound_cons = {
                    'type': 'ineq', 
                    'fun': lambda x: final_constraint_bounds(x)
                }
                
                result2 = minimize(
                    objective,
                    result1.x,
                    method='SLSQP',
                    constraints=[distance_cons, bound_cons],
                    options={'maxiter': 200, 'ftol': 1e-8, 'eps': 1e-8},
                    bounds=bounds
                )
                
                if result2.success:
                    best_circles = result2.x.reshape(-1, 3)
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
    circles = generate_hexagonal_initialization(1.0, 1.0)
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
