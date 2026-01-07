# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
from typing import Tuple
from deap import base, creator, tools, algorithms
import time
from numba import jit
import warnings
from sklearn.cluster import KMeans
from itertools import combinations
import heapq

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@jit(nopython=True)
def fast_distance_squared(x1, y1, x2, y2):
    """Fast computation of squared distance"""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy

@jit(nopython=True)
def fast_validate_circles_fast(circles, n):
    """Fast validation of circle constraints using numba"""
    # Check containment
    for i in range(n):
        x, y, r = circles[i, 0], circles[i, 1], circles[i, 2]
        if not (r <= x <= 1-r and r <= y <= 1-r):
            return False
    
    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i, 0], circles[i, 1], circles[i, 2]
            x2, y2, r2 = circles[j, 0], circles[j, 1], circles[j, 2]
            dist_sq = fast_distance_squared(x1, y1, x2, y2)
            if dist_sq < (r1 + r2) * (r1 + r2):
                return False
    return True

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithms and local optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Better initialization using a proven hexagonal packing approach
    def initialize_hexagonal():
        circles = []
        
        # Use a more systematic approach inspired by hexagonal packing
        # Start with a central region and expand outward
        center_positions = [(0.5, 0.5)]
        corner_positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        
        # Place corner circles with larger radii
        for i, (x, y) in enumerate(corner_positions):
            if len(circles) < n:
                circles.append([x, y, 0.12])
        
        # Place center circle
        if len(circles) < n:
            circles.append([0.5, 0.5, 0.15])
        
        # Create hexagonal grid around the center
        # Hexagonal packing density is about 0.9069
        grid_spacing = 0.25
        base_radius = 0.07
        
        # Generate hexagonal pattern
        rows = 4
        cols = 4
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Hexagonal offset pattern
                x = 0.2 + j * grid_spacing
                y = 0.2 + i * grid_spacing * 0.866  # sqrt(3)/2 ≈ 0.866
                if i % 2 == 1:
                    x += grid_spacing * 0.5
                # Ensure within bounds
                x = max(base_radius, min(1-base_radius, x))
                y = max(base_radius, min(1-base_radius, y))
                circles.append([x, y, base_radius])
        
        # Fill remaining positions with carefully chosen random placements
        while len(circles) < n:
            # Bias towards center but allow edge placements
            x = 0.1 + np.random.random() * 0.8
            y = 0.1 + np.random.random() * 0.8
            # More realistic radii based on position
            center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            r = 0.05 + 0.1 * (1 - center_dist / 0.707)
            r = max(0.03, min(0.15, r))
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Even better initialization using simulated annealing-inspired approach
    def initialize_simulated_annealing():
        # Start with a basic configuration
        circles = []
        
        # Place circles in a strategic pattern
        # Use a 5x5 grid pattern with some randomness
        for i in range(5):
            for j in range(5):
                if len(circles) >= n:
                    break
                x = 0.1 + j * 0.22
                y = 0.1 + i * 0.22
                if i % 2 == 1:
                    x += 0.11
                circles.append([x, y, 0.08])
        
        # Adjust radii based on proximity to center
        for i in range(len(circles)):
            x, y, r = circles[i]
            center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            # Make radii larger near center
            new_r = 0.08 + 0.07 * (1 - center_dist / 0.707)
            new_r = max(0.03, min(0.15, new_r))
            circles[i][2] = new_r
            
        # Fill any remaining slots
        while len(circles) < n:
            x = 0.1 + np.random.random() * 0.8
            y = 0.1 + np.random.random() * 0.8
            center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            r = 0.05 + 0.1 * (1 - center_dist / 0.707)
            r = max(0.03, min(0.15, r))
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Optimized constraint validation using spatial indexing for better performance
    def validate_circles_optimized(circles):
        """More efficient constraint validation using spatial indexing"""
        # Check containment first
        for i in range(len(circles)):
            x, y, r = circles[i]
            if not (r <= x <= 1-r and r <= y <= 1-r):
                return False
        
        # Use spatial indexing for overlap checking - only check nearby circles
        # Create a simple spatial hash for performance
        grid_size = 10
        grid = {}
        
        # Place circles into grid cells
        for i in range(len(circles)):
            x, y, r = circles[i]
            cell_x = int(x * grid_size)
            cell_y = int(y * grid_size)
            cell_key = (cell_x, cell_y)
            if cell_key not in grid:
                grid[cell_key] = []
            grid[cell_key].append(i)
        
        # Check overlaps
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            # Check nearby cells
            cell_x = int(x1 * grid_size)
            cell_y = int(y1 * grid_size)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_cell = (cell_x + dx, cell_y + dy)
                    if neighbor_cell in grid:
                        for j in grid[neighbor_cell]:
                            if i != j:
                                x2, y2, r2 = circles[j]
                                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                                if dist_sq < (r1 + r2)**2:
                                    return False
        return True
    
    # Improved objective function with better handling
    def objective(params):
        # params contains [x1,y1,r1,x2,y2,r2,...,x26,y26,r26]
        circles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            circles.append([x, y, r])
        
        # Return negative sum of radii (since we want to maximize)
        return -sum(circle[2] for circle in circles)
    
    # Better constraint functions using vectorized operations
    def constraint_containment(params):
        # Ensure all circles are within the unit square
        cons = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            # r <= x <= 1-r and r <= y <= 1-r
            cons.append(x - r)      # x - r >= 0
            cons.append(1 - r - x)  # 1 - r - x >= 0
            cons.append(y - r)      # y - r >= 0
            cons.append(1 - r - y)  # 1 - r - y >= 0
        return np.array(cons)
    
    def constraint_nonoverlap(params):
        # Vectorized non-overlap constraints
        cons = []
        # Only compute necessary pairs
        for i in range(n):
            for j in range(i+1, n):
                x1 = params[3*i]
                y1 = params[3*i+1]
                r1 = params[3*i+2]
                x2 = params[3*j]
                y2 = params[3*j+1]
                r2 = params[3*j+2]
                # (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                # We want dist_sq - (r1+r2)^2 >= 0
                cons.append(dist_sq - (r1+r2)**2)
        return np.array(cons)
    
    # Enhanced evolutionary approach with better parameters and selection
    def evolutionary_approach_enhanced():
        # Define the fitness function
        def eval_fitness(individual):
            # Convert individual to circles array
            circles = []
            for i in range(n):
                x = individual[3*i]
                y = individual[3*i+1]
                r = individual[3*i+2]
                circles.append([x, y, r])
            
            # Validate constraints
            if not validate_circles_optimized(np.array(circles)):
                return (-1000000,)  # Invalid solution
            
            # Return negative sum of radii (minimize negative = maximize sum)
            return (-sum(circle[2] for circle in circles),)
        
        # Create DEAP types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        # Use tighter bounds and better initialization
        toolbox.register("attr_float", np.random.uniform, 0.01, 0.99)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 3*n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.02, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=5)
        
        # Create initial population with better diversity
        pop = toolbox.population(n=150)
        
        # Run evolution with more generations
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.4, 
                                             ngen=250, stats=stats, halloffame=hof, verbose=False)
            return hof[0]
        except Exception:
            return None
    
    # Advanced local optimization with better convergence criteria
    def advanced_local_optimization():
        best_sum = 0
        best_result = None
        
        # Try multiple initialization strategies
        init_strategies = [
            initialize_hexagonal,
            initialize_simulated_annealing,
            lambda: KMeans(n_clusters=n, random_state=42).fit(np.random.random((500, 2))).cluster_centers_
        ]
        
        for init_strategy in init_strategies:
            try:
                # Try with multiple restarts
                for restart in range(3):
                    try:
                        if callable(init_strategy):
                            initial_circles = init_strategy()
                        else:
                            initial_circles = init_strategy()
                        
                        # Flatten for optimization
                        initial_params = initial_circles.flatten()
                        
                        # Set up bounds for optimization
                        bounds = []
                        for i in range(n):
                            # x bounds
                            bounds.append((0.001, 0.999))   # x coordinate
                            # y bounds  
                            bounds.append((0.001, 0.999))   # y coordinate
                            # r bounds
                            bounds.append((0.001, 0.499))   # radius
                        
                        # Define constraints
                        cons = [
                            {'type': 'ineq', 'fun': constraint_containment},
                            {'type': 'ineq', 'fun': constraint_nonoverlap}
                        ]
                        
                        # Try multiple methods for robustness
                        methods = ['trust-constr', 'SLSQP']
                        for method in methods:
                            try:
                                result = minimize(
                                    objective,
                                    initial_params,
                                    method=method,
                                    bounds=bounds,
                                    constraints=cons,
                                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
                                )
                                
                                if result.success:
                                    current_sum = -result.fun
                                    if current_sum > best_sum:
                                        best_sum = current_sum
                                        best_result = result.x
                                        break  # Found a good solution
                            except Exception:
                                continue
                                
                    except Exception:
                        continue  # Continue with other restarts if this fails
                        
            except Exception:
                continue  # Continue with other strategies if this fails
        
        return best_result, best_sum
    
    # Custom local search refinement approach
    def local_search_refinement(initial_params):
        """Apply a custom local search refinement"""
        # Convert to circles
        circles = []
        for i in range(n):
            x = initial_params[3*i]
            y = initial_params[3*i+1]
            r = initial_params[3*i+2]
            circles.append([x, y, r])
        
        # Simple gradient-based improvement
        # This is a simplified version that tries small adjustments
        best_params = initial_params.copy()
        best_sum = -objective(initial_params)
        
        # Try small perturbations to improve solution
        for _ in range(1000):
            # Randomly select a circle to perturb
            idx = np.random.randint(0, n)
            params = best_params.copy()
            
            # Slightly adjust one parameter
            param_type = np.random.randint(0, 3)  # 0=x, 1=y, 2=r
            if param_type == 0:  # x
                params[3*idx] = max(0.001, min(0.999, params[3*idx] + np.random.normal(0, 0.005)))
            elif param_type == 1:  # y
                params[3*idx+1] = max(0.001, min(0.999, params[3*idx+1] + np.random.normal(0, 0.005)))
            else:  # r
                params[3*idx+2] = max(0.001, min(0.499, params[3*idx+2] + np.random.normal(0, 0.003)))
            
            # Check if new solution is valid and better
            try:
                circles_new = []
                for i in range(n):
                    x = params[3*i]
                    y = params[3*i+1]
                    r = params[3*i+2]
                    circles_new.append([x, y, r])
                
                if validate_circles_optimized(np.array(circles_new)):
                    new_sum = -objective(params)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_params = params.copy()
            except:
                continue
        
        return best_params
    
    # Main optimization loop
    try:
        best_result = None
        best_sum = 0
        
        # Strategy 1: Enhanced evolutionary approach
        print("Starting enhanced evolutionary approach...")
        evol_result = evolutionary_approach_enhanced()
        if evol_result is not None:
            # Convert back to circles
            circles = []
            for i in range(n):
                x = evol_result[3*i]
                y = evol_result[3*i+1]
                r = evol_result[3*i+2]
                circles.append([x, y, r])
            
            if validate_circles_optimized(np.array(circles)):
                current_sum = sum(circle[2] for circle in circles)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = evol_result
        
        # Strategy 2: Advanced local optimization
        print("Starting advanced local optimization...")
        local_result, local_sum = advanced_local_optimization()
        if local_result is not None and local_sum > best_sum:
            best_sum = local_sum
            best_result = local_result
        
        # Strategy 3: Local search refinement
        if best_result is not None:
            print("Applying local search refinement...")
            refined_result = local_search_refinement(best_result)
            # Validate and compare
            try:
                circles = []
                for i in range(n):
                    x = refined_result[3*i]
                    y = refined_result[3*i+1]
                    r = refined_result[3*i+2]
                    circles.append([x, y, r])
                
                if validate_circles_optimized(np.array(circles)):
                    refined_sum = sum(circle[2] for circle in circles)
                    if refined_sum > best_sum:
                        best_sum = refined_sum
                        best_result = refined_result
            except Exception:
                pass
        
        # Strategy 4: Final aggressive optimization
        if best_result is None:
            print("Trying aggressive optimization with improved starting points...")
            # Use better initialization
            initial_circles = initialize_hexagonal()
            initial_params = initial_circles.flatten()
            
            bounds = []
            for i in range(n):
                bounds.append((0.001, 0.999))   # x coordinate
                bounds.append((0.001, 0.999))   # y coordinate
                bounds.append((0.001, 0.499))   # radius
            
            cons = [
                {'type': 'ineq', 'fun': constraint_containment},
                {'type': 'ineq', 'fun': constraint_nonoverlap}
            ]
            
            # Try with trust-constr which often works better for this problem
            try:
                result = minimize(
                    objective,
                    initial_params,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-13}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result.x
            except Exception:
                pass
        
        # If we found a good result, return it; otherwise return the best initialization
        if best_result is not None:
            optimized_circles = []
            for i in range(n):
                x = best_result[3*i]
                y = best_result[3*i+1]
                r = best_result[3*i+2]
                optimized_circles.append([x, y, r])
            return np.array(optimized_circles)
        else:
            # Fallback to the hexagonal initialization
            return initialize_hexagonal()
            
    except Exception as e:
        # Fallback to hexagonal initialization if anything goes wrong
        print(f"Exception occurred: {e}")
        return initialize_hexagonal()


# EVOLVE-BLOCK-END
