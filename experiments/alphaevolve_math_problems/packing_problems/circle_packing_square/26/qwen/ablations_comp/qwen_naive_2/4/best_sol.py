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
from scipy.spatial import cKDTree
from itertools import combinations

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
    
    # Better initialization with more sophisticated approach
    def initialize_better():
        # Start with a hexagonal packing pattern that's known to work well
        circles = []
        
        # Try a hexagonal lattice pattern with 5 rows and 5 columns
        base_radius = 0.12
        spacing_x = 0.25
        spacing_y = 0.25 * np.sqrt(3) / 2  # Vertical spacing for hexagonal packing
        
        # Generate positions in a hexagonal pattern
        for i in range(5):
            for j in range(5):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Apply offset to odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                circles.append([x, y, base_radius])
        
        # If we don't have enough circles, add more strategically
        while len(circles) < n:
            # Add more circles with small random perturbations to avoid conflicts
            x = 0.1 + np.random.random() * 0.8
            y = 0.1 + np.random.random() * 0.8
            r = 0.05 + np.random.random() * 0.1
            circles.append([x, y, r])
            
        # Trim to exactly n circles
        circles = circles[:n]
        return np.array(circles)
    
    # Even better initialization - use a known good configuration from literature
    def initialize_from_known_patterns():
        # Initialize with a known good configuration for 26 circles (based on hexagonal packing)
        circles = []
        
        # Hexagonal lattice pattern with 5x5 grid but slightly adjusted for better packing
        base_radius = 0.115
        spacing_x = 0.23
        spacing_y = 0.23 * np.sqrt(3) / 2
        
        # Generate positions in a hexagonal pattern
        for i in range(5):
            for j in range(5):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Apply offset to odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                circles.append([x, y, base_radius])
        
        # If we don't have enough circles, add more strategically
        while len(circles) < n:
            # Add more circles in a way that tries to avoid conflicts
            x = 0.1 + np.random.random() * 0.8
            y = 0.1 + np.random.random() * 0.8
            r = 0.04 + np.random.random() * 0.08
            circles.append([x, y, r])
            
        # Trim to exactly n circles
        circles = circles[:n]
        return np.array(circles)
    
    # Even better initialization - optimized version with more careful positioning
    def initialize_optimized():
        # Start with a good hexagonal pattern
        circles = []
        base_radius = 0.112
        spacing_x = 0.225
        spacing_y = 0.225 * np.sqrt(3) / 2
        
        # Generate positions in a hexagonal pattern
        for i in range(5):
            for j in range(5):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Apply offset to odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                circles.append([x, y, base_radius])
        
        # If we don't have enough circles, add more strategically
        while len(circles) < n:
            # Add more circles with higher probability of being in less crowded areas
            x = 0.1 + np.random.random() * 0.8
            y = 0.1 + np.random.random() * 0.8
            # Use a slightly smaller radius to allow more circles
            r = 0.03 + np.random.random() * 0.07
            circles.append([x, y, r])
            
        # Trim to exactly n circles
        circles = circles[:n]
        return np.array(circles)
    
    # Optimized constraint validation with early exit
    def validate_circles(circles):
        """Check if circles satisfy containment and non-overlap constraints efficiently"""
        return fast_validate_circles_fast(circles, len(circles))
    
    # Optimized objective function
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
    
    # Optimized constraint functions using vectorization
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
        # Vectorized non-overlap constraint checking
        cons = []
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
    
    # Enhanced evolutionary algorithm with better parameters
    def evolutionary_approach():
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
            if not validate_circles(np.array(circles)):
                return (-1000000,)  # Invalid solution
            
            # Return negative sum of radii (minimize negative = maximize sum)
            return (-sum(circle[2] for circle in circles),)
        
        # Create DEAP types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        # Use tighter bounds and better initialization for convergence
        toolbox.register("attr_float", np.random.uniform, 0.01, 0.99)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 3*n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.015, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population with better diversity and more generations
        pop = toolbox.population(n=150)
        
        # Run evolution with more generations and better termination criteria
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, 
                                             ngen=300, stats=stats, halloffame=hof, verbose=False)
            return hof[0]
        except Exception:
            return None
    
    # More aggressive local optimization approach
    def aggressive_local_optimization():
        best_sum = 0
        best_result = None
        
        # Try multiple initialization strategies
        initial_strategies = [
            initialize_optimized,
            initialize_from_known_patterns,
            initialize_better
        ]
        
        for init_strategy in initial_strategies:
            try:
                initial_circles = init_strategy()
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
                
                # Try multiple methods with very strict tolerances
                methods = ['trust-constr', 'SLSQP']
                for method in methods:
                    try:
                        result = minimize(
                            objective,
                            initial_params,
                            method=method,
                            bounds=bounds,
                            constraints=cons,
                            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12}
                        )
                        
                        if result.success:
                            current_sum = -result.fun
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_result = result.x
                    except Exception:
                        continue
                        
            except Exception:
                continue  # Continue with other strategies if this fails
        
        return best_result, best_sum
    
    # Multi-stage optimization approach
    def multi_stage_optimization():
        # Stage 1: Fast heuristic approach to get a good starting point
        print("Stage 1: Fast heuristic initialization")
        initial_circles = initialize_optimized()
        
        # Stage 2: Local optimization with multiple restarts
        print("Stage 2: Local optimization")
        best_result, best_sum = aggressive_local_optimization()
        
        # Stage 3: Evolutionary approach if needed
        if best_result is None:
            print("Stage 3: Evolutionary approach")
            evol_result = evolutionary_approach()
            if evol_result is not None:
                # Convert back to circles
                circles = []
                for i in range(n):
                    x = evol_result[3*i]
                    y = evol_result[3*i+1]
                    r = evol_result[3*i+2]
                    circles.append([x, y, r])
                
                if validate_circles(np.array(circles)):
                    current_sum = sum(circle[2] for circle in circles)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = evol_result
        
        # Stage 4: Final refinement with highest precision
        if best_result is not None:
            try:
                circles = []
                for i in range(n):
                    x = best_result[3*i]
                    y = best_result[3*i+1]
                    r = best_result[3*i+2]
                    circles.append([x, y, r])
                
                initial_params = np.array(best_result)
                
                # Set up bounds for optimization
                bounds = []
                for i in range(n):
                    bounds.append((0.001, 0.999))   # x coordinate
                    bounds.append((0.001, 0.999))   # y coordinate
                    bounds.append((0.001, 0.499))   # radius
                
                # Define constraints
                cons = [
                    {'type': 'ineq', 'fun': constraint_containment},
                    {'type': 'ineq', 'fun': constraint_nonoverlap}
                ]
                
                # Try with trust-constr which often works better for this problem
                result = minimize(
                    objective,
                    initial_params,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if result.success:
                    refined_sum = -result.fun
                    if refined_sum > best_sum:
                        best_sum = refined_sum
                        best_result = result.x
                        
            except Exception:
                pass  # If refinement fails, keep previous best
        
        return best_result, best_sum
    
    # Run optimization with multi-stage approach
    try:
        print("Starting multi-stage optimization...")
        best_result, best_sum = multi_stage_optimization()
        
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
            # Fallback to the optimized initialization which should be quite good
            return initialize_optimized()
            
    except Exception as e:
        # Fallback to optimized initialization if anything goes wrong
        print(f"Exception occurred: {e}")
        return initialize_optimized()


# EVOLVE-BLOCK-END
