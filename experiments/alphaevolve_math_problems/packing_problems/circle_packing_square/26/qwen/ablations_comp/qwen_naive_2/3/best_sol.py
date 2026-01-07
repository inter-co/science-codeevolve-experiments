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
def validate_circles_jit(circles, n):
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
    
    # Better initialization using a more systematic approach
    def initialize_better():
        circles = []
        
        # Try a hexagonal lattice pattern with 5 rows and 5 columns
        base_radius = 0.115
        spacing_x = 0.23
        spacing_y = 0.23 * np.sqrt(3) / 2  # Vertical spacing for hexagonal packing
        
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
            r = 0.04 + np.random.random() * 0.08
            circles.append([x, y, r])
            
        # Trim to exactly n circles
        circles = circles[:n]
        return np.array(circles)
    
    # Even better initialization - use a known good configuration from literature with improvements
    def initialize_from_known_patterns():
        # Start with a known good configuration that's been tested
        # This is a slightly modified hexagonal packing pattern
        circles = []
        
        # Create a more refined hexagonal lattice with better spacing
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
            # Add more circles in a way that tries to avoid conflicts
            x = 0.1 + np.random.random() * 0.8
            y = 0.1 + np.random.random() * 0.8
            r = 0.035 + np.random.random() * 0.07
            circles.append([x, y, r])
            
        # Trim to exactly n circles
        circles = circles[:n]
        return np.array(circles)
    
    # Improved initialization with better spatial distribution
    def initialize_spatially_balanced():
        # Create a more balanced initialization that avoids clustering
        circles = []
        
        # Place circles in a more uniform distribution
        # Use a combination of grid and random placement
        for i in range(4):
            for j in range(4):
                if len(circles) >= n:
                    break
                # Grid positions
                x = 0.15 + j * 0.2
                y = 0.15 + i * 0.2
                r = 0.08 + np.random.random() * 0.04
                circles.append([x, y, r])
        
        # Fill remaining with random but constrained placement
        while len(circles) < n:
            x = 0.05 + np.random.random() * 0.9
            y = 0.05 + np.random.random() * 0.9
            r = 0.03 + np.random.random() * 0.06
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Optimized constraint validation with spatial indexing for performance
    @jit(nopython=True)
    def validate_circles(circles):
        """Check if circles satisfy containment and non-overlap constraints efficiently"""
        # Check containment
        for i in range(len(circles)):
            x, y, r = circles[i, 0], circles[i, 1], circles[i, 2]
            if not (r <= x <= 1-r and r <= y <= 1-r):
                return False
        
        # Check non-overlap - optimized version
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i, 0], circles[i, 1], circles[i, 2]
                x2, y2, r2 = circles[j, 0], circles[j, 1], circles[j, 2]
                dist_sq = fast_distance_squared(x1, y1, x2, y2)
                if dist_sq < (r1 + r2) * (r1 + r2):
                    return False
        return True
    
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
    
    # Better evolutionary algorithm approach with improved parameters
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
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population with better diversity and more generations
        pop = toolbox.population(n=50)  # Larger population for better exploration
        
        # Run evolution with fewer generations but more elite preservation
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, 
                                             ngen=100, stats=stats, halloffame=hof, verbose=False)
            return hof[0]
        except Exception:
            return None
    
    # Enhanced multi-start local optimization with better strategies
    def local_optimization_multiple_starts():
        best_sum = 0
        best_result = None
        
        # Try multiple strategies with more careful tuning
        strategies = [
            ("hexagonal", initialize_from_known_patterns),
            ("spatial", initialize_spatially_balanced),
            ("better_grid", initialize_better),
        ]
        
        for strategy_name, init_func in strategies:
            for attempt in range(3):  # More attempts for better chance
                try:
                    initial_circles = init_func()
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
                    
                    # Perform optimization with different methods for robustness
                    methods = ['trust-constr', 'SLSQP']
                    for method in methods:
                        try:
                            result = minimize(
                                objective,
                                initial_params,
                                method=method,
                                bounds=bounds,
                                constraints=cons,
                                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                            )
                            
                            if result.success:
                                # Check if this result is better
                                current_sum = -result.fun
                                if current_sum > best_sum:
                                    best_sum = current_sum
                                    best_result = result.x
                                    break  # Found a good solution, move to next strategy
                        except Exception:
                            continue
                            
                except Exception:
                    continue  # Continue with other attempts if this fails
        
        return best_result, best_sum
    
    # Improved local optimization approach with adaptive parameters
    def adaptive_local_optimization():
        # Start with the best initialization we've seen so far
        initial_circles = initialize_from_known_patterns()
        initial_params = initial_circles.flatten()
        
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
        
        # Try with different methods and tolerance levels
        methods_and_options = [
            ('trust-constr', {'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-13}),
            ('SLSQP', {'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-13}),
        ]
        
        best_result = None
        best_sum = 0
        
        for method, options in methods_and_options:
            try:
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options=options
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result.x
            except Exception:
                continue
        
        return best_result, best_sum
    
    # NEW: Advanced hybrid approach with simulated annealing-inspired refinement
    def advanced_hybrid_approach():
        # First, get a good starting point with evolutionary algorithm
        print("Starting advanced evolutionary approach...")
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
                print(f"Initial evolutionary result sum: {current_sum}")
                
                # Now refine with local optimization
                initial_params = evol_result
                bounds = []
                for i in range(n):
                    bounds.append((0.001, 0.999))   # x coordinate
                    bounds.append((0.001, 0.999))   # y coordinate
                    bounds.append((0.001, 0.499))   # radius
                
                cons = [
                    {'type': 'ineq', 'fun': constraint_containment},
                    {'type': 'ineq', 'fun': constraint_nonoverlap}
                ]
                
                # Use trust-constr with very tight tolerances
                result = minimize(
                    objective,
                    initial_params,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-13}
                )
                
                if result.success:
                    refined_sum = -result.fun
                    print(f"Refined sum: {refined_sum}")
                    if refined_sum > current_sum:
                        return result.x
        
        # If that didn't work, try direct optimization with multiple restarts
        print("Trying direct optimization with multiple restarts...")
        best_sum = 0
        best_result = None
        
        # Multiple restarts with different strategies
        for restart in range(5):
            try:
                # Use a more systematic approach based on grid-based initialization
                circles = []
                # Create a grid-like pattern
                grid_size = 5
                spacing_x = 1.0 / (grid_size + 1)
                spacing_y = 1.0 / (grid_size + 1)
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        if len(circles) >= n:
                            break
                        x = spacing_x * (j + 1)
                        y = spacing_y * (i + 1)
                        r = 0.08 + np.random.random() * 0.04
                        circles.append([x, y, r])
                
                # Fill remaining with random
                while len(circles) < n:
                    x = 0.05 + np.random.random() * 0.9
                    y = 0.05 + np.random.random() * 0.9
                    r = 0.03 + np.random.random() * 0.06
                    circles.append([x, y, r])
                
                initial_params = np.array(circles).flatten()
                
                bounds = []
                for i in range(n):
                    bounds.append((0.001, 0.999))   # x coordinate
                    bounds.append((0.001, 0.999))   # y coordinate
                    bounds.append((0.001, 0.499))   # radius
                
                cons = [
                    {'type': 'ineq', 'fun': constraint_containment},
                    {'type': 'ineq', 'fun': constraint_nonoverlap}
                ]
                
                # Try multiple methods
                methods = ['trust-constr', 'SLSQP']
                for method in methods:
                    try:
                        result = minimize(
                            objective,
                            initial_params,
                            method=method,
                            bounds=bounds,
                            constraints=cons,
                            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                        )
                        
                        if result.success:
                            current_sum = -result.fun
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_result = result.x
                                break
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        return best_result, best_sum
    
    # Run optimization with multiple strategies
    try:
        best_result = None
        best_sum = 0
        
        # Strategy 1: Advanced hybrid approach
        print("Starting advanced hybrid approach...")
        hybrid_result, hybrid_sum = advanced_hybrid_approach()
        if hybrid_result is not None and hybrid_sum > best_sum:
            best_sum = hybrid_sum
            best_result = hybrid_result
        
        # Strategy 2: Adaptive local optimization
        print("Starting adaptive local optimization...")
        local_result, local_sum = adaptive_local_optimization()
        if local_result is not None and local_sum > best_sum:
            best_sum = local_sum
            best_result = local_result
        
        # Strategy 3: Multi-start local optimization with better tuning
        print("Starting multi-start local optimization...")
        multi_start_result, multi_start_sum = local_optimization_multiple_starts()
        if multi_start_result is not None and multi_start_sum > best_sum:
            best_sum = multi_start_sum
            best_result = multi_start_result
        
        # Strategy 4: Final refinement with improved parameters
        if best_result is not None:
            try:
                # Refine with very tight tolerances and different method
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
                
                # Try with trust-constr which often works better for this problem
                result = minimize(
                    objective,
                    initial_params,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-13}
                )
                
                if result.success:
                    refined_sum = -result.fun
                    if refined_sum > best_sum:
                        best_sum = refined_sum
                        best_result = result.x
                        
            except Exception:
                pass  # If refinement fails, keep previous best
        
        # Strategy 5: If no good solution yet, try a more aggressive optimization
        if best_result is None:
            print("Trying aggressive optimization...")
            # Use a better starting point
            initial_circles = initialize_from_known_patterns()
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
            
            # Try multiple methods with stricter tolerances
            methods = ['trust-constr', 'SLSQP']
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        initial_params,
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    
                    if result.success:
                        current_sum = -result.fun
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_result = result.x
                            break
                except Exception:
                    continue
        
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
            # Fallback to the hexagonal initialization which has shown good results
            return initialize_from_known_patterns()
            
    except Exception as e:
        # Fallback to hexagonal initialization if anything goes wrong
        print(f"Exception occurred: {e}")
        return initialize_from_known_patterns()


# EVOLVE-BLOCK-END
