# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from deap import base, creator, tools, algorithms
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithm with geometric initialization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Better geometric initialization using a more systematic approach
    # Try to create a good starting configuration
    circles = np.zeros((n, 3))
    
    # Use a more sophisticated packing approach - try to fill space efficiently
    # Start with a grid-based approach but optimize spacing
    rows = 6
    cols = 6
    if rows * cols < n:
        rows = 5
        cols = 7
    if rows * cols < n:
        rows = 4
        cols = 8
    
    # Calculate spacing for better distribution
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Place circles in a grid pattern with slight randomization
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Add some randomness to avoid perfect grid
            x = (j + 0.5 + np.random.normal(0, 0.1)) * spacing_x
            y = (i + 0.5 + np.random.normal(0, 0.1)) * spacing_y
            
            # Keep within bounds
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            # Initial radius - based on available space
            max_r = min(x, 1-x, y, 1-y)
            r = max_r * 0.4  # Start with smaller radius to allow for optimization
            
            if 0 <= x <= 1 and 0 <= y <= 1:
                circles[idx] = [x, y, r]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining slots with random placements if needed
    for i in range(idx, n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        # Radius should be reasonable for placement
        r = np.random.uniform(0.01, 0.1)
        circles[i] = [x, y, r]
    
    # Phase 2: Evolutionary algorithm approach for better optimization
    # Define fitness function
    def evaluate_individual(individual):
        # individual is [x1,y1,r1,x2,y2,r2,...]
        circles_array = np.array(individual).reshape(-1, 3)
        
        # Check constraints
        total_radius = np.sum(circles_array[:, 2])
        
        # Penalty for constraint violations
        penalty = 0
        
        # Containment penalties
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 1000  # Large penalty
        
        # Overlap penalties
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist = r1 + r2
                if dist_sq < min_dist**2:
                    # Penalty based on how much they overlap
                    overlap = min_dist - np.sqrt(dist_sq)
                    penalty += 10000 * overlap
        
        # Return negative since we want to maximize (minimize negative)
        return -(total_radius - penalty),
    
    # Create DEAP structures
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n*3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.02, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolution
    population = toolbox.population(n=50)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolve for a few generations
    for gen in range(20):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        population[:] = offspring
    
    # Get best individual
    best_individual = tools.selBest(population, 1)[0]
    best_circles = np.array(best_individual).reshape(-1, 3)
    
    # Phase 3: Refinement with local optimization
    # Use scipy optimization for final refinement
    def objective(params):
        # params contains [x1,y1,r1,x2,y2,r2,...]
        circles_array = params.reshape(-1, 3)
        total_radius = np.sum(circles_array[:, 2])
        return -total_radius  # Negative because we minimize
    
    def constraint_containment(i):
        def func(params):
            circles_array = params.reshape(-1, 3)
            x, y, r = circles_array[i]
            # Circle must be within square with radius r
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        return func
    
    def constraint_overlap(i, j):
        def func(params):
            circles_array = params.reshape(-1, 3)
            x1, y1, r1 = circles_array[i]
            x2, y2, r2 = circles_array[j]
            # Distance between centers minus sum of radii (should be >= 0 for non-overlapping)
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            return math.sqrt(dist_sq) - (r1 + r2)
        return func
    
    # Set up constraints for scipy
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': constraint_containment(i)})
    
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': constraint_overlap(i, j)})
    
    # Bounds for variables: x,y in [0,1], r in [0,0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Initial parameters from evolution
    initial_params = best_circles.flatten()
    
    try:
        result = minimize(objective, initial_params, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 500, 'ftol': 1e-6})
        
        if result.success:
            optimized_params = result.x
            best_circles = optimized_params.reshape(-1, 3)
    except Exception:
        # If optimization fails, use the evolutionary result
        pass
    
    # Final local improvement
    best_sum = np.sum(best_circles[:, 2])
    
    # More aggressive local search with better neighborhood exploration
    for _ in range(1000):  # More iterations
        test_circles = best_circles.copy()
        
        # Try to improve one circle at a time
        idx = np.random.randint(0, n)
        
        # Try to increase radius first (if possible)
        old_radius = test_circles[idx, 2]
        test_circles[idx, 2] = min(old_radius + 0.01, 0.5)  # Increase radius slightly
        
        # Try to adjust position
        test_circles[idx, 0] += np.random.normal(0, 0.005)
        test_circles[idx, 1] += np.random.normal(0, 0.005)
        
        # Enforce bounds
        test_circles[idx, 0] = np.clip(test_circles[idx, 0], 0, 1)
        test_circles[idx, 1] = np.clip(test_circles[idx, 1], 0, 1)
        test_circles[idx, 2] = np.clip(test_circles[idx, 2], 0, 0.5)
        
        # Check validity
        valid = True
        for i in range(n):
            if i != idx:
                dx = test_circles[i, 0] - test_circles[idx, 0]
                dy = test_circles[i, 1] - test_circles[idx, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = test_circles[i, 2] + test_circles[idx, 2]
                if dist_sq < min_dist*min_dist:
                    valid = False
                    break
        
        # Check containment
        if valid and test_circles[idx, 0] - test_circles[idx, 2] >= 0 and \
           test_circles[idx, 0] + test_circles[idx, 2] <= 1 and \
           test_circles[idx, 1] - test_circles[idx, 2] >= 0 and \
           test_circles[idx, 1] + test_circles[idx, 2] <= 1:
            
            # Accept if better
            test_sum = np.sum(test_circles[:, 2])
            if test_sum > best_sum:
                best_circles = test_circles
                best_sum = test_sum
    
    return best_circles


# EVOLVE-BLOCK-END
