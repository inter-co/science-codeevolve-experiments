# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from scipy.spatial import cKDTree
from itertools import combinations
from deap import base, creator, tools, algorithms
import random
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithm with geometric initialization and 
    advanced optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    random.seed(42)
    np.random.seed(42)
    
    # Improved initialization using a more effective approach based on known good packings
    circles = np.zeros((n, 3))
    
    # Better initialization: start with a hexagonal-like pattern
    # This creates a more uniform distribution that allows for better optimization
    rows = 6
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    # Create a hexagonal pattern with offset rows
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for better packing
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + 1 + offset) * spacing_x
            y = (i + 1) * spacing_y
            # Initial radius - start with a reasonable value based on spacing
            r = min(spacing_x, spacing_y) * 0.3
            circles[idx] = [x, y, r]
            idx += 1
    
    # Fill remaining positions with more strategic placement
    if idx < n:
        # Place remaining circles in a way that tries to avoid crowding
        for i in range(idx, n):
            # Try to place in areas that might benefit from additional circles
            attempts = 0
            while attempts < 50:
                # Try placing near center with smaller radius first
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                r = np.random.uniform(0.01, 0.05)
                
                # Check if this placement would be valid
                valid = True
                # Check containment
                if x - r < 0 or y - r < 0 or x + r > 1 or y + r > 1:
                    valid = False
                
                if valid:
                    # Check overlap with existing circles
                    for k in range(i):
                        cx, cy, cr = circles[k]
                        dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                        if dist < (r + cr):
                            valid = False
                            break
                
                if valid:
                    circles[i] = [x, y, r]
                    break
                attempts += 1
    
    # Define the objective function and constraints for evolutionary algorithm
    def evaluate_individual(individual):
        """Evaluate fitness of an individual (negative sum of radii for minimization)"""
        circles_array = np.array(individual).reshape(-1, 3)
        
        # Check containment
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            if x - r < 0 or y - r < 0 or x + r > 1 or y + r > 1:
                return (float('inf'),)  # Invalid solution
        
        # Check overlaps
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < (r1 + r2):
                    return (float('inf'),)  # Invalid solution
        
        # Return negative sum of radii (since we want to maximize)
        total_radius = np.sum(circles_array[:, 2])
        return (-total_radius,)
    
    # Evolutionary algorithm approach for better global search
    def evolutionary_approach():
        # Create individual representation: [x1, y1, r1, x2, y2, r2, ...]
        IND_SIZE = n * 3
        
        # Create DEAP classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.001, 0.999)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        pop = toolbox.population(n=50)
        
        # Evaluate initial population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Begin evolution
        for gen in range(100):  # Limit generations to stay within time limits
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:
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
            
            # Replace the old population with the new generation
            pop[:] = offspring
            
            # Print progress
            if gen % 20 == 0:
                fits = [ind.fitness.values[0] for ind in pop]
                length = len(pop)
                mean = sum(fits) / length
                print(f"Generation {gen}: Best = {-mean}")
        
        # Return best individual
        best_ind = tools.selBest(pop, 1)[0]
        return np.array(best_ind).reshape(-1, 3)
    
    # Try evolutionary approach first for global optimization
    try:
        evol_start = time.time()
        evolved_solution = evolutionary_approach()
        evol_time = time.time() - evol_start
        
        # Refine with local optimization if we have time
        if evol_time < 45:  # Leave some time for final refinement
            # Use the evolved solution as starting point for local optimization
            initial_solution = evolved_solution
        else:
            initial_solution = circles
    except Exception as e:
        # Fallback to initial solution if evolutionary fails
        initial_solution = circles
    
    # Final local optimization with scipy
    def compute_total_radius(circles_array):
        """Compute sum of all radii"""
        return np.sum(circles_array[:, 2])
    
    def check_containment(circles_array):
        """Check if all circles are contained within unit square"""
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            if x - r < 0 or y - r < 0 or x + r > 1 or y + r > 1:
                return False
        return True
    
    def check_overlaps(circles_array):
        """Check if any circles overlap"""
        for i, j in combinations(range(len(circles_array)), 2):
            x1, y1, r1 = circles_array[i]
            x2, y2, r2 = circles_array[j]
            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < (r1 + r2):
                return False
        return True
    
    # Objective function to maximize (negative because minimize)
    def objective(circles_flat):
        total_radius = np.sum(circles_flat[2::3])  # Sum of all radii
        return -total_radius
    
    # Constraint functions
    def containment_constraints(circles_flat):
        """Return positive values when all containment constraints are satisfied"""
        circles_array = circles_flat.reshape(-1, 3)
        result = []
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # x - r >= 0, y - r >= 0, 1-x-r >= 0, 1-y-r >= 0
            result.extend([x - r, y - r, 1 - x - r, 1 - y - r])
        return np.array(result)
    
    def overlap_constraints(circles_flat):
        """Return positive values when no overlaps exist"""
        circles_array = circles_flat.reshape(-1, 3)
        result = []
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                # Distance should be >= sum of radii (positive when satisfied)
                result.append(dist - (r1 + r2))
        return np.array(result)
    
    # Bounds for optimization
    bounds = []
    for i in range(n):
        # x, y, r bounds (tighter bounds for better convergence)
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Use the best solution found so far as starting point
    best_circles = initial_solution.copy()
    best_sum = compute_total_radius(best_circles)
    
    # More aggressive local search
    try:
        # Multi-stage local search with more aggressive improvements
        for stage in range(4):
            # Stage 1: Radius improvements
            if stage == 0:
                iterations = 500
                step_size = 0.008
            elif stage == 1:
                iterations = 300
                step_size = 0.005
            elif stage == 2:
                iterations = 200
                step_size = 0.003
            else:  # stage == 3
                iterations = 100
                step_size = 0.001
            
            for iteration in range(iterations):
                improved = False
                
                # Try to improve each circle's radius
                for i in range(n):
                    old_circle = best_circles[i].copy()
                    
                    # Try increasing radius
                    test_r = old_circle[2] + step_size
                    if test_r <= 0.499:
                        # Check if this adjustment maintains constraints
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = test_r
                        
                        # Check containment
                        x, y, r = temp_circles[i]
                        if (r <= x and r <= y and r <= (1-x) and r <= (1-y)):
                            # Check overlap with others using more efficient approach
                            valid = True
                            
                            # Only check nearby circles for efficiency
                            tree = cKDTree(temp_circles[:, :2])
                            # Find neighbors within a reasonable distance
                            neighbors = tree.query_ball_point([x, y], 2*(r + 0.01))
                            
                            for j in neighbors:
                                if i != j:
                                    x1, y1, r1 = temp_circles[i]
                                    x2, y2, r2 = temp_circles[j]
                                    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                                    if dist < (r1 + r2):
                                        valid = False
                                        break
                            
                            if valid:
                                best_circles[i, 2] = test_r
                                improved = True
                                best_sum = compute_total_radius(best_circles)
                
                # Occasionally try small position adjustments too
                if iteration % 10 == 0 and not improved:
                    for i in range(n):
                        old_x, old_y, old_r = best_circles[i]
                        # Small random adjustments
                        dx = np.random.uniform(-step_size*3, step_size*3)
                        dy = np.random.uniform(-step_size*3, step_size*3)
                        test_x = old_x + dx
                        test_y = old_y + dy
                        
                        # Keep within bounds
                        test_x = max(old_r, min(1-old_r, test_x))
                        test_y = max(old_r, min(1-old_r, test_y))
                        
                        # Check if this adjustment improves the configuration
                        temp_circles = best_circles.copy()
                        temp_circles[i, 0] = test_x
                        temp_circles[i, 1] = test_y
                        
                        # Check overlap with others
                        valid = True
                        tree = cKDTree(temp_circles[:, :2])
                        neighbors = tree.query_ball_point([test_x, test_y], 2*(old_r + 0.01))
                        
                        for j in neighbors:
                            if i != j:
                                x1, y1, r1 = temp_circles[i]
                                x2, y2, r2 = temp_circles[j]
                                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                                if dist < (r1 + r2):
                                    valid = False
                                    break
                        
                        if valid:
                            best_circles[i, 0] = test_x
                            best_circles[i, 1] = test_y
                            improved = True
                            best_sum = compute_total_radius(best_circles)
                
                if not improved and stage > 0:
                    break
        
        # Final scipy optimization with better settings
        final_circles = best_circles.copy()
        x0 = final_circles.flatten()
        
        # Define constraints for scipy optimization
        cons = []
        
        # Containment constraints
        cons.append({
            'type': 'ineq', 
            'fun': lambda x: containment_constraints(x)
        })
        
        # Overlap constraints
        cons.append({
            'type': 'ineq', 
            'fun': lambda x: overlap_constraints(x)
        })
        
        # Run optimization with better parameters
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-7, 'eps': 1e-7},
            callback=lambda x: None  # No callback needed
        )
        
        if res.success:
            optimized_circles = res.x.reshape(-1, 3)
            # Validate and clean up the result
            final_result = optimized_circles.copy()
        else:
            # If optimization failed, return our best local search result
            final_result = best_circles
            
    except Exception as e:
        # Fallback to local search result if optimization fails
        final_result = best_circles
    
    # Final validation and cleanup
    for i in range(n):
        # Ensure radius is reasonable
        final_result[i, 2] = max(0.001, min(0.499, final_result[i, 2]))
        
        # Ensure containment
        x, y, r = final_result[i]
        final_result[i] = [
            max(r, min(1-r, x)),
            max(r, min(1-r, y)),
            r
        ]
    
    return final_result


# EVOLVE-BLOCK-END
