# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from deap import base, creator, tools, algorithms
import random
import math
from typing import Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithm with local optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    N_CIRCLES = 26
    POPULATION_SIZE = 200
    GENERATIONS = 150
    MUTATION_RATE = 0.12
    CROSSOVER_RATE = 0.8
    TOURNAMENT_SIZE = 7
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    def create_individual():
        """Create a random valid individual (26 circles with x,y,r)"""
        individual = []
        for _ in range(N_CIRCLES):
            # Better initialization: use a more structured approach
            # Place circles in a grid-like pattern initially, then randomize
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Start with larger radii for better initial coverage
            r = np.random.uniform(0.02, 0.20)
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    def evaluate(individual):
        """Evaluate fitness of an individual (sum of radii) with penalty for violations"""
        circles = np.array(individual).reshape(-1, 3)
        
        # Calculate sum of radii
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for containment violations - more aggressive penalties
        penalty = 0
        for i in range(N_CIRCLES):
            x, y, r = circles[i]
            # Hard penalty for boundary violations
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 100000 * (abs(x - r) + abs(x + r - 1) + abs(y - r) + abs(y + r - 1))
        
        # Penalty for overlap violations - more sophisticated penalty calculation
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < r1 + r2:
                    # Penalty based on how much they overlap, scaled by radii
                    overlap = (r1 + r2 - distance)
                    # Use quadratic penalty for better convergence
                    penalty += 10000 * overlap * overlap
        
        # Return fitness (total radius minus penalty)
        return (total_radius - penalty,)
    
    def mutate(individual):
        """Mutate an individual with adaptive mutation"""
        for i in range(len(individual)):
            if random.random() < MUTATION_RATE:
                if i % 3 == 0:  # x coordinate
                    individual[i] = max(0.01, min(0.99, individual[i] + np.random.normal(0, 0.03)))
                elif i % 3 == 1:  # y coordinate
                    individual[i] = max(0.01, min(0.99, individual[i] + np.random.normal(0, 0.03)))
                else:  # radius
                    individual[i] = max(0.005, min(0.49, individual[i] + np.random.normal(0, 0.03)))
        return individual,
    
    def crossover(ind1, ind2):
        """Crossover two individuals with uniform crossover"""
        if random.random() < CROSSOVER_RATE:
            size = len(ind1)
            # Uniform crossover with some bias towards preserving good features
            for i in range(0, size, 3):
                if random.random() < 0.6:  # Bias towards keeping better values
                    ind1[i:i+3], ind2[i:i+3] = ind2[i:i+3], ind1[i:i+3]
        return ind1, ind2
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    
    # Create initial population
    population = toolbox.population(n=POPULATION_SIZE)
    
    # Evaluate initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Evolution loop with adaptive parameters
    best_fitness_history = []
    for generation in range(GENERATIONS):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population with offspring
        population[:] = offspring
        
        # Track best fitness
        current_best = max([ind.fitness.values[0] for ind in population])
        best_fitness_history.append(current_best)
        
        # Print progress every 20 generations
        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {current_best:.4f}")
    
    # Find the best individual
    best_individual = tools.selBest(population, 1)[0]
    circles = np.array(best_individual).reshape(-1, 3)
    
    # Apply advanced local optimization
    try:
        refined_circles = advanced_local_optimization(circles)
        return refined_circles
    except Exception as e:
        print(f"Advanced local optimization failed: {e}")
        return circles

def advanced_local_optimization(circles: np.ndarray) -> np.ndarray:
    """Apply advanced local optimization using sequential quadratic programming"""
    from scipy.optimize import minimize
    import scipy.optimize as opt
    
    n = len(circles)
    
    # Create constraint functions for both containment and overlap
    def containment_constraints(params):
        circles_flat = params.reshape(-1, 3)
        violations = []
        
        # Containment constraints (more precise)
        for i in range(n):
            x, y, r = circles_flat[i]
            violations.extend([
                x - r,           # x - r >= 0
                1 - x - r,       # 1 - x - r >= 0
                y - r,           # y - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        return np.array(violations)
    
    def overlap_constraints(params):
        circles_flat = params.reshape(-1, 3)
        violations = []
        
        # Overlap constraints  
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_flat[i]
                x2, y2, r2 = circles_flat[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                violations.append(distance - r1 - r2)  # distance - r1 - r2 >= 0
        return np.array(violations)
    
    # Combined constraint function
    def combined_constraint(params):
        return np.concatenate([containment_constraints(params), overlap_constraints(params)])
    
    # Set bounds (tighter bounds)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Initial parameters
    initial_params = circles.flatten()
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        circles_flat = params.reshape(-1, 3)
        return -np.sum(circles_flat[:, 2])  # Negative because we maximize
    
    # Create constraints
    cons = [{'type': 'ineq', 'fun': combined_constraint}]
    
    # Try multiple optimization approaches
    best_result = None
    best_value = float('-inf')
    
    # Method 1: SLSQP with multiple starting points
    for attempt in range(5):
        # Perturb initial solution slightly
        perturbed_params = initial_params + np.random.normal(0, 0.01, len(initial_params)) * (0.5 + attempt * 0.1)
        perturbed_params = np.clip(perturbed_params, 0.001, 0.999)
        
        try:
            result = minimize(
                objective,
                perturbed_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                current_value = -result.fun  # Convert back to maximization value
                if current_value > best_value:
                    best_value = current_value
                    best_result = result
        except:
            continue
    
    # Method 2: Trust-constr with better convergence
    if best_result is None or best_value < 2.5:
        try:
            result = minimize(
                objective,
                initial_params,
                method='trust-constr',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'gtol': 1e-6, 'xtol': 1e-6}
            )
            
            if result.success:
                current_value = -result.fun
                if current_value > best_value:
                    best_value = current_value
                    best_result = result
        except:
            pass
    
    if best_result is not None and best_result.success:
        return best_result.x.reshape(-1, 3)
    else:
        # If optimization fails, return the original solution
        return circles


# EVOLVE-BLOCK-END
