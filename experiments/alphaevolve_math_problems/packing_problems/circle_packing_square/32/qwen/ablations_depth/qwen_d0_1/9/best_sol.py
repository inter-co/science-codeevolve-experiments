# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from deap import base, creator, tools, algorithms
import random
import time
from scipy.spatial.distance import cdist

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: improved evolutionary algorithm + advanced local optimization refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    max_time = 60  # seconds
    start_time = time.time()
    
    # Use evolutionary algorithm for global search
    toolbox = base.Toolbox()
    
    # Create individual (32 circles = 96 parameters: x1,y1,r1,x2,y2,r2,...)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    def create_individual():
        # Generate better initial configuration using a more structured approach
        individual = []
        
        # Start with a grid-like pattern for better distribution
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        for i in range(n):
            # Grid positions with slight randomization
            row = i // grid_size
            col = i % grid_size
            x = spacing_x * (col + 1) + random.uniform(-0.01, 0.01)
            y = spacing_y * (row + 1) + random.uniform(-0.01, 0.01)
            
            # Ensure positions are within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Initial radius based on proximity to other circles
            r = min(0.05, 0.5 / np.sqrt(n))  # Scale with number of circles
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    def evaluate(individual):
        """Evaluate fitness as sum of radii, penalize violations with better penalty scheme"""
        circles = np.array(individual).reshape(-1, 3)
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for constraint violations - softer penalties with better scaling
        penalty = 0
        
        # Boundary penalties - soft penalty based on how much we're violating
        for i in range(n):
            x, y, r = circles[i]
            # Calculate how much we violate boundaries
            left_violation = max(0, r - x)
            right_violation = max(0, x + r - 1)
            bottom_violation = max(0, r - y)
            top_violation = max(0, y + r - 1)
            
            penalty += (left_violation + right_violation + bottom_violation + top_violation) * 100000
        
        # Overlap penalties - more sophisticated penalty calculation
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < r1 + r2:
                    # Soft penalty based on overlap amount
                    overlap = (r1 + r2) - dist
                    penalty += overlap * 1000000  # Much higher penalty for overlaps
                    
        # Additional penalty for very small radii (encourage larger circles)
        small_radius_penalty = 0
        for i in range(n):
            r = circles[i, 2]
            if r < 0.01:
                small_radius_penalty += (0.01 - r) * 10000
                
        penalty += small_radius_penalty
        
        return (total_radius - penalty,)
    
    def mutate(individual):
        # More sophisticated mutation with different strategies
        idx = random.randint(0, len(individual)-1)
        if idx % 3 == 0:  # x coordinate
            # Larger mutations for x
            individual[idx] = max(0.001, min(0.999, individual[idx] + random.gauss(0, 0.02)))
        elif idx % 3 == 1:  # y coordinate
            # Larger mutations for y
            individual[idx] = max(0.001, min(0.999, individual[idx] + random.gauss(0, 0.02)))
        else:  # radius
            # Smaller mutations for radius
            individual[idx] = max(0.001, min(0.499, individual[idx] + random.gauss(0, 0.01)))
        return individual,
    
    def crossover(ind1, ind2):
        # Uniform crossover with better mixing
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize population with better diversity
    population = toolbox.population(n=100)  # Increased population size
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolution parameters - tuned for better performance
    CXPB = 0.8   # Higher crossover probability
    MUTPB = 0.4  # Higher mutation probability
    NGEN = 100   # More generations
    
    # Main evolution loop with early stopping criteria
    best_fitness_history = []
    stagnation_count = 0
    max_stagnation = 20
    
    for gen in range(NGEN):
        if time.time() - start_time > max_time * 0.7:  # Leave more time for refinement
            break
            
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Replace the old population with the new one
        population[:] = offspring
        
        # Track best fitness for early stopping
        current_best = max([ind.fitness.values[0] for ind in population])
        best_fitness_history.append(current_best)
        
        # Check for stagnation
        if len(best_fitness_history) > 10:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-10]
            if recent_improvement < 1e-6:
                stagnation_count += 1
                if stagnation_count > max_stagnation:
                    break
            else:
                stagnation_count = 0
    
    # Get the best individual
    best_individual = tools.selBest(population, 1)[0]
    circles = np.array(best_individual).reshape(-1, 3)
    
    # Refine with enhanced local optimization using multiple approaches
    refined_circles = refine_with_enhanced_local_optimization(circles, max_time - (time.time() - start_time))
    
    return refined_circles

def refine_with_enhanced_local_optimization(initial_circles, remaining_time):
    """Enhanced refinement using multiple optimization approaches"""
    n = len(initial_circles)
    
    # Try multiple approaches and return the best
    best_result = initial_circles.copy()
    best_radius_sum = np.sum(initial_circles[:, 2])
    
    # Approach 1: SLSQP optimization
    try:
        result1 = optimize_with_slqp(initial_circles)
        if result1 is not None:
            radius_sum = np.sum(result1[:, 2])
            if radius_sum > best_radius_sum:
                best_result = result1
                best_radius_sum = radius_sum
    except:
        pass
    
    # Approach 2: Sequential local optimization (greedy improvement)
    try:
        result2 = sequential_local_improvement(initial_circles)
        if result2 is not None:
            radius_sum = np.sum(result2[:, 2])
            if radius_sum > best_radius_sum:
                best_result = result2
                best_radius_sum = radius_sum
    except:
        pass
    
    return best_result

def optimize_with_slqp(initial_circles):
    """Use SLSQP with better constraint handling"""
    n = len(initial_circles)
    
    # Flatten for optimization
    x0 = initial_circles.flatten()
    
    def objective(x_flat):
        # Extract circles
        circles = x_flat.reshape(-1, 3)
        # Maximize sum of radii (minimize negative)
        return -np.sum(circles[:, 2])
    
    def constraint_func(x_flat):
        """Return constraint violations (positive means violation)"""
        circles = x_flat.reshape(-1, 3)
        constraints = []
        
        # Boundary constraints: x-r >= 0, y-r >= 0, 1-x-r >= 0, 1-y-r >= 0
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([
                x - r,           # x - r >= 0
                y - r,           # y - r >= 0  
                1 - x - r,       # 1 - x - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        
        # Overlap constraints: distance - r1 - r2 >= 0
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(dist - r1 - r2)
                
        return np.array(constraints)
    
    # Set up bounds with tighter ranges
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Set up constraints
    cons = [{'type': 'ineq', 'fun': constraint_func}]
    
    # Run optimization with better settings
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                     options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6})
    
    if result.success:
        final_circles = result.x.reshape(-1, 3)
        return final_circles
    
    return None

def sequential_local_improvement(initial_circles):
    """Perform sequential improvement by optimizing one circle at a time"""
    circles = initial_circles.copy()
    n = len(circles)
    
    # Try improving each circle sequentially
    for iteration in range(50):  # Limited iterations
        improved = False
        for i in range(n):
            # Save current circle
            current_circle = circles[i].copy()
            
            # Try to improve this circle's position and radius
            best_circle = current_circle.copy()
            best_radius_sum = np.sum(circles[:, 2])
            
            # Create a neighborhood around current circle
            x, y, r = current_circle
            
            # Try different positions and radii
            for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dr in [-0.01, -0.005, 0, 0.005, 0.01]:
                        new_x = max(0.001, min(0.999, x + dx))
                        new_y = max(0.001, min(0.999, y + dy))
                        new_r = max(0.001, min(0.499, r + dr))
                        
                        # Check if new circle violates constraints
                        valid = True
                        
                        # Check boundary constraints
                        if new_x - new_r < 0 or new_x + new_r > 1 or \
                           new_y - new_r < 0 or new_y + new_r > 1:
                            valid = False
                        
                        # Check overlap constraints
                        if valid:
                            for j in range(n):
                                if i != j:
                                    x2, y2, r2 = circles[j]
                                    dist = np.sqrt((new_x - x2)**2 + (new_y - y2)**2)
                                    if dist < new_r + r2:
                                        valid = False
                                        break
                        
                        if valid:
                            # Temporarily update this circle
                            circles[i] = [new_x, new_y, new_r]
                            new_sum = np.sum(circles[:, 2])
                            
                            if new_sum > best_radius_sum:
                                best_radius_sum = new_sum
                                best_circle = [new_x, new_y, new_r]
                            
                            # Restore previous state
                            circles[i] = current_circle
            
            # Update to best found
            circles[i] = best_circle
            if not np.array_equal(best_circle, current_circle):
                improved = True
        
        # Stop if no improvement was made
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
