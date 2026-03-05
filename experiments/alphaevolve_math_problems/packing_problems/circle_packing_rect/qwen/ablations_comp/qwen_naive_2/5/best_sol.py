# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import time
import random
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from scipy.spatial import Voronoi
import warnings
from scipy.spatial.distance import pdist, squareform

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid evolutionary algorithm with geometric initialization and improved constraint handling.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: perimeter = 4, so width + height = 2
    # Try different aspect ratios to find optimal one
    best_ratio = 1.0  # Square case
    width, height = 1.0, 1.0  # Start with square
    
    # Parameters for evolutionary algorithm
    POP_SIZE = 150
    NGEN = 200
    MUTPB = 0.25
    CXPB = 0.8
    
    # Create individual and population classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define bounds for each variable: [x1, y1, r1, x2, y2, r2, ...]
    def create_individual():
        """Create an individual with 21 circles using enhanced initialization"""
        individual = []
        
        # Try different strategies for better initialization
        # Strategy 1: Hexagonal packing pattern for better density
        n_rows = 5
        n_cols = 5
        padding = 0.05
        
        # Calculate cell size based on rectangle dimensions
        cell_width = (width - 2*padding) / n_cols
        cell_height = (height - 2*padding) / n_rows
        
        # Generate hexagonal grid points
        grid_points = []
        for i in range(n_rows):
            for j in range(n_cols):
                # Offset every other row for hexagonal packing
                x_offset = (j + 0.5) * cell_width
                if i % 2 == 1:
                    x_offset += cell_width / 2
                
                y = padding + (i + 0.5) * cell_height
                x = x_offset
                
                if x <= width - padding and y <= height - padding:
                    grid_points.append((x, y))
        
        # Use the first 21 grid positions
        for i in range(min(21, len(grid_points))):
            x, y = grid_points[i]
            # Add slight randomness to avoid perfect grid
            x += random.uniform(-cell_width/6, cell_width/6)
            y += random.uniform(-cell_height/6, cell_height/6)
            
            # Radius constrained by boundaries
            max_radius = min(x, width - x, y, height - y)
            # Start with a reasonable initial radius
            r = random.uniform(0.01, min(max_radius, 0.12))
            individual.extend([x, y, r])
        
        # Fill remaining circles with random placement but better distribution
        remaining = 21 - len(grid_points)
        if remaining > 0:
            for i in range(remaining):
                # Distribute more evenly using rejection sampling
                attempts = 0
                placed = False
                while not placed and attempts < 100:
                    x = random.uniform(padding, width - padding)
                    y = random.uniform(padding, height - padding)
                    
                    # Check if this position is far enough from existing circles
                    valid_position = True
                    for j in range(0, len(individual), 3):
                        existing_x, existing_y = individual[j], individual[j+1]
                        dist = math.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        # Need to be at least 0.02 away from any other circle center
                        if dist < 0.02:
                            valid_position = False
                            break
                    
                    if valid_position:
                        max_radius = min(x, width - x, y, height - y)
                        r = random.uniform(0.01, min(max_radius, 0.1))
                        individual.extend([x, y, r])
                        placed = True
                    attempts += 1
                    
                if not placed:
                    # Fallback to simple random placement
                    x = random.uniform(padding, width - padding)
                    y = random.uniform(padding, height - padding)
                    max_radius = min(x, width - x, y, height - y)
                    r = random.uniform(0.01, min(max_radius, 0.1))
                    individual.extend([x, y, r])
            
        return creator.Individual(individual)
    
    def evaluate(individual):
        """Evaluate fitness of individual with improved penalty system"""
        circles = np.array(individual).reshape(-1, 3)
        
        # Check boundary constraints
        for i in range(21):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return (float('-inf'),)  # Invalid solution
        
        # Check overlap constraints with penalty
        total_radius = np.sum(circles[:, 2])
        
        # Compute penalty for overlaps with more sophisticated approach
        penalty = 0
        # Precompute distances for efficiency
        distances = cdist(circles[:, :2], circles[:, :2])
        
        # Use more aggressive penalty for overlaps
        for i in range(21):
            for j in range(i+1, 21):
                dist = distances[i, j]
                r1, r2 = circles[i, 2], circles[j, 2]
                if dist < (r1 + r2):
                    # More aggressive penalty based on how much they overlap
                    overlap = (r1 + r2) - dist
                    penalty += overlap**3 * 5000  # Cubic penalty for severe overlaps
        
        # Add penalty for small radii to encourage larger circles
        small_radius_penalty = 0
        for i in range(21):
            _, _, r = circles[i]
            if r < 0.03:
                small_radius_penalty += (0.03 - r) * 500
                
        # Return fitness (total radius minus penalties)
        return (total_radius - penalty - small_radius_penalty,)
    
    def mutate(individual):
        """Mutate an individual with adaptive mutation rate and better strategy"""
        # Adaptive mutation rate based on generation
        adaptive_mutation_rate = MUTPB * (1 - 0.8 * (len(individual) / (len(individual) * 0.3)))
        
        for i in range(len(individual)):
            if random.random() < adaptive_mutation_rate:  # Variable mutation rate
                if i % 3 == 0:  # x coordinate
                    individual[i] = random.uniform(0.01, width - 0.01)
                elif i % 3 == 1:  # y coordinate
                    individual[i] = random.uniform(0.01, height - 0.01)
                else:  # radius
                    # Make sure radius stays within bounds
                    x, y = individual[i-2], individual[i-1]
                    max_radius = min(x, width - x, y, height - y)
                    individual[i] = random.uniform(0.005, min(max_radius, 0.15))
        return individual,
    
    def crossover(ind1, ind2):
        """Crossover two individuals with uniform crossover"""
        # Uniform crossover
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
    
    # Initialize population
    pop = toolbox.population(n=POP_SIZE)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Evolution loop with improved early stopping criteria
    best_fitness_history = []
    stagnation_count = 0
    max_stagnation = 15
    
    for gen in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
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
        
        # Replace population with offspring
        pop[:] = offspring
        
        # Track best fitness
        current_best = max(ind.fitness.values[0] for ind in pop)
        best_fitness_history.append(current_best)
        
        # Check for stagnation
        if len(best_fitness_history) > 10:
            recent_improvement = current_best - best_fitness_history[-10]
            if recent_improvement < 1e-5:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            if stagnation_count >= max_stagnation:
                # Restart with new population if stagnating
                pop = toolbox.population(n=POP_SIZE)
                stagnation_count = 0
    
    # Find the best individual
    best_ind = tools.selBest(pop, 1)[0]
    best_circles = np.array(best_ind).reshape(-1, 3)
    
    # Enhanced refinement with better optimization approach
    def objective(params):
        circles_flat = params.reshape(-1, 3)
        radii = circles_flat[:, 2]
        return -np.sum(radii)  # Negative because we want to maximize
    
    def constraint_func(params):
        circles_flat = params.reshape(-1, 3)
        constraints = []
        
        # Pairwise distance constraints (no overlaps)
        for i, j in combinations(range(21), 2):
            x1, y1, r1 = circles_flat[i]
            x2, y2, r2 = circles_flat[j]
            
            # Distance between centers
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            # Constraint: distance >= sum of radii (no overlap)
            constraints.append(dist - (r1 + r2))
        
        # Boundary constraints: circles must be within rectangle
        for i in range(21):
            x, y, r = circles_flat[i]
            # Left boundary
            constraints.append(x - r)
            # Right boundary  
            constraints.append(width - x - r)
            # Bottom boundary
            constraints.append(y - r)
            # Top boundary
            constraints.append(height - y - r)
            
        return np.array(constraints)
    
    # Use the best evolutionary solution as starting point for local optimization
    initial_params = best_circles.flatten()
    
    # Set up constraints
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Bounds for parameters - tighter bounds
    bounds = []
    for i in range(21):
        # x, y, r for each circle - more constrained
        bounds.extend([(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)])
    
    # Optimization options - more robust settings
    options = {'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
    
    try:
        # Try multiple optimization approaches
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options=options,
            tol=1e-8
        )
        
        if result.success:
            refined_circles = result.x.reshape(-1, 3)
            # Ensure refinement actually improved
            if np.sum(refined_circles[:, 2]) > np.sum(best_circles[:, 2]) * 1.001:  # Small threshold
                return refined_circles
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
        pass
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
