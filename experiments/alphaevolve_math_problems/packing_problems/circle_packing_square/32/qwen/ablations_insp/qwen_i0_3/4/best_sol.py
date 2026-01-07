# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from deap import base, creator, tools, algorithms
import time

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithm with local refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles using a more sophisticated approach
    circles = initialize_advanced_circles(n)
    
    # First optimize using scipy to get a good baseline
    circles = optimize_circles_scipy(circles)
    
    # Then refine with evolutionary algorithm for better results
    circles = optimize_circles_evolutionary(circles)
    
    return circles

def initialize_advanced_circles(n: int) -> np.ndarray:
    """Initialize circle positions using a more strategic approach."""
    # Start with a hexagonal packing pattern to get good initial distribution
    # This helps avoid poor local optima that regular grids can cause
    
    # Create a hexagonal lattice pattern
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Adjust for hexagonal packing
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1) * np.sqrt(3)/2
    
    # For hexagonal packing, alternate rows are shifted
    circles = np.zeros((n, 3))
    
    for i in range(n):
        row = i // cols
        col = i % cols
        # Alternate columns for hexagonal pattern
        x = (col + 1) * spacing_x
        if row % 2 == 1:
            x += spacing_x * 0.5
        y = (row + 1) * spacing_y
        # Initial radius - start with a reasonable value based on spacing
        r = min(spacing_x, spacing_y) * 0.3
        circles[i] = [x, y, r]
    
    # Apply some randomness to avoid getting stuck in symmetric solutions
    np.random.seed(42)
    for i in range(n):
        circles[i, 0] += np.random.normal(0, 0.01 * spacing_x)
        circles[i, 1] += np.random.normal(0, 0.01 * spacing_y)
        circles[i, 2] *= (0.9 + np.random.random() * 0.2)  # Small variation in radii
    
    return circles

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate the sum of all circle radii."""
    return np.sum(circles[:, 2])

def check_constraints(circles: np.ndarray) -> tuple:
    """Check if all constraints are satisfied and return violation metrics."""
    n = circles.shape[0]
    violations = []
    
    # Check containment constraints
    for i in range(n):
        ci = circles[i]
        min_dist_to_boundary = min(ci[0], ci[1], 1-ci[0], 1-ci[1])
        if min_dist_to_boundary < ci[2]:
            violations.append(('containment', i, ci[2] - min_dist_to_boundary))
    
    # Check overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            ci = circles[i]
            cj = circles[j]
            dist = np.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
            overlap = dist - (ci[2] + cj[2])
            if overlap < 0:
                violations.append(('overlap', i, j, -overlap))
    
    return violations

def evaluate_fitness(individual: np.ndarray) -> float:
    """Evaluate fitness of an individual (negative sum of radii for maximization)."""
    # Reshape individual back to circles
    n = len(individual) // 3
    circles = individual.reshape(n, 3)
    
    # Check constraints and penalize heavily for violations
    violations = check_constraints(circles)
    if violations:
        # Heavy penalty for constraint violations
        penalty = sum(violation[-1] for violation in violations) * 1000
        return -np.sum(circles[:, 2]) + penalty
    
    return -np.sum(circles[:, 2])

def mutate_individual(individual: np.ndarray, indpb: float = 0.1) -> np.ndarray:
    """Mutate an individual by slightly perturbing positions and radii."""
    mutated = individual.copy()
    n = len(individual) // 3
    
    for i in range(n):
        # Mutate position and radius
        for j in range(3):  # x, y, r
            if np.random.random() < indpb:
                if j < 2:  # Position coordinates
                    mutated[i*3 + j] += np.random.normal(0, 0.01)
                    # Keep within bounds
                    mutated[i*3 + j] = np.clip(mutated[i*3 + j], 0.001, 0.999)
                else:  # Radius
                    mutated[i*3 + j] += np.random.normal(0, 0.005)
                    # Keep positive
                    mutated[i*3 + j] = max(0.001, mutated[i*3 + j])
    
    return mutated

def create_constraints(n: int) -> list:
    """Create constraint dictionaries for scipy optimization."""
    constraints = []
    
    # Add containment constraints
    for i in range(n):
        def contain_constraint(x, i=i):
            # Extract circle parameters
            ci = x[i*3:i*3+3]
            # Distance to boundaries
            min_dist = min(ci[0], ci[1], 1-ci[0], 1-ci[1])
            return min_dist - ci[2]
        
        constraints.append({'type': 'ineq', 'fun': contain_constraint})
    
    # Add non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(x, i=i, j=j):
                # Extract circle parameters
                ci = x[i*3:i*3+3]
                cj = x[j*3:j*3+3]
                # Distance between centers minus sum of radii
                dist = np.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
                return dist - (ci[2] + cj[2])
            
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    return constraints

def objective_function(circles_flat: np.ndarray) -> float:
    """Objective function to maximize (negative because scipy minimizes)."""
    # Reshape flat array back to circles
    n = len(circles_flat) // 3
    circles = circles_flat.reshape(n, 3)
    # Negative because we want to maximize sum of radii
    return -np.sum(circles[:, 2])

def optimize_circles_scipy(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using scipy."""
    n = initial_circles.shape[0]
    
    # Flatten initial circles for scipy optimization
    initial_flat = initial_circles.flatten()
    
    # Create constraints
    constraints = create_constraints(n)
    
    # Set bounds for each variable (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0.001, 0.999))  # Avoid exact boundaries
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds
        bounds.append((0.001, 0.5))  # Reasonable upper bound
    
    # Use SLSQP optimizer which handles constraints well
    try:
        result = minimize(
            objective_function,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(n, 3)
            # Ensure valid ranges
            optimized_circles[:, 0] = np.clip(optimized_circles[:, 0], 0.001, 0.999)
            optimized_circles[:, 1] = np.clip(optimized_circles[:, 1], 0.001, 0.999)
            optimized_circles[:, 2] = np.clip(optimized_circles[:, 2], 0.001, 0.5)
            return optimized_circles
    except Exception as e:
        print(f"Scipy optimization failed: {e}")
    
    # If optimization fails, return initial circles
    return initial_circles

def optimize_circles_evolutionary(initial_circles: np.ndarray) -> np.ndarray:
    """Refine solution using evolutionary algorithm."""
    n = initial_circles.shape[0]
    
    # Setup DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, 
                     lambda: initial_circles.flatten())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", mutate_individual, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    pop = toolbox.population(n=50)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)
    
    # Evolution parameters
    CXPB = 0.5   # crossover probability
    MUTPB = 0.2  # mutation probability
    NGEN = 50    # number of generations
    
    # Run evolution
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
            ind.fitness.values = (fit,)
        
        # Replace the old population with the new population
        pop[:] = offspring
    
    # Get best individual
    best_ind = tools.selBest(pop, 1)[0]
    optimized_circles = best_ind.reshape(n, 3)
    
    # Ensure valid ranges
    optimized_circles[:, 0] = np.clip(optimized_circles[:, 0], 0.001, 0.999)
    optimized_circles[:, 1] = np.clip(optimized_circles[:, 1], 0.001, 0.999)
    optimized_circles[:, 2] = np.clip(optimized_circles[:, 2], 0.001, 0.5)
    
    return optimized_circles


# EVOLVE-BLOCK-END
