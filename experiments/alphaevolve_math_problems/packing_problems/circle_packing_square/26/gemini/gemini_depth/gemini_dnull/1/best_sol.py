# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from math import sqrt # Used for sqrt in fitness function

# Import DEAP components
from deap import base, creator, tools, algorithms

# Import scipy.optimize components for local search
from scipy.optimize import minimize, NonlinearConstraint

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# Constants for the problem
N_CIRCLES = 26
UNIT_SQUARE_SIZE = 1.0
MIN_RADIUS_ALLOWED = 1e-6 # Minimum possible radius, practically > 0
MAX_RADIUS_ALLOWED = 0.5   # Maximum possible radius for a circle in a unit square (cannot exceed 0.5 for x,y placement)

# --- DEAP setup ---
# Define fitness: We want to maximize the sum of radii. A positive weight means maximization.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Define individual: a numpy array representing all circle parameters (x1,y1,r1, x2,y2,r2, ...)
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# --- Helper functions for individual generation and mutation ---

# --- Helper functions for individual generation ---

def _generate_circle_params(min_r_val, max_r_val):
    """
    Generates x, y, r for a single circle respecting basic bounds.
    The center (x,y) must be at least 'r' from any edge, so x in [r, 1-r], y in [r, 1-r].
    """
    r = np.random.uniform(min_r_val, max_r_val)
    # If r > 0.5, then 1.0 - r < r, making the range invalid. MAX_RADIUS_ALLOWED handles this.
    x = np.random.uniform(r, UNIT_SQUARE_SIZE - r)
    y = np.random.uniform(r, UNIT_SQUARE_SIZE - r)
    return [x, y, r]

def init_individual(icls, n_circles, min_r, max_r, init_strategy='grid_perturb'):
    """
    Initializes an individual (flattened array of circle parameters) with random valid values
    or a grid-based arrangement with perturbation.
    """
    flat_params = []
    if init_strategy == 'random':
        for _ in range(n_circles):
            flat_params.extend(_generate_circle_params(min_r, max_r))
    elif init_strategy == 'grid_perturb':
        # Determine grid dimensions
        num_cols = int(np.ceil(np.sqrt(n_circles)))
        num_rows = int(np.ceil(n_circles / num_cols))
        
        # Target radius based on average optimal radius for 26 circles
        # Optimal sum_radii is around 2.63, so average r is ~0.101
        # Use a slightly higher value to allow for expansion/overlap to be optimized out.
        target_r = 0.105 
        
        # Ensure target_r is within min_r, max_r
        target_r = np.clip(target_r, min_r, max_r)

        current_circle_idx = 0
        # Calculate ideal step size for centers to fill the square
        x_spacing = UNIT_SQUARE_SIZE / num_cols
        y_spacing = UNIT_SQUARE_SIZE / num_rows
        
        for i in range(num_rows):
            for j in range(num_cols):
                if current_circle_idx < n_circles:
                    # Base center for this grid cell
                    x_base = (j * x_spacing) + (x_spacing / 2)
                    y_base = (i * y_spacing) + (y_spacing / 2)
                    
                    # Perturb positions and radius slightly
                    # Use a smaller sigma for perturbation to keep them somewhat structured
                    x = np.random.normal(x_base, x_spacing * 0.05) # 5% of cell width perturbation
                    y = np.random.normal(y_base, y_spacing * 0.05)
                    r = np.random.normal(target_r, target_r * 0.05) # 5% perturbation on radius
                    
                    # Apply clamping to ensure validity *before* adding to flat_params
                    r = np.clip(r, min_r, max_r)
                    x = np.clip(x, r, UNIT_SQUARE_SIZE - r)
                    y = np.clip(y, r, UNIT_SQUARE_SIZE - r)
                    
                    flat_params.extend([x, y, r])
                    current_circle_idx += 1
                else:
                    break
            if current_circle_idx >= n_circles:
                break
    else:
        raise ValueError("Unknown initialization strategy")
    
    return icls(np.array(flat_params, dtype=np.float64))

# --- Helper functions for local optimization (scipy.optimize.minimize) ---

def _objective(params):
    """Minimize negative sum of radii."""
    radii = params[2::3] # Every 3rd element starting from index 2 is a radius
    return -np.sum(radii)

def _containment_constraints(params):
    """
    Returns an array of values for containment constraints.
    All values must be >= 0 for a feasible solution.
    ri <= xi <= 1-ri  =>  xi - ri >= 0  AND  1 - xi - ri >= 0
    ri <= yi <= 1-ri  =>  yi - ri >= 0  AND  1 - yi - ri >= 0
    """
    n = N_CIRCLES
    x_coords = params[0::3]
    y_coords = params[1::3]
    radii = params[2::3]

    constraints = []
    for i in range(n):
        constraints.append(x_coords[i] - radii[i])       # xi - ri >= 0
        constraints.append(UNIT_SQUARE_SIZE - x_coords[i] - radii[i]) # 1 - xi - ri >= 0
        constraints.append(y_coords[i] - radii[i])       # yi - ri >= 0
        constraints.append(UNIT_SQUARE_SIZE - y_coords[i] - radii[i]) # 1 - yi - ri >= 0
    return np.array(constraints)

def _overlap_constraints(params):
    """
    Returns an array of values for non-overlap constraints.
    sqrt[(xi-xj)² + (yi-yj)²] - (ri + rj) >= 0 for i != j
    """
    n = N_CIRCLES
    x_coords = params[0::3]
    y_coords = params[1::3]
    radii = params[2::3]
    
    constraints = []
    # Iterate over unique pairs (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2
            min_dist = radii[i] + radii[j]
            constraints.append(np.sqrt(dist_sq) - min_dist) # dist_ij - (ri + rj) >= 0
    return np.array(constraints)

# Function to apply local optimization
def _apply_local_optimization(individual):
    """
    Applies scipy.optimize.minimize to fine-tune the best individual.
    """
    x0 = individual.copy()

    # Bounds for x, y, r for each circle
    # x in [0, 1], y in [0, 1], r in [MIN_RADIUS_ALLOWED, MAX_RADIUS_ALLOWED]
    bounds_list = [(0, UNIT_SQUARE_SIZE), (0, UNIT_SQUARE_SIZE), (MIN_RADIUS_ALLOWED, MAX_RADIUS_ALLOWED)] * N_CIRCLES
    
    # Define containment constraints (g(x) >= 0)
    containment_nlc = NonlinearConstraint(
        _containment_constraints, 0, np.inf, # lower bound = 0, upper bound = infinity
        jac='2-point' # Use finite difference approximation for Jacobian
    )

    # Define non-overlap constraints (g(x) >= 0)
    overlap_nlc = NonlinearConstraint(
        _overlap_constraints, 0, np.inf, # lower bound = 0, upper bound = infinity
        jac='2-point' # Use finite difference approximation for Jacobian
    )

    constraints = [containment_nlc, overlap_nlc]

    # Perform optimization using SLSQP
    res = minimize(
        _objective, x0,
        method='SLSQP',
        bounds=bounds_list,
        constraints=constraints,
        options={'disp': False, 'maxiter': 700, 'ftol': 1e-8} # Increased maxiter, tighter ftol for precision
    )
    
    if res.success:
        # Update the individual with the optimized parameters
        individual[:] = res.x
        # Recalculate fitness for the optimized solution using the EA's fitness function
        # This is crucial because penalties are not part of scipy's objective,
        # but DEAP needs a valid fitness value and ensures feasibility.
        individual.fitness.values = evaluate_circles(individual)
        # print(f"Local optimization successful. New fitness: {individual.fitness.values[0]}")
        return individual
    else:
        # If local optimization fails, return the original individual
        # print(f"Local optimization failed: {res.message}")
        return individual # Return original if optimization fails

def mutate_individual(individual, mu, sigma, indpb, min_r, max_r):
    """
    Mutates an individual using Gaussian perturbation for each parameter and ensures bounds.
    mu: mean of the Gaussian distribution (typically 0 for perturbation)
    sigma: standard deviation of the Gaussian distribution
    indpb: independent probability for each of the (N_CIRCLES * 3) attributes to be mutated.
           Note: DEAP's tools.mutGaussian mutates each element with indpb. Here we mutate
           a whole circle's (x,y,r) with probability indpb for simplicity or each component.
           Let's clarify: the prompt suggests 'perturbing x, y, r values'.
           We will apply mutation to each (x,y,r) component with 'indpb'.
    min_r, max_r: minimum and maximum allowed radius values.
    """
    # Reshape for easier access to (x,y,r) tuples
    circles = individual.reshape((N_CIRCLES, 3))
    
    for i in range(N_CIRCLES):
        # Mutate x, y, r components with probability indpb
        # Each component (x, y, r) has an independent chance of mutation.
        # This is more aligned with standard `tools.mutGaussian` behavior when applied to a flat array.
        
        # Mutate x
        if random.random() < indpb:
            circles[i, 0] = np.random.normal(circles[i, 0], sigma)
        # Mutate y
        if random.random() < indpb:
            circles[i, 1] = np.random.normal(circles[i, 1], sigma)
        # Mutate r
        if random.random() < indpb:
            circles[i, 2] = np.random.normal(circles[i, 2], sigma)

        # Apply clamping to ensure all parameters are within their fundamental ranges
        # Clamp x, y to [0, 1]
        circles[i, 0] = np.clip(circles[i, 0], 0, UNIT_SQUARE_SIZE)
        circles[i, 1] = np.clip(circles[i, 1], 0, UNIT_SQUARE_SIZE)
        
        # Clamp r to [min_r, max_r]
        circles[i, 2] = np.clip(circles[i, 2], min_r, max_r)
        
        # Further constraint: the center (x,y) must be at least 'r' distance from any edge.
        # This is enforced by clamping x,y to the valid range [r, 1-r] using the circle's current radius.
        r_val = circles[i, 2]
        circles[i, 0] = np.clip(circles[i, 0], r_val, UNIT_SQUARE_SIZE - r_val)
        circles[i, 1] = np.clip(circles[i, 1], r_val, UNIT_SQUARE_SIZE - r_val)
            
    # Modify the original individual in-place with the flattened (mutated) array
    individual[:] = circles.flatten()
    return individual, # DEAP mutation functions expect a tuple return

# --- Fitness function ---
# Penalty factors for constraint violations
OOB_PENALTY_FACTOR = 1000.0 # Penalty per unit of out-of-bounds violation
OVERLAP_PENALTY_FACTOR = 10000.0 # Penalty per unit of overlap distance

def evaluate_circles(individual):
    """
    Evaluates the fitness of an individual (set of circles).
    Fitness = sum_radii - penalty_for_violations.
    """
    circles = individual.reshape((N_CIRCLES, 3))
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]

    # 1. Sum of radii (primary objective)
    sum_radii = np.sum(radii)

    total_penalty = 0.0

    # 2. Containment constraint: ri <= xi <= 1-ri and ri <= yi <= 1-ri
    # Penalize if any part of the circle is outside the square.
    # The mutation and initialization functions try to enforce this, but penalties ensure robust handling.
    
    # Calculate violations for x-coordinates
    x_low_violation = np.maximum(0, radii - x_coords) # x_i < r_i
    x_high_violation = np.maximum(0, x_coords + radii - UNIT_SQUARE_SIZE) # x_i + r_i > 1
    
    # Calculate violations for y-coordinates
    y_low_violation = np.maximum(0, radii - y_coords) # y_i < r_i
    y_high_violation = np.maximum(0, y_coords + radii - UNIT_SQUARE_SIZE) # y_i + r_i > 1

    total_penalty += OOB_PENALTY_FACTOR * np.sum(x_low_violation + x_high_violation + y_low_violation + y_high_violation)

    # Penalize if radii are too small (effectively zero, which is not useful)
    radius_minimum_violation = np.maximum(0, MIN_RADIUS_ALLOWED - radii)
    total_penalty += OOB_PENALTY_FACTOR * np.sum(radius_minimum_violation) * 10 # Higher penalty for tiny radii

    # 3. Non-overlap constraint: sqrt[(xi-xj)² + (yi-yj)²] >= ri + rj for all i≠j
    # Compute pairwise Euclidean distances squared
    coords = circles[:, :2] # (N_CIRCLES, 2) array of (x,y)
    
    # Efficient pairwise distance calculation using broadcasting
    # diff_coords[i, j, :] = coords[i, :] - coords[j, :]
    diff_coords = coords[:, np.newaxis, :] - coords[np.newaxis, :, :] # (N_CIRCLES, N_CIRCLES, 2)
    dist_sq_matrix = np.sum(diff_coords**2, axis=2) # (N_CIRCLES, N_CIRCLES) matrix of squared distances

    # Compute pairwise sum of radii
    sum_radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :] # (N_CIRCLES, N_CIRCLES) matrix of (ri + rj)

    # Calculate overlap violation for unique pairs (i < j)
    # An overlap occurs if dist_ij < r_i + r_j. The violation is (r_i + r_j) - dist_ij.
    # We use np.sqrt(dist_sq_matrix) to get actual distances.
    
    # Ensure numerical stability for sqrt(0) which may occur on diagonal (i==j)
    # However, np.maximum(0, ...) will handle this as diag will be sum_radii - 0.
    # We only care about off-diagonal elements for overlap.
    
    # overlap_violations_matrix[i,j] = max(0, (r_i + r_j) - distance_ij)
    overlap_violations_matrix = np.maximum(0, sum_radii_matrix - np.sqrt(dist_sq_matrix))
    
    # Sum up violations. We only need to consider the upper triangle (excluding diagonal) to avoid
    # double-counting pairs and self-comparison.
    upper_tri_indices = np.triu_indices(N_CIRCLES, k=1) # k=1 excludes the diagonal
    total_penalty += OVERLAP_PENALTY_FACTOR * np.sum(overlap_violations_matrix[upper_tri_indices])

    # Final fitness value: sum of radii penalized by constraint violations
    fitness = sum_radii - total_penalty
    return fitness, # DEAP expects fitness to be a tuple

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Implements an Evolutionary Algorithm using the DEAP framework.
    """
    start_time = time.time()

    # --- DEAP Toolbox Setup ---
    toolbox = base.Toolbox()

    # Register individual and population generators, using grid_perturb strategy for better initial population
    toolbox.register("individual", init_individual, creator.Individual, N_CIRCLES, MIN_RADIUS_ALLOWED, MAX_RADIUS_ALLOWED, init_strategy='grid_perturb')
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic operators
    toolbox.register("evaluate", evaluate_circles)
    toolbox.register("mate", tools.cxBlend, alpha=0.5) # Blend crossover, good for real-valued attributes
    toolbox.register("mutate", mutate_individual, mu=0, sigma=0.03, indpb=0.1, min_r=MIN_RADIUS_ALLOWED, max_r=MAX_RADIUS_ALLOWED)
    toolbox.register("select", tools.selTournament, tournsize=3) # Tournament selection

    # --- EA Parameters ---
    POPULATION_SIZE = 1500 # Increased population size for better exploration
    GENERATIONS = 1200 # Number of generations to run, adjusted for increased pop size
    CXPB = 0.7         # Crossover probability
    MUTPB = 0.1        # Mutation probability (reduced slightly to favor good individuals from crossover)
    # Mutation specific parameters for `mutate_individual`
    MUT_SIGMA = 0.05   # Larger sigma for mutation to allow more significant changes
    MUT_INDPB = 0.05   # Reduced individual component mutation probability to preserve structure

    # Register mutation with updated parameters
    toolbox.register("mutate", mutate_individual, mu=0, sigma=MUT_SIGMA, indpb=MUT_INDPB, min_r=MIN_RADIUS_ALLOWED, max_r=MAX_RADIUS_ALLOWED)
    
    # Set fixed random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialize population
    population = toolbox.population(n=POPULATION_SIZE)

    # Evaluate the entire initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Initialize statistics to track evolution progress
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Logbook for storing statistics
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Record initial population statistics
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    # print(logbook.stream)

    # Main evolutionary loop
    for gen in range(1, GENERATIONS + 1):
        # Check for remaining time before starting a new generation
        elapsed_time = time.time() - start_time
        # Allow some buffer for local optimization and final processing
        if elapsed_time > 170: # Stop EA slightly earlier to leave time for local search
            print(f"Stopping EA early at generation {gen} due to time limit.")
            break

        # Select the next generation of individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals to avoid modifying parents directly
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover on selected offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # Invalidate fitness values of modified individuals
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on offspring
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate individuals with invalid fitness (newly created or mutated)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the current population with the offspring
        population[:] = offspring

        # Gather and record statistics for the current generation
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        # print(logbook.stream)

    # Select the best individual from the final population
    best_individual = tools.selBest(population, 1)[0]
    
    # --- Apply Local Optimization to the best individual ---
    # print(f"Applying local optimization to the best individual from EA (initial sum_radii: {np.sum(best_individual.reshape((N_CIRCLES, 3))[:, 2]):.4f})...")
    # Make a copy to pass to local optimization, so original best_individual is preserved if local opt fails
    optimized_individual = _apply_local_optimization(best_individual.copy())
    
    final_circles = optimized_individual.reshape((N_CIRCLES, 3))

    end_time = time.time()
    # print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    # print(f"Best fitness after local optimization: {optimized_individual.fitness.values[0]:.4f}")
    # print(f"Sum of radii of final circles: {np.sum(final_circles[:, 2]):.4f}")

    return final_circles


# EVOLVE-BLOCK-END
