# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time
from deap import base, creator, tools, algorithms
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.spatial.distance import pdist
import multiprocessing # Added for parallelization
import os # Added for CPU count

# --- Global Constants ---
N_CIRCLES = 32
IND_SIZE = N_CIRCLES * 3
RANDOM_SEED = 42
MIN_RADIUS_CONSTRAINT = 1e-7
PENALTY_FACTOR_CONTAINMENT = 10000.0
PENALTY_FACTOR_OVERLAP = 10000.0

# --- DEAP Setup ---
# Use a try-except block to prevent errors on re-runs in some environments.
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass

# --- DEAP Fitness Function ---
def evaluate_circles(individual):
    """Fitness function for the GA, maximizing radii sum while penalizing violations."""
    circles = np.array(individual).reshape(N_CIRCLES, 3)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    sum_radii = np.sum(np.maximum(r, 0))
    total_penalty = 0.0

    # Penalty for negative radii
    total_penalty += np.sum(np.maximum(0, -r)) * PENALTY_FACTOR_CONTAINMENT * 10
    
    # Penalty for containment violations (circles outside the square)
    containment_viol = np.sum(np.maximum(0, r - x)**2 + np.maximum(0, x + r - 1)**2 +
                              np.maximum(0, r - y)**2 + np.maximum(0, y + r - 1)**2)
    total_penalty += containment_viol * PENALTY_FACTOR_CONTAINMENT

    # Penalty for overlaps
    if N_CIRCLES > 1:
        centers = circles[:, :2]
        dists = pdist(centers)
        radii_sums = np.add.outer(r, r)[np.triu_indices(N_CIRCLES, k=1)]
        overlaps = radii_sums - dists
        overlap_viol = np.sum(overlaps[overlaps > 0]**2)
        total_penalty += overlap_viol * PENALTY_FACTOR_OVERLAP

    return (sum_radii - total_penalty,)

# --- DEAP Toolbox Setup ---
toolbox = base.Toolbox()

def init_individual_grid(icls, attr_r):
    """Initializes individuals by placing centers on a grid for good separation."""
    ind = icls()
    grid_size = math.ceil(math.sqrt(N_CIRCLES))
    points = [( (i + 0.5) / grid_size, (j + 0.5) / grid_size )
              for i in range(grid_size) for j in range(grid_size)]
    selected_points = random.sample(points, N_CIRCLES)
    for x_coord, y_coord in selected_points:
        ind.extend([x_coord, y_coord, attr_r()])
    return ind

toolbox.register("attr_r", random.uniform, 0.001, 0.1)
toolbox.register("individual", init_individual_grid, creator.Individual, toolbox.attr_r)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_circles)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

def mutate_clipped(individual, sigma, indpb):
    """Gaussian mutation with clipping to stay within reasonable bounds."""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(0, sigma)
            coord_type = i % 3
            if coord_type == 0 or coord_type == 1: # x or y
                individual[i] = np.clip(individual[i], 0.0, 1.0)
            else: # r
                individual[i] = np.clip(individual[i], MIN_RADIUS_CONSTRAINT, 0.5)
    return individual,

# Register mutate without a fixed sigma, it will be passed dynamically
toolbox.register("mutate", mutate_clipped, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- SciPy Optimization Functions ---
def objective_scipy(params):
    """Objective: Minimize the negative sum of radii."""
    return -np.sum(params[2::3])

def containment_constraint_scipy(params):
    """Vectorized containment constraints (all must be >= 0)."""
    x, y, r = params[0::3], params[1::3], params[2::3]
    return np.concatenate((x - r, 1 - x - r, y - r, 1 - y - r, r - MIN_RADIUS_CONSTRAINT))

def non_overlap_constraint_scipy(params):
    """Numerically stable, vectorized non-overlap constraints using squared distances."""
    x, y, r = params[0::3], params[1::3], params[2::3]
    centers = np.column_stack((x, y))
    
    dist_sq = pdist(centers, 'sqeuclidean')
    
    radii_sum_matrix = r[:, np.newaxis] + r[np.newaxis, :]
    radii_sum_pairs = radii_sum_matrix[np.triu_indices(N_CIRCLES, k=1)]
    
    return dist_sq - radii_sum_pairs**2

# --- Global Constants ---
N_CIRCLES = 32
IND_SIZE = N_CIRCLES * 3
RANDOM_SEED = 42
MIN_RADIUS_CONSTRAINT = 1e-6 # Adjusted for slightly better numerical stability
PENALTY_FACTOR_CONTAINMENT = 10000.0
PENALTY_FACTOR_OVERLAP = 10000.0

# --- DEAP Setup ---
# Use a try-except block to prevent errors on re-runs in some environments.
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass

# --- DEAP Fitness Function ---
def evaluate_circles(individual):
    """Fitness function for the GA, maximizing radii sum while penalizing violations."""
    circles = np.array(individual).reshape(N_CIRCLES, 3)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    sum_radii = np.sum(np.maximum(r, 0))
    total_penalty = 0.0

    # Penalty for negative radii
    total_penalty += np.sum(np.maximum(0, -r)) * PENALTY_FACTOR_CONTAINMENT * 10
    
    # Penalty for containment violations (circles outside the square)
    containment_viol = np.sum(np.maximum(0, r - x)**2 + np.maximum(0, x + r - 1)**2 +
                              np.maximum(0, r - y)**2 + np.maximum(0, y + r - 1)**2)
    total_penalty += containment_viol * PENALTY_FACTOR_CONTAINMENT

    # Penalty for overlaps
    if N_CIRCLES > 1:
        centers = circles[:, :2]
        dists = pdist(centers)
        radii_sums = np.add.outer(r, r)[np.triu_indices(N_CIRCLES, k=1)]
        overlaps = radii_sums - dists
        overlap_viol = np.sum(overlaps[overlaps > 0]**2)
        total_penalty += overlap_viol * PENALTY_FACTOR_OVERLAP

    return (sum_radii - total_penalty,)

# --- DEAP Toolbox Setup ---
toolbox = base.Toolbox()

def init_individual_grid(icls, attr_r):
    """Initializes individuals by placing centers on a grid for good separation."""
    ind = icls()
    grid_size = math.ceil(math.sqrt(N_CIRCLES))
    points = [( (i + 0.5) / grid_size, (j + 0.5) / grid_size )
              for i in range(grid_size) for j in range(grid_size)]
    selected_points = random.sample(points, N_CIRCLES)
    for x_coord, y_coord in selected_points:
        ind.extend([x_coord, y_coord, attr_r()])
    return ind

toolbox.register("attr_r", random.uniform, 0.001, 0.1)
toolbox.register("individual", init_individual_grid, creator.Individual, toolbox.attr_r)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_circles)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

def mutate_clipped(individual, sigma, indpb):
    """Gaussian mutation with clipping to stay within reasonable bounds."""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(0, sigma)
            coord_type = i % 3
            if coord_type == 0 or coord_type == 1: # x or y
                individual[i] = np.clip(individual[i], 0.0, 1.0)
            else: # r
                individual[i] = np.clip(individual[i], MIN_RADIUS_CONSTRAINT, 0.5)
    return individual,

# Register mutate without a fixed sigma, it will be passed dynamically
toolbox.register("mutate", mutate_clipped, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- SciPy Optimization Functions ---
def objective_scipy(params):
    """Objective: Minimize the negative sum of radii."""
    return -np.sum(params[2::3])

def containment_constraint_scipy(params):
    """Vectorized containment constraints (all must be >= 0)."""
    x, y, r = params[0::3], params[1::3], params[2::3]
    return np.concatenate((x - r, 1 - x - r, y - r, 1 - y - r, r - MIN_RADIUS_CONSTRAINT))

def non_overlap_constraint_scipy(params):
    """Numerically stable, vectorized non-overlap constraints using squared distances."""
    x, y, r = params[0::3], params[1::3], params[2::3]
    centers = np.column_stack((x, y))
    
    dist_sq = pdist(centers, 'sqeuclidean')
    
    radii_sum_matrix = r[:, np.newaxis] + r[np.newaxis, :]
    radii_sum_pairs = radii_sum_matrix[np.triu_indices(N_CIRCLES, k=1)]
    
    return dist_sq - radii_sum_pairs**2

# --- Main Orchestrator Function ---
def circle_packing32() -> np.ndarray:
    """
    Finds an optimal arrangement of 32 circles using a hybrid Genetic Algorithm
    and local SLSQP search, inspired by high-performing solutions.
    """
    start_time = time.time()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Phase 1: Global Search with Genetic Algorithm ---
    # Parameters adjusted for better exploration/exploitation balance within time limits
    POP_SIZE = 600 # Slightly reduced to allow more generations
    NGEN = 900 # Increased generations for adaptive mutation
    CXPB, MUTPB = 0.8, 0.2
    NUM_CANDIDATES = 10
    
    # Adaptive mutation parameters (SIGMA_END adjusted)
    SIGMA_START = 0.05
    SIGMA_END = 0.01 # Slightly increased for more exploration in later GA stages

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(NUM_CANDIDATES)
    
    print("Starting genetic algorithm for global search...")
    
    # Setup multiprocessing pool for parallel evaluation
    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_cores)
    toolbox.register("map", pool.map)

    # Initial evaluation of the population
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    hof.update(pop)
    
    # Manual evolutionary loop for adaptive mutation
    for gen in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values
        
        current_sigma = SIGMA_START - (SIGMA_START - SIGMA_END) * (gen / NGEN)
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant, sigma=current_sigma)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        if (gen + 1) % 100 == 0:
            print(f"  Gen {gen+1}/{NGEN}, sigma={current_sigma:.4f}, best fitness={hof[0].fitness.values[0]:.4f}")

    pool.close()
    pool.join()

    print(f"GA phase finished in {time.time() - start_time:.2f}s. Refining {len(hof)} candidates.")

    # --- Phase 2: Local Refinement with SciPy SLSQP ---
    bounds_lower = np.tile([0.0, 0.0, MIN_RADIUS_CONSTRAINT], N_CIRCLES)
    bounds_upper = np.tile([1.0, 1.0, 0.5], N_CIRCLES)
    scipy_bounds = Bounds(bounds_lower, bounds_upper)

    containment_nlc = NonlinearConstraint(containment_constraint_scipy, 0, np.inf)
    non_overlap_nlc = NonlinearConstraint(non_overlap_constraint_scipy, 0, np.inf)

    best_solution, best_fun_val = None, np.inf

    for i, candidate in enumerate(hof):
        print(f"Refining candidate {i + 1}/{len(hof)}...")
        res = minimize(objective_scipy, np.array(candidate),
                       method='SLSQP',
                       bounds=scipy_bounds,
                       constraints=[containment_nlc, non_overlap_nlc],
                       options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False})

        if res.success and res.fun < best_fun_val:
            best_fun_val = res.fun
            best_solution = res.x
            print(f"  New best solution found: sum_radii = {-best_fun_val:.8f}")
        elif not res.success:
            print(f"  Candidate {i + 1} did not converge: {res.message}")

    # --- Phase 3: Final Polishing Step ---
    if best_solution is not None:
        print("Polishing the best solution found...")
        polish_res = minimize(objective_scipy, best_solution,
                              method='SLSQP',
                              bounds=scipy_bounds,
                              constraints=[containment_nlc, non_overlap_nlc],
                              options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}) # Increased maxiter for polishing
        if polish_res.success and polish_res.fun < best_fun_val:
            best_solution = polish_res.x
            best_fun_val = polish_res.fun
            print(f"  Polishing improved result: sum_radii = {-best_fun_val:.12f}")
        else:
            print("  Polishing did not yield further improvement.")


    if best_solution is None:
        print("Warning: No local optimization succeeded. Using best GA result.")
        final_params = np.array(hof[0])
    else:
        final_params = best_solution
    
    final_params[2::3] = np.maximum(final_params[2::3], MIN_RADIUS_CONSTRAINT)
    final_circles = final_params.reshape(N_CIRCLES, 3)
    
    return final_circles


# EVOLVE-BLOCK-END
