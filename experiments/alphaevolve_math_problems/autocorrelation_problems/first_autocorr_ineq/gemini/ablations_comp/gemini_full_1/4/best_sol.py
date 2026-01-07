# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from deap import base, creator, tools, algorithms
import random
import multiprocessing
import time
from scipy.optimize import minimize
from scipy import signal

# --- Global Configuration & Reproducibility (Inspired by Insp. 2 & 3) ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# EA parameters
MIN_INIT_SEQ_LEN = 20
MAX_INIT_SEQ_LEN = 400
MIN_EVO_SEQ_LEN = 20
MAX_EVO_SEQ_LEN = 2000
HEIGHT_CLIP_MIN = 0.0
HEIGHT_CLIP_MAX = 1000.0

POPULATION_SIZE = 400
GENERATIONS = 5000  # High NGEN, actual limit is time
CXPB = 0.85
ETA_CXSB = 30.0
GAUSS_MUT_INDPB = 0.2
PROB_LEN_MUT = 0.25
INITIAL_SIGMA = 50.0
FINAL_SIGMA = 0.5
STEP_LEN_CHANGE = 20
TOURNAMENT_SIZE = 5
NUM_SEED_INDIVIDUALS_RATIO = 0.2

# Time budget management
TIME_LIMIT_GA_SECONDS = 150
TIME_LIMIT_TOTAL_SECONDS = 175

# --- DEAP Framework Setup ---
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

# --- Objective & Fitness Functions (Inspired by Insp. 2 & 3) ---
def calculate_inv_c1(sequence: list[float]) -> tuple[float,]:
    """Calculates 1/C1 using scipy.signal.fftconvolve for robustness."""
    a_np = np.array(sequence, dtype=np.float64)
    n = len(a_np)
    
    a_np = np.clip(a_np, HEIGHT_CLIP_MIN, HEIGHT_CLIP_MAX)
    sum_a = np.sum(a_np)

    if n < MIN_EVO_SEQ_LEN or sum_a < 0.01:
        return (0.0,)

    b = signal.fftconvolve(a_np, a_np, mode='full')
    max_b = np.max(b) if b.size > 0 else 0

    if max_b <= 1e-12:
        return (0.0,)

    denominator = 2 * n * max_b
    if denominator <= 1e-12:
        return (0.0,)
    
    inv_c1 = (sum_a * sum_a) / denominator
    return (inv_c1,)

def local_search_objective(x: np.ndarray) -> float:
    """Objective for scipy.minimize, returns negative inv_c1 for maximization."""
    return -calculate_inv_c1(x.tolist())[0]

# --- DEAP Toolbox Setup (Inspired by Insp. 2 & 3) ---
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, HEIGHT_CLIP_MIN, HEIGHT_CLIP_MAX)
toolbox.register("evaluate", calculate_inv_c1)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

def var_len_cxSimulatedBinaryBounded(ind1, ind2, eta, low, up):
    """Robust crossover for variable-length individuals."""
    min_len = min(len(ind1), len(ind2))
    if min_len < 2: return ind1, ind2
    
    temp_ind1, temp_ind2 = creator.Individual(ind1[:min_len]), creator.Individual(ind2[:min_len])
    tools.cxSimulatedBinaryBounded(temp_ind1, temp_ind2, eta=eta, low=low, up=up)
    
    ind1[:min_len], ind2[:min_len] = temp_ind1, temp_ind2
    return ind1, ind2

toolbox.register("mate", var_len_cxSimulatedBinaryBounded, eta=ETA_CXSB, low=HEIGHT_CLIP_MIN, up=HEIGHT_CLIP_MAX)

def combined_mutation(individual, mu, sigma, indpb_val, prob_len_mut, min_len, max_len, step_len_change):
    """Combined value and length mutation."""
    tools.mutGaussian(individual, mu=mu, sigma=sigma, indpb=indpb_val)
    for i in range(len(individual)):
        individual[i] = max(HEIGHT_CLIP_MIN, min(HEIGHT_CLIP_MAX, individual[i]))

    if random.random() < prob_len_mut:
        current_len = len(individual)
        if random.random() < 0.5: # Grow
            if current_len < max_len:
                num_to_add = random.randint(1, min(step_len_change, max_len - current_len))
                for _ in range(num_to_add): individual.insert(random.randint(0, len(individual)), toolbox.attr_float())
        else: # Shrink
            if current_len > min_len:
                num_to_remove = random.randint(1, min(step_len_change, current_len - min_len))
                for _ in range(num_to_remove):
                    if len(individual) > min_len: del individual[random.randint(0, len(individual) - 1)]
    return individual,

toolbox.register("mutate", combined_mutation, mu=0.0, indpb_val=GAUSS_MUT_INDPB,
                 prob_len_mut=PROB_LEN_MUT, min_len=MIN_EVO_SEQ_LEN, max_len=MAX_EVO_SEQ_LEN,
                 step_len_change=STEP_LEN_CHANGE)

# --- Constructive Population Initialization (Inspired by Insp. 2 & 3) ---
def generate_random_individual():
    length = random.randint(MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN)
    ind = creator.Individual([toolbox.attr_float() for _ in range(length)])
    if sum(ind) < 0.01: ind[random.randint(0, len(ind)-1)] = random.uniform(0.1, 10.0)
    return ind

def _initialize_population_seeded(pop_size):
    population = []
    num_seed = int(pop_size * NUM_SEED_INDIVIDUALS_RATIO)
    num_sparse, num_palindromic = num_seed // 2, num_seed - (num_seed // 2)

    for _ in range(num_sparse):
        n = random.randint(MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN)
        seq = [0.0] * n; k = max(1, int(np.sqrt(n))); indices = random.sample(range(n), min(k, n))
        for idx in indices: seq[idx] = random.uniform(0.5, 5.0)
        population.append(creator.Individual(seq))

    for _ in range(num_palindromic):
        length = random.randint(MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN); half_len = (length + 1) // 2
        half_seq = [toolbox.attr_float() for _ in range(half_len)]
        full_seq = half_seq + (half_seq[:-1][::-1] if length % 2 == 1 else half_seq[::-1])
        population.append(creator.Individual(full_seq))

    for _ in range(pop_size - len(population)):
        population.append(generate_random_individual())
    return population

toolbox.register("population", _initialize_population_seeded, pop_size=POPULATION_SIZE)

# --- Main Hybrid Evolutionary Algorithm (Inspired by Insp. 2 & 3) ---
def search_for_best_sequence() -> list[float]:
    """Hybrid GA + Local Search with constructive initialization and adaptive mutation."""
    start_time = time.time()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    best_from_ga = None

    try:
        pop = toolbox.population()
        hof = tools.HallOfFame(1)
        
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
        hof.update(pop)
        
        print("Starting Hybrid GA with constructive population...")
        for gen in range(GENERATIONS):
            if time.time() - start_time > TIME_LIMIT_GA_SECONDS:
                print(f"\nGA time limit ({TIME_LIMIT_GA_SECONDS}s) reached at gen {gen}.")
                break

            offspring = toolbox.select(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2); del child1.fitness.values, child2.fitness.values
            
            gen_ratio = min(1.0, gen / GENERATIONS if GENERATIONS > 1 else 1.0)
            current_sigma = FINAL_SIGMA + (INITIAL_SIGMA - FINAL_SIGMA) * (1 - gen_ratio)**2.0
            
            for mutant in offspring:
                toolbox.mutate(mutant, sigma=current_sigma); del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses): ind.fitness.values = fit
            
            pop[:] = offspring; hof.update(pop)

        best_from_ga = hof[0] if hof else tools.selBest(pop, 1)[0]
    finally:
        pool.close(); pool.join()

    if not best_from_ga: return [1.0] * MIN_EVO_SEQ_LEN
    best_from_ga_fitness, = calculate_inv_c1(best_from_ga)
    final_sequence_np = np.array(best_from_ga)
    
    remaining_time = TIME_LIMIT_TOTAL_SECONDS - (time.time() - start_time)
    if remaining_time > 5:
        print(f"GA finished. Best 1/C1: {best_from_ga_fitness:.6f}. Polishing...")
        bounds = [(HEIGHT_CLIP_MIN, HEIGHT_CLIP_MAX)] * len(best_from_ga)
        res = minimize(local_search_objective, np.array(best_from_ga), method='L-BFGS-B', bounds=bounds, options={'maxiter': 250})
        
        if -res.fun > best_from_ga_fitness:
            print(f"Polishing improved 1/C1 to: {-res.fun:.6f}")
            final_sequence_np = res.x
    else:
        print(f"GA finished with best 1/C1: {best_from_ga_fitness:.6f}. Skipping polishing.")

    final_sequence = np.clip(final_sequence_np, HEIGHT_CLIP_MIN, HEIGHT_CLIP_MAX).tolist()
    if np.sum(final_sequence) < 0.01: return [1.0] * MIN_EVO_SEQ_LEN
    
    return final_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
