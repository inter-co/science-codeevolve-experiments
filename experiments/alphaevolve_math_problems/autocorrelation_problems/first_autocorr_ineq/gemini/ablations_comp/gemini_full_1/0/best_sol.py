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

# --- Global Configuration & Reproducibility (Inspired by all inspirations) ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# EA parameters - Tuned based on best-performing inspirations
GLOBAL_CONFIG = {
    "HEIGHT_CLIP_MIN": 0.0,
    "HEIGHT_CLIP_MAX": 1000.0,
    "MIN_INIT_SEQ_LEN": 20,
    "MAX_INIT_SEQ_LEN": 400,
    "MIN_EVO_SEQ_LEN": 20,
    "MAX_EVO_SEQ_LEN": 2000, # Increased max length for more exploration
    "POPULATION_SIZE": 500,
    "CXPB": 0.9, # High crossover probability
    "ALPHA_CXBLEND": 0.5, # Parameter for cxBlend
    "TOURNAMENT_SIZE": 7, # Increased selection pressure
    "GAUSS_MUT_INDPB": 0.2, # Independent probability for each attribute to be mutated
    "PROB_LEN_MUT": 0.3, # Probability of length mutation occurring
    "STEP_LEN_CHANGE": 25, # Max number of elements to add/remove
    "INITIAL_SIGMA": 50.0, # High for exploration
    "FINAL_SIGMA": 0.5, # Low for fine-tuning
    "NUM_SEED_INDIVIDUALS_RATIO": 0.2, # Proportion of population to seed
    "LOCAL_SEARCH_MAXITER": 300,
    "SUM_MIN_THRESHOLD": 0.01, # Minimum sum of 'a' to avoid division by zero
    "MAX_B_MIN_THRESHOLD": 1e-12, # Minimum max_b to avoid division by zero
}

# Helper function for calculating 1/C1 (fitness function for GA)
# Using scipy.signal.fftconvolve for robustness, as seen in inspirations.
def calculate_inv_c1(sequence: list[float]) -> tuple[float]:
    if not sequence: return 0.0,
    
    a = np.array(sequence, dtype=np.float64)
    n = len(a)
    a = np.clip(a, GLOBAL_CONFIG["HEIGHT_CLIP_MIN"], GLOBAL_CONFIG["HEIGHT_CLIP_MAX"])
    sum_a = np.sum(a)

    # Added n < MIN_EVO_SEQ_LEN check (inspired by inspirations)
    if n < GLOBAL_CONFIG["MIN_EVO_SEQ_LEN"] or sum_a < GLOBAL_CONFIG["SUM_MIN_THRESHOLD"]:
        return 0.0,

    b = signal.fftconvolve(a, a, mode='full')
    
    max_b = np.max(b) if b.size > 0 else 0
    if max_b < GLOBAL_CONFIG["MAX_B_MIN_THRESHOLD"]:
        return 0.0,

    denominator = (2 * n * max_b)
    if denominator < GLOBAL_CONFIG["MAX_B_MIN_THRESHOLD"]: # Reusing threshold for consistency
        return 0.0,

    inv_c1 = (sum_a ** 2) / denominator
    return inv_c1,

# Objective function for local search (returns negative inv_c1 for minimization)
def local_search_objective(x: np.ndarray) -> float:
    """Objective function for scipy.minimize, returns negative inv_c1 for maximization."""
    return -calculate_inv_c1(x.tolist())[0]

# --- DEAP Setup ---
# Create Fitness and Individual classes if they don't already exist
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, GLOBAL_CONFIG["HEIGHT_CLIP_MIN"], GLOBAL_CONFIG["HEIGHT_CLIP_MAX"])
toolbox.register("evaluate", calculate_inv_c1)

# --- Custom Genetic Operators inspired by high-performing inspirations ---

# From Inspiration 1: A robust crossover operator for variable-length individuals.
# It blends the common part and swaps the tails, which is more effective than
# the default operators that assume fixed-length sequences. This fixes a key
# flaw in using standard operators with variable-length individuals in eaSimple.
def var_len_cxBlend_and_swapTails(ind1, ind2, alpha, min_evo_len, max_evo_len, height_clip_min, height_clip_max):
    child1, child2 = list(ind1), list(ind2)
    min_l = min(len(child1), len(child2))

    # Blend crossover on the common part
    for i in range(min_l):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        x1, x2 = child1[i], child2[i]
        child1[i] = (1. - gamma) * x1 + gamma * x2
        child2[i] = gamma * x1 + (1. - gamma) * x2
        # Immediately clip values to maintain constraints
        child1[i] = max(height_clip_min, min(height_clip_max, child1[i]))
        child2[i] = max(height_clip_min, min(height_clip_max, child2[i]))

    # Swap tails, a key part of this effective variable-length operator
    child1[min_l:], child2[min_l:] = child2[min_l:], child1[min_l:]

    # Enforce length constraints and assign back to DEAP individuals
    for i, child_list in enumerate([child1, child2]):
        if len(child_list) > max_evo_len:
            child_list = child_list[:max_evo_len]
        elif len(child_list) < min_evo_len:
            padding_needed = min_evo_len - len(child_list)
            padding = [toolbox.attr_float() for _ in range(padding_needed)]
            child_list.extend(padding)
        
        target_ind = ind1 if i == 0 else ind2
        target_ind[:] = child_list # Modify the individual in-place
    
    return ind1, ind2

# Registering the new, correct crossover and more aggressive selection
toolbox.register("mate", var_len_cxBlend_and_swapTails, 
                 alpha=GLOBAL_CONFIG["ALPHA_CXBLEND"],
                 min_evo_len=GLOBAL_CONFIG["MIN_EVO_SEQ_LEN"],
                 max_evo_len=GLOBAL_CONFIG["MAX_EVO_SEQ_LEN"],
                 height_clip_min=GLOBAL_CONFIG["HEIGHT_CLIP_MIN"],
                 height_clip_max=GLOBAL_CONFIG["HEIGHT_CLIP_MAX"])
toolbox.register("select", tools.selTournament, tournsize=GLOBAL_CONFIG["TOURNAMENT_SIZE"])

# Refined combined mutation operator (from inspirations)
def combined_mutation(individual, mu, sigma, indpb_val, prob_len_mut, step_len_change, min_evo_len, max_evo_len, height_clip_min, height_clip_max):
    # Mutate values using Gaussian mutation
    tools.mutGaussian(individual, mu=mu, sigma=sigma, indpb=indpb_val)
    
    # Mutate length with a certain probability
    if random.random() < prob_len_mut:
        current_len = len(individual)
        if random.random() < 0.5 and current_len < max_evo_len: # Grow sequence
            num_to_add = random.randint(1, min(step_len_change, max_evo_len - current_len))
            for _ in range(num_to_add):
                individual.insert(random.randint(0, len(individual)), toolbox.attr_float())
        elif current_len > min_evo_len: # Shrink sequence
            num_to_remove = random.randint(1, min(step_len_change, current_len - min_evo_len))
            for _ in range(num_to_remove):
                if len(individual) > min_evo_len: # Ensure we don't go below min length
                    del individual[random.randint(0, len(individual) - 1)]
    
    # Ensure all values are within bounds after mutation
    for i in range(len(individual)):
        individual[i] = max(height_clip_min, min(height_clip_max, individual[i]))
    return individual,

# Register mutation without sigma, so it can be passed dynamically.
# This enables the adaptive mutation strategy from inspirations.
toolbox.register("mutate", combined_mutation, 
                 mu=0.0, 
                 indpb_val=GLOBAL_CONFIG["GAUSS_MUT_INDPB"],
                 prob_len_mut=GLOBAL_CONFIG["PROB_LEN_MUT"],
                 step_len_change=GLOBAL_CONFIG["STEP_LEN_CHANGE"],
                 min_evo_len=GLOBAL_CONFIG["MIN_EVO_SEQ_LEN"],
                 max_evo_len=GLOBAL_CONFIG["MAX_EVO_SEQ_LEN"],
                 height_clip_min=GLOBAL_CONFIG["HEIGHT_CLIP_MIN"],
                 height_clip_max=GLOBAL_CONFIG["HEIGHT_CLIP_MAX"])

# --- Constructive Population Initialization (from inspirations) ---
def generate_random_individual():
    """Generates a random individual with length within the initial bounds, ensuring sum > 0.01."""
    length = random.randint(GLOBAL_CONFIG["MIN_INIT_SEQ_LEN"], GLOBAL_CONFIG["MAX_INIT_SEQ_LEN"])
    ind = creator.Individual(toolbox.attr_float() for _ in range(length))
    # Ensure generated individual is not all zeros (inspired by inspirations)
    if np.sum(ind) < GLOBAL_CONFIG["SUM_MIN_THRESHOLD"]:
        # Give it a small positive value if it's too close to zero
        if len(ind) > 0:
            ind[random.randint(0, len(ind) - 1)] = random.uniform(0.1, 10.0)
        else: # If length is 0, make it a minimal valid sequence
            ind.append(random.uniform(0.1, 10.0))
    return ind

def _initialize_population_seeded(pop_size, num_seed_individuals):
    """
    Initializes the population with a mix of random, sparse, and palindromic sequences.
    Sparse sequences have few non-zero elements. Palindromic sequences are symmetric.
    """
    population = []
    num_sparse = num_seed_individuals // 2
    num_palindromic = num_seed_individuals - num_sparse

    for _ in range(num_sparse):
        n = random.randint(GLOBAL_CONFIG["MIN_INIT_SEQ_LEN"], GLOBAL_CONFIG["MAX_INIT_SEQ_LEN"])
        seq = [0.0] * n
        k = max(1, int(np.sqrt(n))) # Number of non-zero elements
        indices = random.sample(range(n), min(k, n))
        for idx in indices: seq[idx] = random.uniform(0.5, 5.0) # Small positive values
        population.append(creator.Individual(seq))

    for _ in range(num_palindromic):
        length = random.randint(GLOBAL_CONFIG["MIN_INIT_SEQ_LEN"], GLOBAL_CONFIG["MAX_INIT_SEQ_LEN"])
        half_len = (length + 1) // 2
        half_seq = [toolbox.attr_float() for _ in range(half_len)]
        full_seq = half_seq + (half_seq[:-1][::-1] if length % 2 == 1 else half_seq[::-1])
        population.append(creator.Individual(full_seq))

    for _ in range(pop_size - len(population)): # Fill the rest with truly random individuals
        population.append(generate_random_individual())
    return population

# --- Main Search Function (Hybrid GA + Local Search with Time Limit and Adaptive Mutation) ---
def search_for_best_sequence() -> list[float]:
    """
    Hybrid GA + Local Search, inspired by Inspirations 1, 2 & 3.
    Features: Constructive initialization, adaptive mutation, robust variable-length operators,
    a time-limited manual evolution loop, and a final polishing step with L-BFGS-B.
    """
    # Time budget management (from all inspirations)
    TIME_LIMIT_TOTAL_SECONDS = 178.0
    TIME_LIMIT_GA_SECONDS = 165.0 # Leave time for final polishing
    
    start_time_total = time.time()
    best_overall_sequence = np.array([1.0] * GLOBAL_CONFIG["MIN_EVO_SEQ_LEN"]) # Default fallback

    # Use multiprocessing pool for parallel evaluation
    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)
        
        # 1. Initialize population with constructive seeding
        num_seed_individuals = int(GLOBAL_CONFIG["POPULATION_SIZE"] * GLOBAL_CONFIG["NUM_SEED_INDIVIDUALS_RATIO"])
        pop = _initialize_population_seeded(GLOBAL_CONFIG["POPULATION_SIZE"], num_seed_individuals)
        hof = tools.HallOfFame(1)

        # 2. Evaluate initial population
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)
        
        print(f"Starting GA with population size {GLOBAL_CONFIG['POPULATION_SIZE']}. Initial best 1/C1: {hof[0].fitness.values[0]:.6f}")

        # 3. Manual, time-limited GA loop with adaptive mutation (inspired by all inspirations)
        start_time_ga = time.time()
        gen = 0
        while time.time() - start_time_ga < TIME_LIMIT_GA_SECONDS:
            gen += 1
            # Selection
            offspring = toolbox.select(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < GLOBAL_CONFIG["CXPB"]:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

            # Adaptive Mutation (from Insp. 2 & 3)
            time_ratio = min(1.0, (time.time() - start_time_ga) / TIME_LIMIT_GA_SECONDS)
            current_sigma = GLOBAL_CONFIG["FINAL_SIGMA"] + (GLOBAL_CONFIG["INITIAL_SIGMA"] - GLOBAL_CONFIG["FINAL_SIGMA"]) * (1 - time_ratio)**2
            
            for mutant in offspring:
                # MUTPB is 1.0, so every offspring mutates
                toolbox.mutate(mutant, sigma=current_sigma) # Pass sigma dynamically
                del mutant.fitness.values

            # Evaluate individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)
            # Print progress (inspired by inspirations)
            if gen % 100 == 0:
                print(f"  Gen {gen}, Time elapsed: {time.time() - start_time_ga:.1f}s, Best 1/C1: {hof[0].fitness.values[0]:.6f}")
        
        # Select the best individual from the GA phase (cleaner selection from inspirations)
        best_from_ga = hof[0] if hof and hof[0].fitness.valid else tools.selBest(pop, 1)[0]
        
        if not best_from_ga: return [1.0] * GLOBAL_CONFIG["MIN_EVO_SEQ_LEN"] # Fallback
        
        best_overall_sequence = np.array(best_from_ga)
        best_from_ga_fitness = best_from_ga.fitness.values[0]
        print(f"\nGA finished. Best 1/C1 found by GA: {best_from_ga_fitness:.6f}")

        # 4. Local Search Polishing Step (with time check from Insp. 2 & 3)
        remaining_time = TIME_LIMIT_TOTAL_SECONDS - (time.time() - start_time_total)
        if remaining_time > 5: # Ensure enough time for local search
            print(f"Remaining time {remaining_time:.1f}s. Starting local search polishing...")
            bounds = [(GLOBAL_CONFIG["HEIGHT_CLIP_MIN"], GLOBAL_CONFIG["HEIGHT_CLIP_MAX"])] * len(best_from_ga)
            res = minimize(
                local_search_objective,
                np.array(best_from_ga),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': GLOBAL_CONFIG["LOCAL_SEARCH_MAXITER"], 'disp': False}
            )
            
            if -res.fun > best_from_ga_fitness:
                print(f"Polishing improved 1/C1 to: {-res.fun:.6f}")
                best_overall_sequence = res.x
            else:
                print(f"Local search did not improve 1/C1. Best 1/C1 remains: {best_from_ga_fitness:.6f}")
        else:
            print(f"Not enough time ({remaining_time:.1f}s) for local search. Skipping polishing.")

    # Final clipping and sum check for robustness
    final_sequence = np.clip(best_overall_sequence, GLOBAL_CONFIG["HEIGHT_CLIP_MIN"], GLOBAL_CONFIG["HEIGHT_CLIP_MAX"]).tolist()
    if not final_sequence or np.sum(final_sequence) < GLOBAL_CONFIG["SUM_MIN_THRESHOLD"]:
        print("Warning: Final sequence is invalid or sum is too low, returning default.")
        return [1.0] * GLOBAL_CONFIG["MIN_EVO_SEQ_LEN"]

    return final_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
