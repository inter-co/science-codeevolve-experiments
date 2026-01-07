# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from deap import base, creator, tools, algorithms
import random
import multiprocessing
from scipy.optimize import minimize
# Removed scipy.signal as we'll use numpy.fft directly for convolution as in Inspiration 1

# Global random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- GA Configuration Constants ---
POP_SIZE = 500
NGEN = 2500 # Increased generations from 1500 to 2500, matching Inspiration 1's successful configuration.
CXPB, MUTPB = 0.8, 1.0 # Crossover and Mutation probabilities (MUTPB is overall chance for any mutation type)

MIN_EVO_SEQ_LEN, MAX_EVO_SEQ_LEN = 20, 1000 # Max length aligned with problem spec
MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN = 20, 200 # Initial population typically smaller to explore
HEIGHT_MIN, HEIGHT_MAX = 0.0, 1000.0

NUM_SEED_INDIVIDUALS = int(POP_SIZE * 0.2) # 20% of population is seeded

# Adaptive mutation parameters
INITIAL_SIGMA = 0.1 * HEIGHT_MAX # Initial mutation range as proportion of total height range
FINAL_SIGMA = 0.005 * HEIGHT_MAX # Final mutation range

# Granular mutation probabilities for finer control
IND_VAL_MUT_PB = 0.2 # Independent probability for each attribute's value to be mutated
ADD_GENE_PB = 0.1 # Probability of attempting to add a gene
DEL_GENE_PB = 0.1 # Probability of attempting to delete a gene

# Helper function for calculating 1/C1 (fitness function for GA)
def calculate_inv_c1(sequence: list[float]) -> tuple[float]:
    if not sequence: return 0.0,
    
    a = np.array(sequence, dtype=np.float64)
    n = len(a)
    a = np.clip(a, HEIGHT_MIN, HEIGHT_MAX)
    sum_a = np.sum(a)

    if n == 0 or sum_a < 0.01:
        return 0.0,

    # Compute convolution b = a * a using direct FFT for efficiency (O(N log N)), as in Inspiration 1.
    conv_len = 2 * n - 1
    if conv_len <= 0: return 0.0, # Handle cases where convolution length is invalid
    
    # Determine FFT size (next power of 2 for efficiency)
    fft_size = int(2**np.ceil(np.log2(conv_len)))
    
    A_fft = np.fft.fft(a, fft_size)
    B_fft = A_fft * A_fft
    b_full = np.fft.ifft(B_fft).real
    b = b_full[:conv_len] # Take only the valid part of the convolution
    
    max_b = np.max(b) if b.size > 0 else 0
    if max_b < 1e-9: # Check for near-zero max_b to avoid division by tiny numbers
        return 0.0,

    denominator = (2 * n * max_b)
    # The check for denominator == 0 is now redundant if max_b < 1e-9 is handled, as 2*n is always positive.
    # Removed: if denominator == 0: return 0.0,

    inv_c1 = (sum_a * sum_a) / denominator
    return inv_c1,

# Objective function for local search (returns negative inv_c1 for minimization)
def local_search_objective(x: np.ndarray) -> float:
    """Objective function for scipy.minimize, returns negative inv_c1 for maximization."""
    # scipy.minimize expects a flat array, so convert to list for calculate_inv_c1
    return -calculate_inv_c1(x.tolist())[0]

# --- DEAP Setup ---
# Create Fitness and Individual classes if they don't already exist
# This is a common pattern when using DEAP in scripts that might be re-run or imported
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, HEIGHT_MIN, HEIGHT_MAX)
toolbox.register("evaluate", calculate_inv_c1)
toolbox.register("select", tools.selTournament, tournsize=5)

# --- Constructive Population Initialization ---
def _initialize_population_seeded(pop_size, num_seed_individuals):
    """
    Initializes the population with a mix of random, sparse, palindromic, periodic,
    and structured sparse sequences (added from Inspiration 1).
    """
    population = []
    
    # Proportions for seeded types (adjusted from Inspiration 1)
    num_palindromic = num_seed_individuals // 4
    num_sparse = num_seed_individuals // 4
    num_periodic = num_seed_individuals // 4
    num_structured_sparse = num_seed_individuals - num_palindromic - num_sparse - num_periodic # Remaining for new type

    # Palindromic seeds (symmetric structures)
    for _ in range(num_palindromic):
        n = random.randint(MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN)
        half_len = (n + 1) // 2
        half_seq = [toolbox.attr_float() for _ in range(half_len)]
        full_seq = half_seq + (half_seq[:-1][::-1] if n % 2 == 1 else half_seq[::-1])
        population.append(creator.Individual(full_seq))

    # Sparse (Sidon-like) seeds (few non-zero elements at random positions)
    for _ in range(num_sparse):
        n = random.randint(MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN)
        seq = [HEIGHT_MIN] * n
        k = max(1, int(np.sqrt(n))) # Number of non-zero elements, heuristic for Sidon
        indices = random.sample(range(n), min(k, n))
        for idx in indices: seq[idx] = random.uniform(0.5, 5.0) # Small positive values
        
        # Ensure sum > 0.01 (robustness check)
        if np.sum(seq) < 0.01 and n > 0:
            seq[random.randint(0, n-1)] = random.uniform(0.1, 1.0)
        elif n == 0:
            seq = [random.uniform(0.1, 1.0)] # If n=0, make it a valid sequence
        population.append(creator.Individual(seq))

    # Periodic/Alternating Sequences (from Insp 2)
    for _ in range(num_periodic):
        n = random.randint(MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN)
        sequence_list = []
        pattern_type = random.choice([0, 1, 2]) # 0: 1,0,1,0...; 1: 1,1,0,0,1,1,0,0...; 2: 1,0,0,1,0,0...
        
        if pattern_type == 0: # 1,0,1,0...
            for i in range(n):
                sequence_list.append(random.uniform(0.5, 1.5) if i % 2 == 0 else HEIGHT_MIN)
        elif pattern_type == 1: # 1,1,0,0...
            for i in range(n):
                sequence_list.append(random.uniform(0.5, 1.5) if (i % 4 == 0 or i % 4 == 1) else HEIGHT_MIN)
        else: # 1,0,0,1,0,0... (sparse, but periodic)
            for i in range(n):
                sequence_list.append(random.uniform(0.5, 1.5) if i % 3 == 0 else HEIGHT_MIN)
        
        # Ensure sum > 0.01 for valid fitness evaluation
        if np.sum(sequence_list) < 0.01 and n > 0:
            sequence_list[random.randint(0, n-1)] = random.uniform(0.1, 1.0)
        elif n == 0:
            sequence_list = [random.uniform(0.1, 1.0)]
        population.append(creator.Individual(sequence_list))

    # NEW: Structured Sparse Seeds (equally spaced non-zero elements, from Inspiration 1)
    for _ in range(num_structured_sparse):
        n = random.randint(MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN)
        seq = [HEIGHT_MIN] * n
        k = max(1, int(np.sqrt(n))) # Number of non-zero elements, heuristic for Sidon density
        
        if n > 0 and k > 0:
            # Place k non-zero elements as evenly spaced as possible
            spacing = n / k
            for i in range(k):
                idx = int(round(i * spacing))
                if idx >= n: idx = n - 1 # Ensure index is within bounds
                seq[idx] = random.uniform(0.5, 5.0) # Small positive values

        # Ensure sum > 0.01 (robustness check)
        if np.sum(seq) < 0.01 and n > 0:
            seq[random.randint(0, n-1)] = random.uniform(0.1, 1.0)
        elif n == 0:
            seq = [random.uniform(0.1, 1.0)]
        population.append(creator.Individual(seq))

    # Fill the rest with truly random individuals
    for _ in range(pop_size - len(population)):
        length = random.randint(MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN)
        population.append(creator.Individual(toolbox.attr_float() for _ in range(length)))
    return population

# --- Hybrid Crossover Operator (Inspired by Insp. 2 & 3) ---
def hybrid_crossover(ind1, ind2):
    """Randomly chooses between one-point and blend crossover."""
    if random.random() < 0.5:
        # One-point crossover for structural diversity
        # This will swap tails, potentially changing lengths if parents have different lengths
        size1, size2 = len(ind1), len(ind2)
        if min(size1, size2) < 2: return ind1, ind2 # Not enough points for crossover
        
        cxpoint = random.randint(1, min(size1, size2) - 1)

        temp1 = ind1[cxpoint:]
        temp2 = ind2[cxpoint:]
        
        ind1[cxpoint:] = temp2
        ind2[cxpoint:] = temp1
        
    else:
        # Blend crossover for value fine-tuning (only on common part)
        # Ensure ind1 and ind2 are lists for cxBlend
        tools.cxBlend(ind1, ind2, alpha=0.5)
        # Ensure values stay within bounds after blending
        for i in range(len(ind1)): ind1[i] = max(HEIGHT_MIN, min(HEIGHT_MAX, ind1[i]))
        for i in range(len(ind2)): ind2[i] = max(HEIGHT_MIN, min(HEIGHT_MAX, ind2[i]))
    return ind1, ind2

toolbox.register("mate", hybrid_crossover)

# --- Custom adaptive mutation operator (inspired by Insp. 1, 2, 3) ---
def mutAdaptiveVariableGaussian(individual, generation_progress):
    """
    Applies adaptive Gaussian mutation and structural mutations.
    The strength of the Gaussian mutation decreases as the search progresses.
    """
    # Calculate adaptive mutation strength
    adaptive_sigma = INITIAL_SIGMA * (1.0 - generation_progress) + FINAL_SIGMA * generation_progress

    # 1. Value mutation with adaptive strength
    for i in range(len(individual)):
        if random.random() < IND_VAL_MUT_PB:
            individual[i] += random.gauss(0, adaptive_sigma)
            individual[i] = max(HEIGHT_MIN, min(HEIGHT_MAX, individual[i]))

    # 2. Structural mutations (add or remove genes)
    current_n = len(individual)
    
    # Try to add a gene
    if random.random() < ADD_GENE_PB and current_n < MAX_EVO_SEQ_LEN:
        insert_idx = random.randint(0, current_n)
        individual.insert(insert_idx, toolbox.attr_float())
        current_n += 1

    # Try to delete a gene
    if random.random() < DEL_GENE_PB and current_n > MIN_EVO_SEQ_LEN:
        if current_n > 0: # Ensure there's something to delete
            del_idx = random.randint(0, current_n - 1)
            del individual[del_idx]
            current_n -= 1

    # 3. Repair mechanism to enforce constraints (length)
    while len(individual) < MIN_EVO_SEQ_LEN:
        individual.append(toolbox.attr_float())
    while len(individual) > MAX_EVO_SEQ_LEN:
        del individual[random.randint(0, len(individual) - 1)]
    
    # Ensure all values are within bounds after mutation
    for i in range(len(individual)):
        individual[i] = max(HEIGHT_MIN, min(HEIGHT_MAX, individual[i]))

    return individual,

toolbox.register("mutate", mutAdaptiveVariableGaussian)


# --- Main Search Function (Hybrid GA + Local Search) ---
def search_for_best_sequence() -> list[float]:
    """
    Searches for the best coefficient sequence using a hybrid GA + local search approach, with
    constructive population seeding and parallel evaluation.
    """
    # Use multiprocessing pool for parallel evaluation
    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)
        
        # Initialize population with constructive seeding
        pop = _initialize_population_seeded(POP_SIZE, NUM_SEED_INDIVIDUALS)
        hof = tools.HallOfFame(1) # Stores the single best individual found

        # Evaluate the initial population
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop) # Update hall of fame with initial population

        # Custom evolutionary loop to pass generation progress to mutation
        for gen in range(NGEN):
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals to create new ones
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Apply crossover and mutation on the offspring
            for i in range(1, len(offspring), 2):
                if random.random() < CXPB:
                    offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                    # Invalidate the fitness of the modified individuals
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            
            # Apply mutation with adaptive strength
            generation_progress = gen / NGEN
            for ind in offspring:
                if random.random() < MUTPB: # Overall mutation probability
                    toolbox.mutate(ind, generation_progress=generation_progress)
                    # Invalidate the fitness of the modified individual
                    del ind.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the old population by the offspring
            pop[:] = offspring
            hof.update(pop) # Update hall of fame with current population

        # If no individual is found (e.g., all evaluations returned 0.0), return a default
        if not hof or not hof[0].fitness.values: 
            return [1.0] * MIN_EVO_SEQ_LEN
        
        best_from_ga = hof[0]
        
        # Perform local search on the best individual from GA
        # Bounds for local search are dynamic based on the length of the GA individual
        bounds = [(HEIGHT_MIN, HEIGHT_MAX)] * len(best_from_ga)
        res = minimize(
            local_search_objective,
            np.array(best_from_ga),
            method='L-BFGS-B', # A good choice for bounded, non-convex problems
            bounds=bounds,
            options={'maxiter': 250, 'disp': False} # Limited iterations for speed
        )
        
        polished_sequence = res.x
        final_fitness_polished, = calculate_inv_c1(polished_sequence.tolist())
        
        # Compare the fitness of the GA best vs. the locally polished version
        if final_fitness_polished > best_from_ga.fitness.values[0]:
            best_overall_sequence = polished_sequence
        else:
            best_overall_sequence = np.array(best_from_ga)

    # Final clipping and sum check for robustness
    final_sequence = np.clip(best_overall_sequence, HEIGHT_MIN, HEIGHT_MAX).tolist()
    if np.sum(final_sequence) < 0.01:
        # Fallback if the best sequence is numerically unstable or effectively zero
        return [1.0] * MIN_EVO_SEQ_LEN

    return final_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
