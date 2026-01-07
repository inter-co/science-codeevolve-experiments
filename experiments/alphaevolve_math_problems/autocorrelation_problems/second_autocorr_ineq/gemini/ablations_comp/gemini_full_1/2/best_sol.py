# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
import time # Needed for time-limited loop
from scipy.signal import fftconvolve # Use fast convolution
import numba # Retain Numba for loop acceleration
from deap import base, creator, tools, algorithms # For GA
from multiprocessing import Pool # For parallel evaluation
from functools import partial # For passing fixed arguments to mapped functions

# --- Global Parameters for Parametric Representation & Optimization ---
# Inspired by Insp 2's parametric approach and high resolution.
# Reduced SEQUENCE_LENGTH for faster evaluations (Inspired by Insp 3)
SEQUENCE_LENGTH = 5000
# Increased N_BASIS_FUNCTIONS for more complex shapes (Inspired by Insp 3)
N_BASIS_FUNCTIONS = 12

# Step width for the high-resolution function f.
STEP_WIDTH_F = 0.5 / SEQUENCE_LENGTH

# --- C2 Calculation (Optimized with FFT and Numba) ---

@numba.jit(nopython=True, fastmath=True)
def _compute_norms_from_g(g_values: np.ndarray, dt_g: float) -> float:
    """
    Numba-accelerated helper to compute norms from the autoconvolution array g.
    This part is loop-heavy and benefits greatly from JIT compilation.
    """
    g_len = len(g_values)
    if g_len == 0:
        return 0.0
    
    # 1. Compute ||g||₂² (L2-norm squared) via piecewise linear integration.
    g_l2_sq = 0.0
    if g_len > 1:
        # Vectorized sum is faster inside Numba than a Python loop
        y1 = g_values[:-1]
        y2 = g_values[1:]
        g_l2_sq = np.sum(y1**2 + y1*y2 + y2**2)
        g_l2_sq *= (dt_g / 3.0)

    # 2. Compute ||g||₁ (L1-norm) approximation.
    g_l1 = np.sum(np.abs(g_values)) / (g_len + 1)
    
    # 3. Compute ||g||∞ (Infinity-norm).
    g_linf = np.max(np.abs(g_values))

    # Handle cases where denominators might be zero or extremely small.
    if g_l1 < 1e-12 or g_linf < 1e-12:
        return 0.0
    
    c2 = g_l2_sq / (g_l1 * g_linf)
    return c2 if np.isfinite(c2) else 0.0

def compute_c2(f_values: np.ndarray) -> float:
    """
    Calculates C2 using fftconvolve and a Numba helper for norm calculations.
    Renamed from calculate_c2_value_fft for consistency with Inspirations.
    """
    if np.sum(f_values) < 1e-12:
        return 0.0

    # Autoconvolution g = f*f using FFT for performance on large arrays.
    g_values = fftconvolve(f_values, f_values, mode='full')
    
    # Call the Numba-JIT compiled function for the expensive norm calculations.
    return _compute_norms_from_g(g_values, STEP_WIDTH_F)

# --- Parametric Function Definition & DEAP-specific helpers ---

def decode_gaussian_params(params: list[float]) -> np.ndarray:
    """
    Decodes a flat array of parameters into a high-resolution step function `f`
    represented as a sum of Gaussian basis functions.
    """
    f_values = np.zeros(SEQUENCE_LENGTH, dtype=np.float64)
    x_grid = np.arange(SEQUENCE_LENGTH)
    
    for i in range(N_BASIS_FUNCTIONS):
        amp, mean_norm, std_norm = params[i*3 : (i+1)*3]
        
        # Ensure parameters are within valid ranges after mutation/crossover
        amp = max(0.0, amp)
        mean_norm = np.clip(mean_norm, 0.0, 1.0)
        std_norm = np.clip(std_norm, 0.005, 0.4) # Updated bound (Inspired by Insp 3)
        
        mean = mean_norm * (SEQUENCE_LENGTH - 1)
        std_dev = std_norm * SEQUENCE_LENGTH
        
        if std_dev < 1.0: std_dev = 1.0 # Prevent division by zero
        
        f_values += amp * np.exp(-0.5 * ((x_grid - mean) / std_dev)**2)
    
    f_values = np.maximum(0.0, f_values)
    max_val = np.max(f_values)
    if max_val > 1e-9:
        f_values /= max_val # Normalize max height to 1 for stability.
    
    return f_values

def evaluate_c2_deap_wrapper(individual_params: list[float]) -> tuple[float,]:
    """DEAP wrapper to decode parameters, compute C2, and return a tuple."""
    f_array = decode_gaussian_params(individual_params)
    c2_val = compute_c2(f_array)
    return (c2_val,)

def create_individual_gaussian_params(individual_creator):
    """
    Generates an individual as a list of parameters for Gaussian basis functions
    with diverse initializations (Inspired by Insp 1).
    """
    params = []
    
    initial_strategy = random.choice(['random_uniform', 'central_peak', 'sparse_wide', 'sparse_narrow', 'uniform_spread'])

    if initial_strategy == 'random_uniform':
        for _ in range(N_BASIS_FUNCTIONS):
            params.extend([
                random.uniform(0.0, 1.5),  # Amplitude
                random.uniform(0.0, 1.0),  # Mean (normalized)
                random.uniform(0.005, 0.4)   # Std Dev (normalized)
            ])
    elif initial_strategy == 'central_peak':
        num_central = random.randint(1, 2)
        for _ in range(num_central):
            params.extend([
                random.uniform(0.8, 1.5), # High amplitude
                random.uniform(0.4, 0.6), # Central mean
                random.uniform(0.1, 0.3)  # Moderate std dev
            ])
        for _ in range(N_BASIS_FUNCTIONS - num_central):
            params.extend([
                random.uniform(0.0, 0.5), # Lower amplitude
                random.uniform(0.0, 1.0),
                random.uniform(0.005, 0.4)
            ])
    elif initial_strategy == 'sparse_wide':
        num_active = random.randint(1, max(1, N_BASIS_FUNCTIONS // 2))
        for _ in range(num_active):
            params.extend([
                random.uniform(0.5, 1.5),
                random.uniform(0.0, 1.0),
                random.uniform(0.2, 0.4) # Wider std dev
            ])
        for _ in range(N_BASIS_FUNCTIONS - num_active):
            params.extend([0.0, random.uniform(0.0, 1.0), random.uniform(0.005, 0.4)]) # Inactive
    elif initial_strategy == 'sparse_narrow':
        num_active = random.randint(1, max(1, N_BASIS_FUNCTIONS // 2))
        for _ in range(num_active):
            params.extend([
                random.uniform(0.5, 1.5),
                random.uniform(0.0, 1.0),
                random.uniform(0.005, 0.05) # Narrow std dev
            ])
        for _ in range(N_BASIS_FUNCTIONS - num_active):
            params.extend([0.0, random.uniform(0.0, 1.0), random.uniform(0.005, 0.4)]) # Inactive
    elif initial_strategy == 'uniform_spread':
        for i in range(N_BASIS_FUNCTIONS):
            params.extend([
                random.uniform(0.5, 1.0),  # Moderate Amplitude
                i / (N_BASIS_FUNCTIONS - 1.0) if N_BASIS_FUNCTIONS > 1 else 0.5, # Evenly spaced means
                random.uniform(0.05, 0.2)   # Moderate Std Dev
            ])
            
    return individual_creator(params)

# --- Main Optimization Driver ---

def construct_function() -> list[float]:
    """
    Uses a parallelized Genetic Algorithm (DEAP) to optimize the parameters
    of a Gaussian-basis representation of `f` to maximize the C2 constant.
    Includes adaptive mutation and a final local search phase.
    """
    random.seed(42)
    np.random.seed(42)

    # --- GA Parameters ---
    POP_SIZE = 300 # Increased population size for better exploration
    CXPB = 0.8 # Crossover Probability
    MUTPB_BASE = 0.2 # Base probability of an individual undergoing mutation
    IND_MUTATION_RATE = 0.2 # Probability of a gene mutating within an individual (for Gaussian noise)
    TIME_LIMIT_SECONDS = 175 # Allocate time for main GA and local search
    NUM_ELITES = int(0.1 * POP_SIZE) # Elitism to preserve best solutions

    # --- DEAP Setup ---
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except RuntimeError:
        pass # Avoids error if run multiple times in same session

    toolbox = base.Toolbox()
    pool = Pool() # Initialize multiprocessing pool
    toolbox.register("map", pool.map) # Register the parallel map function

    # Register operators for the Gaussian basis function representation
    toolbox.register("individual", create_individual_gaussian_params, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_c2_deap_wrapper)
    toolbox.register("mate", tools.cxBlend, alpha=0.5) # Blend crossover for real-valued params
    
    # Custom mutate function with adaptive strength and varied operators (inspired by Insp 1)
    def custom_mutate_params_adaptive(individual, current_strength, indpb_gene):
        mutated_individual = list(individual) # Create a mutable copy
        num_genes = len(mutated_individual)
        
        mutation_type_choice = random.random()

        if mutation_type_choice < 0.7:  # Gaussian noise (most common)
            # Apply Gaussian noise to a fraction of genes
            for i in range(num_genes):
                if random.random() < indpb_gene:
                    mutated_individual[i] += np.random.normal(0, current_strength)
        elif mutation_type_choice < 0.9:  # Random reset for a few parameters
            num_to_reset = random.randint(1, max(1, num_genes // 5))
            for _ in range(num_to_reset):
                idx = random.randrange(num_genes)
                if idx % 3 == 0: # Amplitude param
                    mutated_individual[idx] = random.uniform(0.0, 1.5)
                elif idx % 3 == 1: # Mean param
                    mutated_individual[idx] = random.uniform(0.0, 1.0)
                else: # Std Dev param
                    mutated_individual[idx] = random.uniform(0.005, 0.4)
        else: # Swap/shuffle-like mutation for two Gaussian components
            if N_BASIS_FUNCTIONS > 1:
                idx1 = random.randrange(N_BASIS_FUNCTIONS) * 3
                idx2 = random.randrange(N_BASIS_FUNCTIONS) * 3
                if idx1 != idx2:
                    # Swap parameters of two entire Gaussian components (amp, mean, std)
                    mutated_individual[idx1:idx1+3], mutated_individual[idx2:idx2+3] = \
                        mutated_individual[idx2:idx2+3], mutated_individual[idx1:idx1+3]

        # Ensure parameters stay within their valid ranges after mutation
        for i in range(N_BASIS_FUNCTIONS):
            mutated_individual[i*3 + 0] = max(0.0, mutated_individual[i*3 + 0]) # Amp >= 0
            mutated_individual[i*3 + 1] = np.clip(mutated_individual[i*3 + 1], 0.0, 1.0) # Mean in [0,1]
            mutated_individual[i*3 + 2] = np.clip(mutated_individual[i*3 + 2], 0.005, 0.4) # StdDev in [0.005, 0.4]
        
        # Update original individual with mutated values
        for i, val in enumerate(mutated_individual):
            individual[i] = val
            
        return individual,

    toolbox.register("select", tools.selTournament, tournsize=4)

    # --- Evolutionary Process with Time Limit, Parallelism, and Elitism ---
    start_time = time.time()
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    # Evaluate the initial population in parallel
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    generation = 0
    # The loop continues until the time limit is reached
    ga_time_limit = TIME_LIMIT_SECONDS * 0.9 # Allocate 90% for main GA, 10% for local search
    
    while time.time() - start_time < ga_time_limit:
        generation += 1
        
        # Adaptive Mutation Strength: Decays over time for exploration -> exploitation shift (Inspired by Insp 1)
        time_fraction = min(1.0, (time.time() - start_time) / ga_time_limit)
        # Stronger initial exploration, then rapid decay, with a minimum strength
        current_mutation_strength = 0.2 * (1.0 - time_fraction)**2 + 0.005 
        
        # Unregister and re-register mutate operator with current strength using partial for dynamic argument
        if 'mutate' in toolbox.register.__dict__: # Check if 'mutate' is already registered
            toolbox.unregister("mutate") 
        toolbox.register("mutate", partial(custom_mutate_params_adaptive, 
                                            current_strength=current_mutation_strength, 
                                            indpb_gene=IND_MUTATION_RATE))

        # Select the next generation individuals (including elites)
        elites = tools.selBest(pop, k=NUM_ELITES)
        elites = list(map(toolbox.clone, elites))

        # Select the rest of the offspring using tournament selection
        offspring = toolbox.select(pop, k=POP_SIZE - NUM_ELITES)
        
        # Apply crossover and mutation to the offspring
        # MUTPB_BASE is the probability that an individual will be mutated.
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB_BASE)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind: # Only evaluate if there are invalid individuals
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        
        # Replace the old population by the elites and the new offspring
        pop[:] = elites + offspring
        hof.update(pop)
        
    # --- Final Local Search Refinement Phase (inspired by Insp 1) ---
    local_search_start_time = time.time()
    remaining_time = TIME_LIMIT_SECONDS - (local_search_start_time - start_time)
    
    if hof and remaining_time > 5: # Only run if there's meaningful time left and a best individual exists
        best_individual_params = hof[0]
        current_best_score = hof[0].fitness.values[0]
        
        # Use a small, decaying mutation strength for local refinement
        local_search_strength = 0.02 # Starting strength for local search
        local_search_indpb_gene = 0.1 # Lower gene mutation probability for fine-tuning
        
        num_local_search_iterations = 0
        while time.time() - local_search_start_time < remaining_time:
            num_local_search_iterations += 1
            
            # Decay local search strength
            local_search_strength *= 0.99 
            local_search_strength = max(local_search_strength, 0.001) # Minimum strength
            
            # Create a mutated version of the current best individual
            neighbor_params = toolbox.clone(best_individual_params)
            # Use the custom adaptive mutation function with local search strength
            neighbor_params, = custom_mutate_params_adaptive(neighbor_params, 
                                                            current_strength=local_search_strength, 
                                                            indpb_gene=local_search_indpb_gene)
            
            neighbor_score = toolbox.evaluate(neighbor_params)[0]
            
            if neighbor_score > current_best_score:
                current_best_score = neighbor_score
                best_individual_params = neighbor_params
                # Update HallOfFame if a new global best is found
                hof.update([best_individual_params])
            
            if num_local_search_iterations % 50 == 0: # Periodically check time
                if time.time() - local_search_start_time >= remaining_time:
                    break

    pool.close()
    pool.join()

    if not hof:
        # Fallback if no valid individual is found (should not happen with proper bounds with diverse init)
        return [0.0] * SEQUENCE_LENGTH

    # Decode the best individual's parameters into the final f_values
    best_params = hof[0]
    final_f_values = decode_gaussian_params(best_params)
    
    return final_f_values.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
