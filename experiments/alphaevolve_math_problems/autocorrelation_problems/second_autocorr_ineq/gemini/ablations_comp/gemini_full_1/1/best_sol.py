# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
import time
from scipy.signal import fftconvolve
import numba
from deap import base, creator, tools, algorithms
from multiprocessing import Pool

# --- Global Parameters for Parametric Representation & Optimization ---
SEQUENCE_LENGTH = 25000
N_BASIS_FUNCTIONS = 10

# Step width for the high-resolution function f.
STEP_WIDTH_F = 0.5 / SEQUENCE_LENGTH

# --- C2 Calculation (Optimized with FFT and Numba) ---

@numba.jit(nopython=True, fastmath=True)
def _compute_norms_from_g(g_values: np.ndarray, dt_g: float) -> float:
    """Numba-accelerated helper to compute norms from the autoconvolution array g."""
    g_len = len(g_values)
    if g_len < 2: return 0.0
    
    y1 = g_values[:-1]
    y2 = g_values[1:]
    g_l2_sq = np.sum(y1**2 + y1*y2 + y2**2) * (dt_g / 3.0)

    g_abs = np.abs(g_values)
    g_l1 = np.sum(g_abs) / (g_len + 1)
    g_linf = np.max(g_abs)

    if g_l1 < 1e-12 or g_linf < 1e-12: return 0.0
    
    c2 = g_l2_sq / (g_l1 * g_linf)
    return c2 if np.isfinite(c2) else 0.0

def compute_c2(f_values: np.ndarray) -> float:
    """Calculates C2 using fftconvolve and a Numba helper for norm calculations."""
    if np.sum(f_values) < 1e-12: return 0.0
    g_values = fftconvolve(f_values, f_values, mode='full')
    return _compute_norms_from_g(g_values, STEP_WIDTH_F)

# --- Parametric Function Definition & DEAP-specific helpers ---

def decode_gaussian_params(params: list[float]) -> np.ndarray:
    """Decodes parameters into a sum of Gaussian basis functions."""
    f_values = np.zeros(SEQUENCE_LENGTH, dtype=np.float64)
    x_grid = np.arange(SEQUENCE_LENGTH)
    
    for i in range(N_BASIS_FUNCTIONS):
        amp, mean_norm, std_norm = params[i*3 : (i+1)*3]
        
        amp = max(0.0, amp)
        mean_norm = np.clip(mean_norm, 0.0, 1.0)
        std_norm = np.clip(std_norm, 0.01, 0.4)
        
        mean = mean_norm * (SEQUENCE_LENGTH - 1)
        std_dev = std_norm * SEQUENCE_LENGTH
        
        if std_dev < 1.0: std_dev = 1.0
        
        f_values += amp * np.exp(-0.5 * ((x_grid - mean) / std_dev)**2)
    
    f_values = np.maximum(0.0, f_values)
    max_val = np.max(f_values)
    if max_val > 1e-9: f_values /= max_val
    
    return f_values

def evaluate_c2_deap_wrapper(individual_params: list[float]) -> tuple[float,]:
    """DEAP wrapper to decode parameters, compute C2, and return a tuple."""
    return (compute_c2(decode_gaussian_params(individual_params)),)

# --- Population Seeding Strategies (inspired by Insp 1 & 2) ---

def create_random_individual(creator_func):
    params = []
    for _ in range(N_BASIS_FUNCTIONS):
        params.extend([random.uniform(0.0, 1.5), random.uniform(0.0, 1.0), random.uniform(0.01, 0.4)])
    return creator_func(params)

def create_symmetric_individual(creator_func):
    params = []
    num_pairs = N_BASIS_FUNCTIONS // 2
    for _ in range(num_pairs):
        amp = random.uniform(0.5, 1.5)
        mean_norm = random.uniform(0.0, 0.5) 
        std = random.uniform(0.01, 0.4)
        params.extend([amp, mean_norm, std]); params.extend([amp, 1.0 - mean_norm, std])
    if N_BASIS_FUNCTIONS % 2 != 0:
        params.extend([random.uniform(0.5, 1.5), 0.5, random.uniform(0.01, 0.4)])
    return creator_func(params)

def create_sparse_individual(creator_func):
    """Generates an individual with only a few active Gaussians, inspired by Insp 1."""
    params = [0.0] * (N_BASIS_FUNCTIONS * 3)
    num_active = random.randint(1, 3)
    active_indices = random.sample(range(N_BASIS_FUNCTIONS), num_active)
    for i in active_indices:
        params[i*3] = random.uniform(0.5, 1.5); params[i*3+1] = random.uniform(0.0, 1.0); params[i*3+2] = random.uniform(0.01, 0.4)
    return creator_func(params)

# --- Main Optimization Driver ---

def construct_function() -> list[float]:
    """Uses a memetic algorithm (GA + local search) to optimize Gaussian parameters."""
    random.seed(42); np.random.seed(42)

    # --- GA Parameters ---
    POP_SIZE = 300; CXPB, MUTPB = 0.8, 0.6; TIME_LIMIT_SECONDS = 178
    GA_TIME_SHARE = 0.9; NUM_ELITES = int(0.05 * POP_SIZE)
    SYMMETRIC_SEED_RATIO, SPARSE_SEED_RATIO = 0.4, 0.2

    # --- DEAP Setup ---
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,)); creator.create("Individual", list, fitness=creator.FitnessMax)
    except RuntimeError: pass
    
    toolbox = base.Toolbox(); pool = Pool(); toolbox.register("map", pool.map)
    toolbox.register("random_individual", create_random_individual, creator.Individual)
    toolbox.register("symmetric_individual", create_symmetric_individual, creator.Individual)
    toolbox.register("sparse_individual", create_sparse_individual, creator.Individual)
    toolbox.register("evaluate", evaluate_c2_deap_wrapper); toolbox.register("mate", tools.cxBlend, alpha=0.5)
    
    def custom_mutate_params(individual, time_fraction):
        # Adaptive mutation strength and probability
        current_sigma_val = 0.25 * (1.0 - time_fraction)**1.5 + 0.01
        current_indpb_val = 0.3 * (1.0 - time_fraction) + 0.1 # Probability for a single parameter to be affected by Gaussian noise

        mutation_type_choice = random.random()

        if mutation_type_choice < 0.6: # 60% Gaussian noise on individual parameters (most common)
            tools.mutGaussian(individual, mu=0.0, sigma=current_sigma_val, indpb=current_indpb_val)
        elif mutation_type_choice < 0.75: # 15% Random reset of one or two Gaussian parameter sets
            num_resets = random.randint(1, min(2, N_BASIS_FUNCTIONS))
            for _ in range(num_resets):
                idx = random.randint(0, N_BASIS_FUNCTIONS - 1)
                individual[idx*3] = random.uniform(0.0, 1.5) # Amp
                individual[idx*3+1] = random.uniform(0.0, 1.0) # Mean
                individual[idx*3+2] = random.uniform(0.01, 0.4) # Std
        elif mutation_type_choice < 0.9: # 15% Swap two Gaussian parameter sets
            if N_BASIS_FUNCTIONS >= 2:
                idx1, idx2 = random.sample(range(N_BASIS_FUNCTIONS), 2)
                for j in range(3): # Swap amp, mean, std
                    individual[idx1*3+j], individual[idx2*3+j] = individual[idx2*3+j], individual[idx1*3+j]
        else: # 10% Block mutation (e.g., scale amplitudes of a block or shift means)
            if N_BASIS_FUNCTIONS >= 2:
                block_size = random.randint(1, max(1, N_BASIS_FUNCTIONS // 2))
                start_idx_gaussian = random.randint(0, N_BASIS_FUNCTIONS - block_size)
                
                block_op = random.choice(['scale_amp', 'shift_mean'])
                for i in range(start_idx_gaussian, start_idx_gaussian + block_size):
                    if block_op == 'scale_amp':
                        individual[i*3] *= random.uniform(0.5, 1.5)
                    elif block_op == 'shift_mean':
                        individual[i*3+1] += random.uniform(-0.1, 0.1) # Small shift

        # Ensure parameters stay within their valid ranges after mutation
        for i in range(N_BASIS_FUNCTIONS):
            individual[i*3] = max(0.0, individual[i*3])
            individual[i*3+1] = np.clip(individual[i*3+1], 0.0, 1.0)
            individual[i*3+2] = np.clip(individual[i*3+2], 0.01, 0.4)
        return individual,
    toolbox.register("mutate", custom_mutate_params)
    toolbox.register("select", tools.selTournament, tournsize=5)

    # --- Global GA Search Phase ---
    start_time = time.time(); ga_time_limit = start_time + TIME_LIMIT_SECONDS * GA_TIME_SHARE
    num_sym = int(POP_SIZE * SYMMETRIC_SEED_RATIO); num_sparse = int(POP_SIZE * SPARSE_SEED_RATIO)
    num_rand = POP_SIZE - num_sym - num_sparse
    pop = ([toolbox.symmetric_individual() for _ in range(num_sym)] + [toolbox.sparse_individual() for _ in range(num_sparse)] + [toolbox.random_individual() for _ in range(num_rand)])
    
    hof = tools.HallOfFame(1)
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
    hof.update(pop)

    while time.time() < ga_time_limit:
        time_fraction = (time.time() - start_time) / (ga_time_limit - start_time)
        elites = list(map(toolbox.clone, tools.selBest(pop, k=NUM_ELITES)))
        offspring = toolbox.select(pop, k=POP_SIZE - NUM_ELITES)
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB: toolbox.mate(child1, child2); del child1.fitness.values; del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB: toolbox.mutate(mutant, time_fraction=time_fraction); del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses): ind.fitness.values = fit
        pop[:] = elites + offspring; hof.update(pop)

    # --- Local Search Refinement Phase (inspired by Insp 2) ---
    current_best_ind = creator.Individual(hof[0]); current_best_ind.fitness.values = hof[0].fitness.values
    local_search_strength_factor = 0.05 # Initial sigma factor for local search
    local_search_indpb = 0.05 # Probability for a parameter to be mutated in local search

    while time.time() - start_time < TIME_LIMIT_SECONDS:
        neighbors = [toolbox.clone(current_best_ind) for _ in range(POP_SIZE // 10)]
        for neighbor in neighbors:
            # Apply Gaussian mutation with decaying strength and low individual probability
            tools.mutGaussian(neighbor, mu=0.0, sigma=local_search_strength_factor, indpb=local_search_indpb)
            # Ensure bounds after mutation
            for i in range(N_BASIS_FUNCTIONS):
                neighbor[i*3] = max(0.0, neighbor[i*3]); neighbor[i*3+1] = np.clip(neighbor[i*3+1], 0.0, 1.0); neighbor[i*3+2] = np.clip(neighbor[i*3+2], 0.01, 0.4)
            del neighbor.fitness.values
        fitnesses = toolbox.map(toolbox.evaluate, neighbors)
        for ind, fit in zip(neighbors, fitnesses): ind.fitness.values = fit
        best_neighbor = tools.selBest(neighbors, k=1)[0]
        if best_neighbor.fitness.values[0] > current_best_ind.fitness.values[0]:
            current_best_ind = best_neighbor
        else: 
            local_search_strength_factor *= 0.8 # Decay strength if no improvement
        if local_search_strength_factor < 1e-4: break # Stop if search radius is too small
    hof.update([current_best_ind])

    pool.close(); pool.join()
    return decode_gaussian_params(hof[0]).tolist() if hof else [0.0] * SEQUENCE_LENGTH

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
