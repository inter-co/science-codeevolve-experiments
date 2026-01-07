# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from deap import base, creator, tools, algorithms
import random
import multiprocessing
import time
from scipy.optimize import minimize
from collections import deque

# --- Global Configuration & Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- Fitness Function ---
def calculate_inv_c1(sequence: list[float]) -> tuple[float,]:
    a = np.array(sequence, dtype=np.float64)
    n = len(a)
    a_clipped = np.clip(a, 0.0, 1000.0)
    sum_a = np.sum(a_clipped)

    if n == 0 or sum_a < 0.01:
        return 0.0,

    conv_len = 2 * n - 1
    fft_size = int(2**np.ceil(np.log2(conv_len))) if conv_len > 0 else 1
    
    A_fft = np.fft.fft(a_clipped, fft_size)
    B_fft = A_fft * A_fft
    b_full = np.fft.ifft(B_fft).real
    b = b_full[:conv_len] if conv_len > 0 else np.array([])
    max_b = np.max(b) if b.size > 0 else 0

    if max_b < 1e-9:
        return 0.0,

    denominator = (2 * n * max_b)
    if denominator == 0:
        return 0.0,

    inv_c1 = (sum_a * sum_a) / denominator
    return inv_c1,

# --- DEAP Setup ---
if hasattr(creator, "FitnessMax"): del creator.FitnessMax
if hasattr(creator, "Individual"): del creator.Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
MIN_INIT_SEQ_LEN = 20
MAX_INIT_SEQ_LEN = 400
MAX_EVO_SEQ_LEN = 1000
MIN_EVO_SEQ_LEN = 20

toolbox.register("attr_float", random.uniform, 0.0, 1000.0)

def generate_individual(min_len, max_len):
    length = random.randint(min_len, max_len)
    ind = creator.Individual([toolbox.attr_float() for _ in range(length)])
    if all(x == 0.0 for x in ind):
        if length > 0: ind[random.randint(0, length - 1)] = random.uniform(0.1, 10.0)
    return ind

toolbox.register("individual", generate_individual, MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", calculate_inv_c1)

# --- Advanced Genetic Operators ---
def var_len_cxBlend_and_swapTails(ind1, ind2, alpha, min_len, max_len):
    child1, child2 = list(ind1), list(ind2)
    min_l = min(len(child1), len(child2))
    for i in range(min_l):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        x1, x2 = child1[i], child2[i]
        child1[i] = (1. - gamma) * x1 + gamma * x2
        child2[i] = gamma * x1 + (1. - gamma) * x2
    child1[min_l:], child2[min_l:] = child2[min_l:], child1[min_l:]
    for i, child_list in enumerate([child1, child2]):
        if len(child_list) > max_len: child_list = child_list[:max_len]
        elif len(child_list) < min_len:
            child_list.extend([toolbox.attr_float() for _ in range(min_len - len(child_list))])
        target_ind = ind1 if i == 0 else ind2
        target_ind[:] = child_list
    return ind1, ind2

def constructive_mutation(individual, mu, sigma, indpb_val, prob_len_mut, prob_struct_mut, min_len, max_len, step_len_change):
    for i in range(len(individual)):
        if random.random() < indpb_val:
            individual[i] += random.gauss(mu, sigma)
    if random.random() < prob_struct_mut:
        current_len = len(individual)
        if min_len <= current_len <= max_len and current_len > 1:
            mutation_type = random.choice(['mirror_half', 'repeat_segment', 'shuffle_segment', 'reverse_segment'])
            temp_ind = list(individual)
            if mutation_type == 'mirror_half':
                half_len = current_len // 2
                if half_len > 0:
                    if current_len % 2 == 1: temp_ind[half_len+1:] = temp_ind[:half_len][::-1]
                    else: temp_ind[half_len:] = temp_ind[:half_len][::-1]
            elif mutation_type == 'repeat_segment' and current_len > 4:
                start_idx, end_idx = sorted(random.sample(range(current_len // 2), 2))
                segment = temp_ind[start_idx:end_idx]
                insert_idx = random.randint(0, current_len)
                temp_ind[insert_idx:insert_idx] = segment
            elif mutation_type == 'shuffle_segment' and current_len > 2:
                start_idx, end_idx = sorted(random.sample(range(current_len), 2))
                segment_to_shuffle = temp_ind[start_idx:end_idx+1]
                random.shuffle(segment_to_shuffle)
                temp_ind[start_idx:end_idx+1] = segment_to_shuffle
            elif mutation_type == 'reverse_segment' and current_len > 2:
                start_idx, end_idx = sorted(random.sample(range(current_len), 2))
                segment_to_reverse = temp_ind[start_idx:end_idx+1]
                temp_ind[start_idx:end_idx+1] = segment_to_reverse[::-1]
            individual[:] = temp_ind
    if random.random() < prob_len_mut:
        current_len = len(individual)
        if random.random() < 0.5 and current_len < max_len:
            num_to_add = random.randint(1, min(step_len_change, max_len - current_len))
            for _ in range(num_to_add): individual.insert(random.randint(0, len(individual)), toolbox.attr_float())
        elif current_len > min_len:
            num_to_remove = random.randint(1, min(step_len_change, current_len - min_len))
            for _ in range(num_to_remove):
                if len(individual) > min_len: del individual[random.randint(0, len(individual) - 1)]
    individual[:] = [max(0.0, min(1000.0, x)) for x in individual]
    return individual,

toolbox.register("mate", var_len_cxBlend_and_swapTails, alpha=0.5, min_len=MIN_EVO_SEQ_LEN, max_len=MAX_EVO_SEQ_LEN)
toolbox.register("mutate", constructive_mutation, mu=0.0, indpb_val=0.15, prob_len_mut=0.25, prob_struct_mut=0.15, min_len=MIN_EVO_SEQ_LEN, max_len=MAX_EVO_SEQ_LEN, step_len_change=30)
toolbox.register("select", tools.selTournament, tournsize=7)

# --- Memetic Algorithm Main Function with Stagnation Control ---
def search_for_best_sequence() -> list[float]:
    TIME_LIMIT_GA = 155
    TIME_LIMIT_TOTAL = 178
    POP_SIZE = 500
    CXPB, MUTPB = 0.9, 1.0
    INITIAL_SIGMA, FINAL_SIGMA = 50.0, 0.5
    NUM_SEED_INDIVIDUALS = int(POP_SIZE * 0.2)
    
    # Parameters for stagnation control
    STAGNATION_WINDOW = 15
    STAGNATION_THRESHOLD = 1e-5
    DIVERSITY_INJECTION_RATIO = 0.2

    def _initialize_population_seeded(pop_size, num_seeds):
        population = []
        for _ in range(num_seeds // 2):
            n = random.randint(MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN)
            seq = [0.0] * n; k = max(1, int(np.sqrt(n))); indices = random.sample(range(n), min(k, n))
            for idx in indices: seq[idx] = random.uniform(1.0, 20.0)
            population.append(creator.Individual(seq))
        for _ in range(num_seeds - (num_seeds // 2)):
            length = random.randint(MIN_INIT_SEQ_LEN, MAX_INIT_SEQ_LEN)
            half_len = (length + 1) // 2
            half_seq = [toolbox.attr_float() for _ in range(half_len)]
            full_seq = half_seq + (half_seq[:-1][::-1] if length % 2 == 1 else half_seq[::-1])
            population.append(creator.Individual(full_seq))
        population.extend(toolbox.population(n=pop_size - len(population)))
        return population

    def local_search_objective(x: np.ndarray) -> float:
        return -calculate_inv_c1(x.tolist())[0]

    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)
        pop = _initialize_population_seeded(POP_SIZE, NUM_SEED_INDIVIDUALS)
        hof = tools.HallOfFame(1)
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
        hof.update(pop)
        start_time = time.time(); gen = 0
        fitness_history = deque(maxlen=STAGNATION_WINDOW)
        
        while time.time() - start_time < TIME_LIMIT_GA:
            gen += 1; offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB: toolbox.mate(child1, child2); del child1.fitness.values; del child2.fitness.values
            time_ratio = min(1.0, (time.time() - start_time) / TIME_LIMIT_GA)
            current_sigma = FINAL_SIGMA + (INITIAL_SIGMA - FINAL_SIGMA) * (1 - time_ratio)**2
            for mutant in offspring:
                if random.random() < MUTPB: toolbox.mutate(mutant, sigma=current_sigma); del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses): ind.fitness.values = fit
            pop[:] = offspring; hof.update(pop)
            
            # --- Stagnation Detection and Diversity Injection ---
            fitness_history.append(hof[0].fitness.values[0])
            if (gen > STAGNATION_WINDOW and len(fitness_history) == STAGNATION_WINDOW and (max(fitness_history) - min(fitness_history)) < STAGNATION_THRESHOLD):
                print(f"--- Stagnation detected at gen {gen}. Injecting diversity. ---")
                fitness_history.clear()
                pop.sort(key=lambda ind: ind.fitness.values[0])
                num_to_replace = int(POP_SIZE * DIVERSITY_INJECTION_RATIO)
                new_individuals = [toolbox.individual() for _ in range(num_to_replace)]
                pop[:num_to_replace] = new_individuals
                new_fitnesses = toolbox.map(toolbox.evaluate, new_individuals)
                for ind, fit in zip(new_individuals, new_fitnesses): ind.fitness.values = fit
        
        print(f"GA phase finished after {gen} generations. Best fitness: {hof[0].fitness.values[0]:.6f}")

        if not hof: return [1.0] * MIN_EVO_SEQ_LEN
        best_from_ga = hof[0]; best_from_ga_fitness = best_from_ga.fitness.values[0]
        best_overall_sequence = np.array(best_from_ga)

        if TIME_LIMIT_TOTAL - (time.time() - start_time) > 5 and len(best_from_ga) > 0:
            print("Starting polishing step...")
            res = minimize(local_search_objective, np.array(best_from_ga), method='L-BFGS-B', bounds=[(0.0, 1000.0)] * len(best_from_ga), options={'maxiter': 500, 'disp': False})
            if -res.fun > best_from_ga_fitness:
                print(f"Polishing improved fitness to {-res.fun:.6f}"); best_overall_sequence = res.x
            else: print("Polishing did not improve fitness.")
    
    final_sequence = np.clip(best_overall_sequence, 0.0, 1000.0).tolist()
    if np.sum(final_sequence) < 0.01: return [1.0] * MIN_EVO_SEQ_LEN
    return final_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
