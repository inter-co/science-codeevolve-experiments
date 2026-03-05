# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import time
from numba import jit

@jit(nopython=True)
def compute_autoconvolution_jit(f_vals, n_steps):
    """Compute autoconvolution efficiently using Numba"""
    # Create convolution result array
    g = np.zeros(2 * n_steps - 1)
    
    # Compute convolution manually for efficiency
    for i in range(n_steps):
        for j in range(n_steps):
            idx = i + j
            if idx < len(g):
                g[idx] += f_vals[i] * f_vals[j]
    
    return g

@jit(nopython=True)
def compute_c2_metrics_jit(g_vals):
    """Compute C2 metrics efficiently using Numba"""
    # Compute norms
    g_norm_1 = 0.0
    g_norm_inf = 0.0
    g_norm_2_sq = 0.0
    
    for i in range(len(g_vals)):
        val = g_vals[i]
        g_norm_1 += val
        g_norm_inf = max(g_norm_inf, val)
        g_norm_2_sq += val * val
    
    # Avoid division by zero
    if g_norm_1 <= 1e-12 or g_norm_inf <= 1e-12:
        return 0.0
    
    # Compute C2
    c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
    return c2

def evaluate_step_function(f_vals):
    """Evaluate a step function and return C2 value"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)
        
        # Compute autoconvolution
        n_steps = len(f_vals)
        if n_steps == 0:
            return 0.0
            
        g_vals = compute_autoconvolution_jit(f_vals, n_steps)
        
        # Compute C2 metric
        c2 = compute_c2_metrics_jit(g_vals)
        return c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0
    except Exception:
        return 0.0

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using evolutionary strategy"""
    # Use a more systematic approach: genetic algorithm with adaptive parameters
    np.random.seed(42)  # For reproducibility
    
    # Initial population size and generations
    pop_size = 50
    generations = 100
    n_steps = 500  # Fixed number of steps for consistency
    
    # Initialize population with diverse step functions
    population = []
    for _ in range(pop_size):
        # Create a mix of different patterns: peaks, ramps, flat regions
        pattern_type = np.random.choice(['peak', 'ramp', 'flat', 'mixed'])
        
        if pattern_type == 'peak':
            # Create a single high peak
            f_vals = np.zeros(n_steps)
            peak_pos = np.random.randint(0, n_steps)
            peak_height = np.random.uniform(0.5, 2.0)
            f_vals[peak_pos] = peak_height
        elif pattern_type == 'ramp':
            # Create a ramp pattern
            f_vals = np.linspace(0, 1, n_steps)
            # Randomly flip or scale
            if np.random.random() > 0.5:
                f_vals = f_vals[::-1]
            if np.random.random() > 0.5:
                f_vals *= np.random.uniform(0.5, 2.0)
        elif pattern_type == 'flat':
            # Create a flat distribution
            f_vals = np.ones(n_steps) * np.random.uniform(0.1, 1.0)
        else:  # mixed
            # Combine different patterns
            f_vals = np.zeros(n_steps)
            # Add some peaks
            for _ in range(np.random.randint(1, 5)):
                pos = np.random.randint(0, n_steps)
                height = np.random.uniform(0.5, 2.0)
                f_vals[pos] = height
            # Add some ramp components
            ramp_start = np.random.randint(0, n_steps//2)
            ramp_end = np.random.randint(n_steps//2, n_steps)
            ramp_vals = np.linspace(0.1, 1.0, ramp_end - ramp_start)
            f_vals[ramp_start:ramp_end] = np.maximum(f_vals[ramp_start:ramp_end], ramp_vals)
            
        # Add some noise to make it more realistic
        noise_factor = 0.1
        f_vals = np.maximum(f_vals + np.random.normal(0, noise_factor, n_steps), 0)
        
        population.append(f_vals)
    
    best_fitness = 0.0
    best_individual = None
    
    # Evolutionary process
    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_step_function(individual)
            fitness_scores.append(fitness)
            
        # Track best
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Selection: keep top 50% 
        sorted_indices = np.argsort(fitness_scores)[::-1]
        selected_population = [population[i] for i in sorted_indices[:pop_size//2]]
        
        # Create new population through crossover and mutation
        new_population = selected_population.copy()
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            parent1 = selected_population[np.random.randint(0, len(selected_population))]
            parent2 = selected_population[np.random.randint(0, len(selected_population))]
            
            # Simple crossover
            crossover_point = np.random.randint(1, len(parent1)-1)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            
            # Mutation: add small random perturbations
            mutation_rate = 0.1
            for i in range(len(child)):
                if np.random.random() < mutation_rate:
                    child[i] = max(0, child[i] + np.random.normal(0, 0.1))
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    # Return the best individual found
    if best_individual is not None:
        return best_individual.tolist()
    else:
        # Fallback to a simple construction if no good solution found
        return [1.0] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
