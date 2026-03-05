# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List

def compute_autoconvolution_norms(f_values: List[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    - ||g||₂² (L2 norm squared of autoconvolution)
    - ||g||₁ (L1 norm of autoconvolution)  
    - ||g||∞ (L-infinity norm of autoconvolution)
    """
    # Create step function on [-1/4, 1/4] with given heights
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Step width
    step_width = 0.5 / n
    
    # Compute autoconvolution g = f * f using discrete convolution
    # f is treated as piecewise constant on intervals
    # We'll use scipy's convolution which handles the discrete case properly
    g = signal.convolve(f_values, f_values, mode='full')
    
    # The convolution result has length 2*n - 1, centered around index n-1
    # But we're interested in the autoconvolution over [-1/2, 1/2] 
    # For our purposes, we just need the norms of the resulting sequence
    
    # Compute the norms
    g_squared = np.array(g)**2
    g_abs = np.abs(np.array(g))
    
    # ||g||₂² = sum of squares of g
    norm_g_2_sq = np.sum(g_squared)
    
    # ||g||₁ = sum of absolute values of g
    norm_g_1 = np.sum(g_abs)
    
    # ||g||∞ = max absolute value of g
    norm_g_inf = np.max(g_abs)
    
    return norm_g_2_sq, norm_g_1, norm_g_inf

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 value for given step function."""
    norm_g_2_sq, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_g_1 <= 1e-15 or norm_g_inf <= 1e-15:
        return 0.0
    
    return norm_g_2_sq / (norm_g_1 * norm_g_inf)

def evolve_step_function() -> List[float]:
    """
    Evolve a step function using evolutionary strategy to maximize C2.
    Uses a combination of genetic algorithm and local search.
    """
    # Parameters
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Initial population
    def create_individual():
        # Random number of steps between 50 and 500
        n_steps = random.randint(50, 500)
        # Random heights between 0 and 1
        return [random.random() for _ in range(n_steps)]
    
    def fitness(individual):
        return calculate_c2(individual)
    
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [fitness(ind) for ind in population]
        
        # Selection (tournament selection)
        def select_parent():
            tournament_size = 3
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            return population[winner_index][:]
        
        # Create new population
        new_population = []
        
        # Keep best individual (elitism)
        best_idx = np.argmax(fitness_scores)
        new_population.append(population[best_idx][:])
        
        # Generate rest through crossover and mutation
        while len(new_population) < population_size:
            # Selection
            parent1 = select_parent()
            parent2 = select_parent()
            
            # Crossover
            if random.random() < crossover_rate and len(parent1) > 1 and len(parent2) > 1:
                # Uniform crossover
                child1, child2 = [], []
                min_len = min(len(parent1), len(parent2))
                for i in range(min_len):
                    if random.random() < 0.5:
                        child1.append(parent1[i])
                        child2.append(parent2[i])
                    else:
                        child1.append(parent2[i])
                        child2.append(parent1[i])
                
                # Add remaining elements from longer parent
                if len(parent1) > min_len:
                    child1.extend(parent1[min_len:])
                elif len(parent2) > min_len:
                    child2.extend(parent2[min_len:])
                    
                child1 = child1[:len(parent1)]  # Trim to original size
                child2 = child2[:len(parent2)]
            else:
                child1, child2 = parent1[:], parent2[:]
            
            # Mutation
            for i in range(len(child1)):
                if random.random() < mutation_rate:
                    child1[i] = max(0, child1[i] + random.gauss(0, 0.1))  # Ensure non-negative
            
            for i in range(len(child2)):
                if random.random() < mutation_rate:
                    child2[i] = max(0, child2[i] + random.gauss(0, 0.1))  # Ensure non-negative
            
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        population = new_population
    
    # Return best individual
    final_fitnesses = [calculate_c2(ind) for ind in population]
    best_idx = np.argmax(final_fitnesses)
    return population[best_idx]

def construct_function() -> list[float]:
    """Construct optimized step function with high C2 value."""
    # Try multiple random initializations and pick the best
    best_result = None
    best_c2 = -1
    
    for _ in range(5):  # Try 5 different evolutions
        try:
            result = evolve_step_function()
            c2 = calculate_c2(result)
            if c2 > best_c2:
                best_c2 = c2
                best_result = result
        except Exception:
            continue
    
    # If no evolution worked, fall back to a reasonable construction
    if best_result is None:
        # Create a simple symmetric pattern that tends to work well
        n_steps = 200
        result = [0.0] * n_steps
        # Create a peak in the middle with gradual decay
        for i in range(n_steps):
            center = n_steps // 2
            distance = abs(i - center)
            # Gaussian-like shape
            result[i] = max(0, 1.0 - distance * 0.01)
        
        # Normalize to prevent extreme values
        total = sum(result)
        if total > 0:
            result = [x / total * 10 for x in result]
        
        return result
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
