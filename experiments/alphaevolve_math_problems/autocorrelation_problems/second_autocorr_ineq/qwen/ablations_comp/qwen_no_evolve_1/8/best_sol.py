# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List
import time

def evaluate_c2(f_values: List[float]) -> tuple[float, float, float]:
    """
    Evaluate C2 for a given step function represented by f_values.
    Returns (c2, benchmark_ratio, eval_time)
    """
    start_time = time.time()
    
    # Define domain [-1/4, 1/4] with appropriate spacing
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0
    
    # Create step function on [-1/4, 1/4]
    x_domain = np.linspace(-0.25, 0.25, n_steps + 1)
    dx = x_domain[1] - x_domain[0]
    
    # Ensure non-negative values
    f_values = [max(0, val) for val in f_values]
    
    # Compute autoconvolution g = f * f
    # Using discrete convolution
    g = signal.convolve(f_values, f_values, mode='full')
    
    # Adjust for proper scaling (convolution normalization)
    g = g * dx
    
    # Compute norms
    g_abs = np.abs(g)
    
    # ||g||₂² using trapezoidal-like piecewise integration
    # For piecewise linear segments, integrate quadratic approximation
    g_norm_2_squared = 0.0
    for i in range(len(g) - 1):
        y1, y2 = g[i], g[i+1]
        h = dx  # Assuming equal spacing
        # Trapezoidal rule for integral of g^2
        # Using quadratic approximation: (h/3)(y1^2 + y1*y2 + y2^2)
        g_norm_2_squared += (h/3.0) * (y1*y1 + y1*y2 + y2*y2)
    
    # ||g||₁
    g_norm_1 = np.sum(g_abs) * dx  # Approximate integral
    
    # ||g||∞
    g_norm_inf = np.max(g_abs)
    
    # Avoid division by zero
    if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
        c2 = 0.0
    else:
        c2 = g_norm_2_squared / (g_norm_1 * g_norm_inf)
    
    eval_time = time.time() - start_time
    benchmark_ratio = c2 / 0.962
    
    return c2, benchmark_ratio, eval_time

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using evolutionary approach."""
    
    # Parameters for evolutionary search
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    elite_size = 5
    
    # Initialize population with random step functions
    def generate_random_function(length: int) -> list[float]:
        return [random.uniform(0, 1) for _ in range(length)]
    
    def mutate_function(f_values: list[float]) -> list[float]:
        mutated = f_values.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = max(0, mutated[i] + random.gauss(0, 0.1))
        return mutated
    
    def crossover(parent1: list[float], parent2: list[float]) -> list[float]:
        # Uniform crossover
        child = []
        for i in range(min(len(parent1), len(parent2))):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    # Initial population
    population = [generate_random_function(random.randint(50, 500)) for _ in range(population_size)]
    
    best_c2 = 0.0
    best_solution = None
    
    # Evolutionary loop
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            c2, _, _ = evaluate_c2(individual)
            fitness_scores.append(c2)
        
        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_c2:
            best_c2 = fitness_scores[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()
        
        # Selection (tournament selection)
        selected = []
        for _ in range(population_size):
            tournament_size = 3
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        # Elitism - keep best individuals
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elite = [population[i].copy() for i in elite_indices]
        
        # Crossover and mutation
        new_population = elite.copy()
        while len(new_population) < population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = crossover(parent1, parent2)
            child = mutate_function(child)
            new_population.append(child)
        
        population = new_population[:population_size]
    
    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to simple construction if no evolution worked
        return [0.5] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
