# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms for the autoconvolution of a step function.
    Returns (||g||₂², ||g||₁, ||g||∞) where g = f * f
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Create step function on [-1/4, 1/4] with equal spacing
    dx = 0.5 / (n - 1) if n > 1 else 0.5
    x = np.linspace(-0.25, 0.25, n)
    
    # Compute autoconvolution g = f * f using discrete convolution
    # We use the fact that f is piecewise constant
    g = signal.convolve(f_values, f_values, mode='full')
    
    # Adjust indices for proper positioning
    g_len = len(g)
    g_x = np.linspace(-0.5, 0.5, g_len)
    
    # Compute norms
    # ||g||₂² = ∫ g² dx ≈ sum(g² * dx) using trapezoidal rule approximation
    # But since we're dealing with piecewise linear interpolation,
    # we use a more accurate approach with piecewise quadratic integration
    
    # For simplicity with piecewise linear approach:
    # Using trapezoidal rule for ||g||₂² calculation
    g_squared = g * g
    # Trapezoidal rule: sum((y[i] + y[i+1])/2 * dx) for intervals
    if len(g) >= 2:
        dx_g = 0.5 / (g_len - 1) if g_len > 1 else 0.5
        g_norm_2_sq = np.trapz(g_squared, dx=dx_g)
    else:
        g_norm_2_sq = g[0] * g[0] if len(g) > 0 else 0.0
    
    # ||g||₁ = ∫ |g| dx ≈ sum(|g| * dx) 
    g_norm_1 = np.sum(np.abs(g)) * (0.5 / (g_len - 1)) if g_len > 1 else np.sum(np.abs(g))
    
    # ||g||∞ = max |g|
    g_norm_inf = np.max(np.abs(g)) if len(g) > 0 else 0.0
    
    return g_norm_2_sq, g_norm_1, g_norm_inf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞)
    """
    try:
        g_norm_2_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
            return 0.0
            
        c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
        return c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0
    except Exception:
        return 0.0

def create_initial_population(pop_size: int, min_steps: int = 10, max_steps: int = 500) -> List[List[float]]:
    """Create initial population of step function configurations"""
    population = []
    for _ in range(pop_size):
        n_steps = random.randint(min_steps, max_steps)
        # Generate step heights with some structure - prefer higher values at center
        heights = []
        for i in range(n_steps):
            # Prefer higher values around center, lower at edges
            pos = (i / (n_steps - 1) if n_steps > 1 else 0) * 2 - 1  # -1 to 1
            # Gaussian-like distribution centered at 0
            height = max(0, 1.0 - abs(pos) * 0.5 + random.random() * 0.3)
            heights.append(height)
        population.append(heights)
    return population

def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate a single individual"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add small random change
            mutated[i] = max(0, mutated[i] + random.gauss(0, 0.1))
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Single-point crossover between two parents"""
    if len(parent1) == 0 or len(parent2) == 0:
        return parent1 if len(parent1) > 0 else parent2
    
    min_len = min(len(parent1), len(parent2))
    if min_len == 0:
        return parent1 if len(parent1) > 0 else parent2
    
    crossover_point = random.randint(0, min_len - 1)
    
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def evolve_step_functions() -> List[float]:
    """
    Evolve step functions to maximize C₂ using a genetic algorithm
    """
    pop_size = 50
    generations = 100
    mutation_rate = 0.1
    
    # Initialize population
    population = create_initial_population(pop_size)
    
    best_fitness = 0.0
    best_individual = None
    
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_c2(individual)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
        
        # Selection: tournament selection
        new_population = []
        
        # Keep best individual (elitism)
        if best_individual is not None:
            new_population.append(best_individual)
        
        # Generate rest through selection and crossover
        while len(new_population) < pop_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(pop_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            
            # Select another parent
            tournament_indices2 = random.sample(range(pop_size), tournament_size)
            tournament_fitness2 = [fitness_scores[i] for i in tournament_indices2]
            winner_index2 = tournament_indices2[tournament_fitness2.index(max(tournament_fitness2))]
            
            # Crossover
            child = crossover(population[winner_index], population[winner_index2])
            
            # Mutation
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    # Return best individual found
    return best_individual if best_individual is not None else []

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using evolutionary approach."""
    # Try several different approaches and return the best one
    best_result = []
    best_c2 = 0.0
    
    # Run evolution multiple times with different seeds
    for seed in [42, 123, 456, 789, 999]:
        random.seed(seed)
        np.random.seed(seed)
        try:
            result = evolve_step_functions()
            c2 = evaluate_c2(result)
            if c2 > best_c2:
                best_c2 = c2
                best_result = result
        except Exception:
            continue
    
    return best_result if best_result else [1.0]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
