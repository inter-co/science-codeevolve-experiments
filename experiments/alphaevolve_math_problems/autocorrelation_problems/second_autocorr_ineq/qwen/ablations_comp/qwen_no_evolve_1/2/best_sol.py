# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from typing import List
import random

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the norms needed for C2 calculation.
    """
    # Create step function on [-1/4, 1/4] with given values
    n = len(f_values)
    if n == 0:
        return 0, 0, 0
    
    # Define the domain
    domain = np.linspace(-0.25, 0.25, 2*n+1)  # More points for better resolution
    step_width = domain[1] - domain[0]
    
    # Create the step function
    f = np.zeros_like(domain)
    for i in range(n):
        start_idx = i * 2
        end_idx = (i + 1) * 2
        f[start_idx:end_idx] = f_values[i]
    
    # Compute autoconvolution g = f * f
    # Using discrete convolution
    g = signal.convolve(f, f, mode='full')
    
    # Adjust indices for the correct domain
    g_domain = np.linspace(-0.5, 0.5, len(g))
    
    # Extract the part corresponding to [-0.25, 0.25] (where our function supports)
    mid_idx = len(g) // 2
    g_centered = g[mid_idx - n : mid_idx + n]
    
    # Compute norms
    g_squared = g_centered ** 2
    g_abs = np.abs(g_centered)
    
    # ||g||₂² (L2 norm squared)
    l2_norm_sq = np.sum(g_squared) * step_width
    
    # ||g||₁ (L1 norm)
    l1_norm = np.sum(g_abs) * step_width
    
    # ||g||∞ (infinity norm)
    linf_norm = np.max(g_abs)
    
    return l2_norm_sq, l1_norm, linf_norm

def calculate_c2(f_values: List[float]) -> float:
    """
    Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞)
    """
    l2_norm_sq, l1_norm, linf_norm = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if l1_norm == 0 or linf_norm == 0:
        return 0
    
    return l2_norm_sq / (l1_norm * linf_norm)

def evolve_step_functions() -> List[float]:
    """
    Evolve step functions using a genetic algorithm approach to maximize C2.
    """
    # Parameters
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    elite_size = 5
    
    # Initialize population
    population = []
    for _ in range(population_size):
        # Create random step function with 50-200 steps
        n_steps = random.randint(50, 200)
        individual = [random.uniform(0, 1) for _ in range(n_steps)]
        population.append(individual)
    
    best_individual = None
    best_c2 = 0
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            c2 = calculate_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_c2:
                best_c2 = c2
                best_individual = individual.copy()
        
        # Sort by fitness
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite
        new_population = population[:elite_size]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    return best_individual if best_individual is not None else []

def tournament_selection(population, fitness_scores, tournament_size=3):
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_index]

def crossover(parent1, parent2):
    """Perform uniform crossover between two individuals."""
    if len(parent1) != len(parent2):
        # Make them same length by truncating or padding
        min_len = min(len(parent1), len(parent2))
        parent1 = parent1[:min_len]
        parent2 = parent2[:min_len]
        
        # Pad shorter one
        if len(parent1) < len(parent2):
            parent1.extend([0] * (len(parent2) - len(parent1)))
        elif len(parent2) < len(parent1):
            parent2.extend([0] * (len(parent1) - len(parent2)))
    
    # Uniform crossover
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    return child

def mutate(individual, mutation_rate):
    """Mutate an individual."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = random.uniform(0, 1)
    return mutated

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Use evolutionary approach to find optimal step function
    try:
        result = evolve_step_functions()
        return result
    except Exception:
        # Fallback to simple approach if evolution fails
        return [random.uniform(0, 1) for _ in range(100)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
