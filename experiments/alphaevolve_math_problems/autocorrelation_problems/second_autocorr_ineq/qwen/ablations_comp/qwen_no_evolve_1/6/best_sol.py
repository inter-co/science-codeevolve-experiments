# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time
import random
from typing import List, Tuple

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    f_values: step heights for equally spaced steps on [-1/4, 1/4]
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    if not f_values:
        return 0.0, 0.0, 0.0
    
    # Create step function
    n_steps = len(f_values)
    dx = 0.5 / n_steps  # spacing between steps
    x = np.linspace(-0.25, 0.25, n_steps, endpoint=False) + dx/2
    
    # Ensure non-negative values
    f = np.array(f_values)
    f = np.maximum(f, 0.0)
    
    # Compute autoconvolution using fast convolution
    # Normalize the step function so that integral equals sum of heights
    # For autoconvolution, we need to properly scale
    g = signal.convolve(f, f, mode='full')
    
    # The convolution result has 2*n_steps - 1 elements
    # We need to map back to appropriate x coordinates
    g_x = np.linspace(-0.5, 0.5, len(g), endpoint=False) + 0.5/len(g)
    
    # Compute norms
    # ||g||₂² = ∫ g(x)² dx ≈ sum(g²) * dx
    g_squared = g * g
    norm_g2_squared = np.sum(g_squared) * (0.5 / len(g))
    
    # ||g||₁ = ∫ |g(x)| dx ≈ sum(|g|) * dx  
    norm_g1 = np.sum(np.abs(g)) * (0.5 / len(g))
    
    # ||g||∞ = max |g(x)|
    norm_ginf = np.max(np.abs(g))
    
    return norm_g2_squared, norm_g1, norm_ginf

def compute_c2(f_values: List[float]) -> float:
    """
    Compute C2 value for given step function heights.
    """
    norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_g1 <= 1e-15 or norm_ginf <= 1e-15:
        return 0.0
    
    return norm_g2_sq / (norm_g1 * norm_ginf)

def construct_function() -> List[float]:
    """
    Evolutionary algorithm approach to construct step function with high C2 value.
    Uses a combination of genetic algorithm and local search optimizations.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Parameters for evolution
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Initialize population with diverse step functions
    def create_individual(n_steps: int = 200) -> List[float]:
        # Start with some basic patterns that might work well
        pattern_type = random.choice(['uniform', 'peak', 'double_peak', 'gaussian'])
        
        if pattern_type == 'uniform':
            return [random.uniform(0.5, 1.5) for _ in range(n_steps)]
        elif pattern_type == 'peak':
            # Create a single peak in the center
            peak_height = random.uniform(1.0, 2.0)
            half_width = n_steps // 4
            individual = [0.0] * n_steps
            center = n_steps // 2
            for i in range(max(0, center - half_width), min(n_steps, center + half_width)):
                # Gaussian-like decay
                dist = abs(i - center)
                individual[i] = peak_height * np.exp(-dist**2 / (2 * (half_width/2)**2))
            return individual
        elif pattern_type == 'double_peak':
            # Two peaks
            peak1_height = random.uniform(0.8, 1.5)
            peak2_height = random.uniform(0.8, 1.5)
            half_width = n_steps // 6
            individual = [0.0] * n_steps
            center1 = n_steps // 3
            center2 = 2 * n_steps // 3
            for i in range(n_steps):
                dist1 = abs(i - center1)
                dist2 = abs(i - center2)
                individual[i] = (peak1_height * np.exp(-dist1**2 / (2 * (half_width/2)**2)) +
                               peak2_height * np.exp(-dist2**2 / (2 * (half_width/2)**2)))
            return individual
        else:  # gaussian
            # Create a smooth bell curve
            sigma = n_steps / 10.0
            mu = n_steps / 2.0
            individual = []
            for i in range(n_steps):
                val = np.exp(-((i - mu)**2) / (2 * sigma**2))
                individual.append(random.uniform(0.5, 1.5) * val)
            return individual
    
    # Evaluate fitness of individuals
    def evaluate(individual: List[float]) -> float:
        try:
            return compute_c2(individual)
        except:
            return 0.0
    
    # Create initial population
    population = [create_individual(random.randint(100, 500)) for _ in range(population_size)]
    
    # Evolution loop
    best_fitness = 0.0
    best_individual = None
    
    for generation in range(generations):
        # Evaluate fitness of current population
        fitness_scores = [evaluate(individual) for individual in population]
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Selection: tournament selection
        def tournament_select():
            tournament_size = 3
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            return population[winner_index].copy()
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Keep best individual (elitism)
        new_population.append(best_individual.copy())
        
        while len(new_population) < population_size:
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            if random.random() < crossover_rate and len(parent1) > 1:
                crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            def mutate(individual: List[float]) -> List[float]:
                mutated = individual.copy()
                for i in range(len(mutated)):
                    if random.random() < mutation_rate:
                        # Add small random change
                        mutated[i] += random.gauss(0, 0.1) * mutated[i]
                        mutated[i] = max(0, mutated[i])  # Ensure non-negative
                return mutated
            
            new_population.append(mutate(child1))
            if len(new_population) < population_size:
                new_population.append(mutate(child2))
        
        population = new_population[:population_size]
    
    # Final evaluation of best individual
    if best_individual is not None:
        final_c2 = compute_c2(best_individual)
        # If the final evaluation shows a better result, use it
        if final_c2 > best_fitness:
            return best_individual
    
    return best_individual if best_individual is not None else [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
