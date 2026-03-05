# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import convolve
from scipy.optimize import differential_evolution
import time

def compute_autocorrelation_constant(sequence):
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    C₁ = 2n * max(convolution) / (sum(sequence))^2
    """
    if len(sequence) == 0:
        return float('inf')
    
    seq_sum = sum(sequence)
    if seq_sum < 0.01:
        return float('inf')
    
    # Compute convolution using fast convolution
    conv = convolve(sequence, sequence, mode='full')
    max_conv = max(conv)
    
    n = len(sequence)
    if max_conv <= 0:
        return float('inf')
    
    c1 = (2 * n * max_conv) / (seq_sum ** 2)
    return c1

def compute_inverse_c1(sequence):
    """Compute 1/C₁ for maximization purposes."""
    c1 = compute_autocorrelation_constant(sequence)
    if c1 <= 0:
        return 0
    return 1.0 / c1

def generate_random_sequence(length_range=(10, 100)):
    """Generate a random step function with specified length."""
    n = random.randint(*length_range)
    # Generate heights in [0, 1000] with some randomness
    sequence = [random.uniform(0, 1000) for _ in range(n)]
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1):
    """Mutate a sequence by randomly changing some elements."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = random.uniform(0, 1000)
    return mutated

def crossover_sequences(seq1, seq2):
    """Perform crossover between two sequences."""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    min_len = min(len(seq1), len(seq2))
    crossover_point = random.randint(0, min_len)
    
    # Create offspring by combining parts of both sequences
    child = seq1[:crossover_point] + seq2[crossover_point:]
    
    # Ensure we don't create empty sequences
    if len(child) == 0:
        child = [random.uniform(0, 1000)]
    
    return child

def evolutionary_search(max_time=50.0):
    """
    Evolutionary algorithm to find optimal step function.
    Uses a population-based approach with selection, crossover, and mutation.
    """
    start_time = time.time()
    
    # Population parameters
    population_size = 50
    generations = 1000
    elite_size = 5
    
    # Initialize population
    population = [generate_random_sequence() for _ in range(population_size)]
    
    best_sequence = None
    best_inv_c1 = 0
    
    # Main evolutionary loop
    for gen in range(generations):
        if time.time() - start_time > max_time:
            break
            
        # Evaluate fitness (inverse C₁)
        fitness_scores = []
        for individual in population:
            inv_c1 = compute_inverse_c1(individual)
            fitness_scores.append(inv_c1)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = individual.copy()
        
        # Sort population by fitness
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite individuals
        new_population = sorted_population[:elite_size]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(sorted_population, sorted_fitness, 3)
            parent2 = tournament_selection(sorted_population, sorted_fitness, 3)
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Mutation
            child = mutate_sequence(child, mutation_rate=0.1)
            
            new_population.append(child)
        
        population = new_population
    
    return best_sequence if best_sequence is not None else generate_random_sequence()

def tournament_selection(population, fitness_scores, tournament_size):
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_index]

def search_for_best_sequence() -> list[float]:
    """Main function to search for the best coefficient sequence."""
    # Try evolutionary approach first
    try:
        sequence = evolutionary_search(max_time=55.0)
        return sequence
    except Exception as e:
        # Fallback to simple random approach
        return generate_random_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
