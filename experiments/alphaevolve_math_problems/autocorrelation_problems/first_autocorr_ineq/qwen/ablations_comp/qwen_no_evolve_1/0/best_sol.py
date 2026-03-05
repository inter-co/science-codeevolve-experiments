# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy import signal
from scipy.optimize import differential_evolution
import time

def compute_autocorrelation_constant(sequence):
    """
    Compute the first autocorrelation inequality constant C₁ for a given sequence.
    C₁ = 2n * max(convolution) / (sum(sequence))²
    We want to maximize 1/C₁ = (sum(sequence))² / (2n * max(convolution))
    """
    if len(sequence) == 0:
        return 0
    
    # Ensure sequence has positive sum
    seq_sum = sum(sequence)
    if seq_sum < 0.01:
        return 0
    
    # Use FFT for efficient convolution
    # Autoconvolve the sequence
    conv = signal.convolve(sequence, sequence, mode='full')
    
    # Get the maximum value (this corresponds to the peak of the convolution)
    max_conv = max(conv)
    
    # Calculate C₁
    n = len(sequence)
    if max_conv == 0:
        return 0
    
    c1 = (2 * n * max_conv) / (seq_sum ** 2)
    
    # Return 1/C₁ (what we want to maximize)
    return 1.0 / c1 if c1 > 0 else 0

def generate_random_sequence(length_range=(10, 100)):
    """Generate a random step function with specified length"""
    n = random.randint(*length_range)
    # Generate heights in [0, 1000] range
    sequence = [random.uniform(0, 1000) for _ in range(n)]
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1):
    """Create a mutated version of the sequence"""
    new_sequence = sequence.copy()
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            # Randomly change height with some bounds
            new_sequence[i] = random.uniform(0, 1000)
    return new_sequence

def crossover_sequences(seq1, seq2):
    """Create offspring by combining two sequences"""
    if len(seq1) != len(seq2):
        # If lengths differ, truncate to shorter length
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    # Single-point crossover
    point = random.randint(0, len(seq1))
    new_seq = seq1[:point] + seq2[point:]
    
    # Ensure minimum length
    if len(new_seq) < 5:
        # Pad with random values if too short
        while len(new_seq) < 5:
            new_seq.append(random.uniform(0, 1000))
    
    return new_seq

def evolutionary_search(max_time=55.0):
    """
    Evolutionary algorithm to find optimal sequence
    Uses a population-based approach with selection, crossover, and mutation
    """
    start_time = time.time()
    
    # Initialize population
    population_size = 50
    population = []
    
    # Generate initial diverse population
    for _ in range(population_size):
        seq = generate_random_sequence((10, 200))
        population.append(seq)
    
    best_fitness = 0
    best_sequence = None
    
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        
        # Evaluate fitness for all individuals
        fitness_scores = []
        for seq in population:
            fitness = compute_autocorrelation_constant(seq)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = seq.copy()
        
        # Sort population by fitness (descending)
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep top 30% (elitism)
        elite_size = population_size // 3
        new_population = sorted_population[:elite_size]
        
        # Create offspring from top performers
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = random.randint(0, elite_size - 1)
            parent2_idx = random.randint(0, elite_size - 1)
            
            # Crossover
            child = crossover_sequences(sorted_population[parent1_idx], 
                                      sorted_population[parent2_idx])
            
            # Mutation
            child = mutate_sequence(child, mutation_rate=0.1)
            
            new_population.append(child)
        
        population = new_population
        
        # Occasionally introduce completely new sequences
        if generation % 10 == 0:
            for i in range(min(5, population_size // 10)):
                population[random.randint(0, population_size - 1)] = generate_random_sequence((10, 200))
    
    return best_sequence if best_sequence is not None else generate_random_sequence()

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence"""
    try:
        # Use evolutionary search
        sequence = evolutionary_search(max_time=55.0)
        
        # Validate the sequence
        fitness = compute_autocorrelation_constant(sequence)
        if fitness < 0.1:  # If fitness is very low, generate a better one
            sequence = generate_random_sequence((50, 300))
            
        return sequence
    except Exception as e:
        # Fallback to simple random approach
        return generate_random_sequence((50, 300))

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
