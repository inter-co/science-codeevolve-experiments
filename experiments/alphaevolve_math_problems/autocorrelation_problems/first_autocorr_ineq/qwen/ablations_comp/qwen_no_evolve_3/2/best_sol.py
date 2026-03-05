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
    Computes the autocorrelation constant C₁ for a given sequence.
    C₁ = 2n * max(convolution) / (sum(sequence))^2
    We want to maximize 1/C₁ = (sum(sequence))^2 / (2n * max(convolution))
    """
    if len(sequence) == 0:
        return 0
    
    # Ensure sequence is numpy array
    seq = np.array(sequence)
    
    # Compute convolution (auto-correlation)
    # Using full mode to get complete convolution
    conv_result = convolve(seq, seq, mode='full')
    
    # Take only the middle part (the actual auto-correlation)
    # For a sequence of length n, auto-correlation has length 2n-1
    # Middle element corresponds to zero lag
    middle_idx = len(conv_result) // 2
    max_conv = np.max(conv_result)
    
    # Sum of sequence squared
    sum_sq = np.sum(seq)**2
    
    # Avoid division by zero
    if sum_sq < 1e-12:
        return 0
    
    # Compute C₁
    n = len(seq)
    c1 = 2 * n * max_conv / sum_sq
    
    # Return 1/C₁ (what we want to maximize)
    return 1.0 / c1 if c1 > 0 else 0

def generate_random_sequence(length_range=(10, 500)):
    """Generate a random step function with specified length"""
    n = random.randint(*length_range)
    # Generate sequence with random heights (non-negative)
    sequence = [random.uniform(0, 100) for _ in range(n)]
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1):
    """Create a mutated version of the sequence"""
    new_seq = sequence.copy()
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            # Mutate by adding/subtracting a small random value
            change = random.uniform(-10, 10)
            new_seq[i] = max(0, new_seq[i] + change)  # Keep non-negative
    return new_seq

def crossover_sequences(seq1, seq2):
    """Perform crossover between two sequences"""
    if len(seq1) != len(seq2):
        # If lengths differ, truncate to shorter length
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    # Single point crossover
    crossover_point = random.randint(1, len(seq1) - 1)
    new_seq = seq1[:crossover_point] + seq2[crossover_point:]
    return new_seq

def evolutionary_search(max_time=60):
    """Evolutionary algorithm to find optimal sequence"""
    start_time = time.time()
    
    # Initial population
    population_size = 50
    population = [generate_random_sequence() for _ in range(population_size)]
    
    best_sequence = None
    best_inv_c1 = 0
    
    generation = 0
    
    while time.time() - start_time < max_time:
        generation += 1
        
        # Evaluate fitness (1/C₁) for all individuals
        fitness_scores = []
        for seq in population:
            inv_c1 = compute_autocorrelation_constant(seq)
            fitness_scores.append(inv_c1)
            
            if inv_c1 > best_inv_c1 and np.sum(seq) > 0.01:
                best_inv_c1 = inv_c1
                best_sequence = seq.copy()
        
        # Check if we've beaten the benchmark
        if best_inv_c1 > 1.0 / 1.5031:
            break
            
        # Selection - keep top 50% based on fitness
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        selected_indices = sorted_indices[:population_size//2]
        selected_population = [population[i] for i in selected_indices]
        
        # Create new population through crossover and mutation
        new_population = selected_population.copy()
        
        # Add some diversity with mutations
        while len(new_population) < population_size:
            # Select parents
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Mutation
            child = mutate_sequence(child, mutation_rate=0.1)
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Occasionally add completely new random sequences for diversity
        if generation % 10 == 0:
            for _ in range(5):
                population.append(generate_random_sequence())
    
    return best_sequence if best_sequence is not None else generate_random_sequence()

def search_for_best_sequence() -> list[float]:
    """Main function to search for the best coefficient sequence"""
    try:
        # Use evolutionary approach
        sequence = evolutionary_search(max_time=55)  # Leave some buffer time
        return sequence
    except Exception as e:
        # Fallback to simple approach if something fails
        return generate_random_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
