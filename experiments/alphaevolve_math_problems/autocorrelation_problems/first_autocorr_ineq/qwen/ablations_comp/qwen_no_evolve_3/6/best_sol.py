# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List, Tuple
import time

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute C₁ = 2n * max(convolution) / (sum(sequence))²
    Returns (C₁, 1/C₁) for a given sequence
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Use FFT for efficient convolution
    conv = signal.fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    
    sum_seq = sum(sequence)
    if sum_seq < 0.01:
        return float('inf'), 0.0
    
    n = len(sequence)
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    
    return c1, inv_c1

def generate_random_step_function(n: int) -> List[float]:
    """Generate a random step function with n steps"""
    # Generate random heights, clip to [0, 1000]
    return [max(0, min(1000, random.uniform(0, 100))) for _ in range(n)]

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Create a mutated version of the sequence"""
    new_seq = sequence.copy()
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            # Add small random perturbation
            new_seq[i] += random.gauss(0, 0.1 * max(1, new_seq[i]))
            new_seq[i] = max(0, new_seq[i])  # Ensure non-negative
    return new_seq

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Create offspring from two parent sequences"""
    if len(seq1) != len(seq2):
        # If lengths differ, make them same by truncating or padding
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    # Single-point crossover
    point = random.randint(1, len(seq1) - 1)
    child = seq1[:point] + seq2[point:]
    
    # Add some random noise to maintain diversity
    for i in range(len(child)):
        if random.random() < 0.05:
            child[i] += random.gauss(0, 0.05 * max(1, child[i]))
            child[i] = max(0, child[i])
    
    return child

def optimize_with_evolutionary_algorithm(max_time_seconds: float = 60) -> List[float]:
    """
    Use evolutionary algorithm to optimize the step function
    """
    start_time = time.time()
    
    # Initial population
    population_size = 50
    population = []
    
    # Generate initial diverse population
    for _ in range(population_size):
        n = random.randint(50, 500)  # Random sequence length
        individual = generate_random_step_function(n)
        population.append(individual)
    
    best_individual = None
    best_inv_c1 = 0.0
    
    generation = 0
    while time.time() - start_time < max_time_seconds:
        generation += 1
        
        # Evaluate fitness (1/C₁) for all individuals
        fitness_scores = []
        for individual in population:
            _, inv_c1 = compute_autocorrelation_constant(individual)
            fitness_scores.append(inv_c1)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_inv_c1:
            best_inv_c1 = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Selection: tournament selection
        selected = []
        tournament_size = 3
        for _ in range(population_size):
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual.copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Mutation
            child = mutate_sequence(child, mutation_rate=0.15)
            
            # Ensure minimum size and non-zero sum
            if len(child) < 10:
                child.extend([random.uniform(0, 100) for _ in range(10 - len(child))])
            
            # Make sure sum is meaningful
            if sum(child) < 0.01:
                child[0] = max(0.01, child[0])
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Occasionally introduce completely new individuals
        if generation % 10 == 0:
            for i in range(min(5, population_size // 10)):
                if len(population) > i:
                    population[i] = generate_random_step_function(random.randint(50, 500))
    
    return best_individual if best_individual is not None else generate_random_step_function(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence"""
    try:
        # Try evolutionary algorithm first
        return optimize_with_evolutionary_algorithm(max_time_seconds=55)
    except Exception as e:
        # Fallback to simple random search if evolution fails
        print(f"Fallback due to error: {e}")
        return generate_random_step_function(200)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
