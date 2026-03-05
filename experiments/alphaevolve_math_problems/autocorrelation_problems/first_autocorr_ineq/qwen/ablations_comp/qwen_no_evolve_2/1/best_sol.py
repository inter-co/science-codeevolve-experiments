# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
from scipy.fft import fft, ifft
import random
from typing import List, Tuple
import time

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    Returns (C₁, 1/C₁) where C₁ = 2n * max(convolution) / (sum(sequence))²
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Convert to numpy array
    a = np.array(sequence)
    
    # Compute convolution using FFT for efficiency (O(n log n) instead of O(n²))
    # Convolution with itself
    conv_result = convolve(a, a, mode='full')
    
    # We only care about the valid convolution part (not the padding)
    # For a * a, we get 2*n-1 elements, centered around index n-1
    # But we're interested in the maximum value among all possible convolutions
    max_conv = np.max(conv_result)
    
    sum_a = np.sum(a)
    
    # Avoid division by zero
    if sum_a < 1e-10:
        return float('inf'), 0.0
    
    n = len(sequence)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    
    return c1, inv_c1

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with positive values"""
    # Generate random values in [0, 1] and scale appropriately
    return [random.uniform(0.1, 100.0) for _ in range(length)]

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Create a mutated version of the sequence"""
    new_seq = sequence.copy()
    
    # Randomly change some elements
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            # Apply small perturbation
            new_seq[i] = max(0.01, new_seq[i] * random.uniform(0.8, 1.2))
    
    # Occasionally add/remove elements to explore different sequence lengths
    if random.random() < 0.1 and len(new_seq) > 1:
        # Remove element
        idx = random.randint(0, len(new_seq) - 1)
        new_seq.pop(idx)
    elif random.random() < 0.1 and len(new_seq) < 1000:
        # Add element
        idx = random.randint(0, len(new_seq))
        new_seq.insert(idx, random.uniform(0.1, 100.0))
    
    return new_seq

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Create offspring from two parent sequences"""
    # Simple crossover: take first half from seq1, second half from seq2
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if seq1 else seq2
    
    split_point = min_len // 2
    child = seq1[:split_point] + seq2[split_point:]
    
    # Ensure we don't exceed maximum length
    if len(child) > 1000:
        child = child[:1000]
    
    return child

def evolutionary_search(max_time_seconds: float = 60.0) -> List[float]:
    """
    Evolutionary algorithm to find optimal sequence for maximizing 1/C₁
    """
    start_time = time.time()
    
    # Initial population
    population_size = 50
    population = []
    
    # Create initial population with varying sequence lengths
    for _ in range(population_size):
        length = random.randint(10, 500)  # Reasonable range
        individual = generate_random_sequence(length)
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
        
        # Selection - tournament selection
        selected_indices = []
        tournament_size = 3
        for _ in range(population_size):
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices.append(winner_idx)
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Keep best individual (elitism)
        if best_individual is not None:
            new_population.append(best_individual)
        
        # Generate rest through crossover and mutation
        while len(new_population) < population_size:
            # Select parents
            parent1_idx = random.choice(selected_indices)
            parent2_idx = random.choice(selected_indices)
            
            # Crossover
            child = crossover_sequences(population[parent1_idx], population[parent2_idx])
            
            # Mutation
            child = mutate_sequence(child, mutation_rate=0.1)
            
            new_population.append(child)
        
        # Trim to exact population size
        population = new_population[:population_size]
        
        # Occasionally introduce completely new individuals
        if generation % 10 == 0:
            for i in range(min(5, population_size // 10)):
                if len(population) < population_size:
                    length = random.randint(10, 500)
                    individual = generate_random_sequence(length)
                    population.append(individual)
    
    return best_individual if best_individual is not None else []

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    # Use evolutionary algorithm with time limit
    try:
        sequence = evolutionary_search(max_time_seconds=55.0)
        if not sequence:
            # Fallback to simple approach
            sequence = generate_random_sequence(random.randint(50, 200))
        return sequence
    except Exception as e:
        # Fallback in case of error
        return generate_random_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
