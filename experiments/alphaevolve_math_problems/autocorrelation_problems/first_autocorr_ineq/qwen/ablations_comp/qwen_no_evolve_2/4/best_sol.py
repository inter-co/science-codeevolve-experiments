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
    
    Returns:
        tuple: (C₁, 1/C₁) where C₁ = 2n * max(convolution) / (sum(sequence))²
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure sequence is numpy array
    a = np.array(sequence)
    sum_a = np.sum(a)
    
    # Check if sum is too small
    if sum_a < 0.01:
        return float('inf'), 0.0
    
    # Compute convolution using FFT for efficiency (O(n log n) instead of O(n²))
    # We compute a * a (autoconvolution)
    n = len(a)
    
    # Using FFT-based convolution
    # For autoconvolution, we can use: conv(a,a) = ifft(fft(a)^2)
    fft_a = fft(a, 2*n - 1)  # Zero-padding to avoid circular convolution effects
    conv_fft = fft_a * fft_a
    b = ifft(conv_fft).real[:n]  # Take only the first n elements
    
    # The convolution result gives us the autoconvolution values
    # But we need the full convolution result for the maximum
    full_conv = ifft(conv_fft).real  # Full convolution result
    
    # Maximum value of the autoconvolution
    max_b = np.max(full_conv)
    
    # Compute C₁ = 2n * max(b) / (sum(a))²
    C1 = (2 * n * max_b) / (sum_a ** 2)
    
    # Return both C₁ and 1/C₁
    return C1, 1.0 / C1

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with non-negative values."""
    return [random.uniform(0, 1000) for _ in range(length)]

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Create a mutated version of the sequence."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Random change to the element
            mutated[i] = max(0, mutated[i] + random.gauss(0, 10))
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    if len(seq1) != len(seq2):
        # Make them same length by truncating or padding
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    # Single-point crossover
    point = random.randint(1, len(seq1) - 1)
    child = seq1[:point] + seq2[point:]
    return child

def evolutionary_search(max_time_seconds: float = 60.0) -> List[float]:
    """
    Use evolutionary algorithm to find the best sequence.
    """
    start_time = time.time()
    
    # Initial population
    population_size = 50
    population = []
    
    # Generate initial population with varying lengths
    for _ in range(population_size):
        length = random.randint(10, 500)
        individual = generate_random_sequence(length)
        population.append(individual)
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Evolutionary parameters
    generations = 0
    max_generations = 1000
    
    while time.time() - start_time < max_time_seconds and generations < max_generations:
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            try:
                _, inv_c1 = compute_autocorrelation_constant(individual)
                fitness_scores.append(inv_c1)
            except:
                fitness_scores.append(0.0)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_inv_c1:
            best_inv_c1 = fitness_scores[max_fitness_idx]
            best_sequence = population[max_fitness_idx].copy()
        
        # Selection - tournament selection
        selected = []
        tournament_size = 3
        for _ in range(population_size):
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Elitism: keep the best individual
        new_population.append(best_sequence.copy())
        
        # Generate offspring
        while len(new_population) < population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Mutation
            child = mutate_sequence(child, mutation_rate=0.1)
            
            # Ensure minimum length
            if len(child) < 5:
                child.extend([random.uniform(0, 1000) for _ in range(5 - len(child))])
            
            # Clip values to [0, 1000]
            child = [max(0, min(1000, x)) for x in child]
            
            new_population.append(child)
        
        population = new_population[:population_size]
        generations += 1
    
    return best_sequence if best_sequence is not None else generate_random_sequence(100)

def direct_optimization_approach() -> List[float]:
    """
    Alternative approach: Direct optimization with gradient-free methods.
    """
    # Try some carefully constructed sequences
    candidates = []
    
    # Simple step functions
    for n in [50, 100, 200, 300, 400]:
        # Uniform step function
        uniform_seq = [1.0] * n
        candidates.append(uniform_seq)
        
        # Decreasing step function
        decreasing_seq = [1.0 / (i + 1) for i in range(n)]
        candidates.append(decreasing_seq)
        
        # Alternating pattern
        alternating_seq = [1.0 if i % 2 == 0 else 0.5 for i in range(n)]
        candidates.append(alternating_seq)
    
    # More sophisticated approach: golden ratio based
    golden_ratio = (1 + np.sqrt(5)) / 2
    for n in [100, 200, 300]:
        golden_seq = [golden_ratio ** (i % 10) for i in range(n)]
        candidates.append(golden_seq)
    
    # Try some random but structured sequences
    for _ in range(20):
        n = random.randint(50, 400)
        # Generate sequence with exponential decay
        seq = [np.exp(-i/10.0) for i in range(n)]
        candidates.append(seq)
    
    # Find the best among candidates
    best_inv_c1 = 0.0
    best_sequence = None
    
    for candidate in candidates:
        try:
            _, inv_c1 = compute_autocorrelation_constant(candidate)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = candidate
        except:
            continue
    
    return best_sequence if best_sequence is not None else generate_random_sequence(100)

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence.
    Uses a hybrid approach combining evolutionary search and direct optimization.
    """
    # Try direct optimization first
    direct_result = direct_optimization_approach()
    _, direct_inv_c1 = compute_autocorrelation_constant(direct_result)
    
    # Then try evolutionary search for potentially better results
    try:
        evolutionary_result = evolutionary_search(max_time_seconds=50.0)
        _, evol_inv_c1 = compute_autocorrelation_constant(evolutionary_result)
        
        # Return the better of the two
        if evol_inv_c1 > direct_inv_c1:
            return evolutionary_result
        else:
            return direct_result
    except:
        return direct_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
