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
    
    # Ensure sequence has positive sum
    total_sum = sum(sequence)
    if total_sum < 0.01:
        return float('inf'), 0.0
    
    # Use FFT for efficient convolution
    n = len(sequence)
    
    # Convert to numpy array for FFT operations
    a = np.array(sequence)
    
    # Compute autoconvolution using FFT
    # For circular convolution: a * a = ifft(fft(a) * fft(a))
    # For linear convolution, we pad appropriately
    fft_a = fft(a, 2*n - 1)
    conv_fft = fft_a * fft_a
    conv_result = ifft(conv_fft).real
    
    # Extract the linear convolution part (first 2n-1 elements)
    # But we only care about the non-circular part which is the middle portion
    # Actually, let's just compute it correctly:
    # Linear convolution of a with itself gives 2n-1 elements
    linear_conv = convolve(a, a, mode='full')
    
    # Maximum value of the convolution
    max_conv = np.max(linear_conv)
    
    # Compute C₁ = 2n * max(convolution) / (sum(sequence))²
    c1 = 2 * n * max_conv / (total_sum ** 2)
    
    # Return both C₁ and 1/C₁
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    return c1, inv_c1

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with non-negative values"""
    return [random.uniform(0, 1000) for _ in range(length)]

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Create a mutated version of the sequence"""
    new_seq = sequence.copy()
    
    # Randomly change some elements
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            # Change element with some random value
            new_seq[i] = random.uniform(0, 1000)
    
    # Occasionally add/remove elements to explore different sequence lengths
    if random.random() < 0.05 and len(new_seq) > 1:
        # Remove an element
        idx = random.randint(0, len(new_seq) - 1)
        new_seq.pop(idx)
    elif random.random() < 0.05:
        # Add an element
        idx = random.randint(0, len(new_seq))
        new_seq.insert(idx, random.uniform(0, 1000))
    
    return new_seq

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences"""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Take alternating elements from both sequences
    min_len = min(len(seq1), len(seq2))
    new_seq = []
    
    for i in range(min_len):
        if i % 2 == 0:
            new_seq.append(seq1[i])
        else:
            new_seq.append(seq2[i])
    
    # Add remaining elements from longer sequence
    if len(seq1) > min_len:
        new_seq.extend(seq1[min_len:])
    elif len(seq2) > min_len:
        new_seq.extend(seq2[min_len:])
    
    return new_seq

def evolutionary_search(max_time_seconds: float = 60.0) -> List[float]:
    """
    Use evolutionary algorithm to find optimal sequence.
    This is a completely different approach from the LP-based method.
    """
    start_time = time.time()
    
    # Population parameters
    population_size = 50
    generations = 1000
    elite_size = 5
    
    # Initialize population
    population = []
    for _ in range(population_size):
        length = random.randint(10, 500)  # Reasonable range
        individual = generate_random_sequence(length)
        population.append(individual)
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Evolutionary process
    for gen in range(generations):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Evaluate fitness (inverse of C₁)
        fitness_scores = []
        for individual in population:
            _, inv_c1 = compute_autocorrelation_constant(individual)
            fitness_scores.append(inv_c1)
        
        # Update best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_inv_c1:
            best_inv_c1 = fitness_scores[max_fitness_idx]
            best_sequence = population[max_fitness_idx].copy()
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(elite_size):
            new_population.append(sorted_population[i].copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = random.randint(0, elite_size - 1)
            parent2_idx = random.randint(0, elite_size - 1)
            
            parent1 = sorted_population[parent1_idx]
            parent2 = sorted_population[parent2_idx]
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Mutation
            child = mutate_sequence(child, mutation_rate=0.1)
            
            new_population.append(child)
        
        population = new_population[:population_size]
    
    return best_sequence if best_sequence is not None else generate_random_sequence(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use evolutionary approach instead of LP
        sequence = evolutionary_search(max_time_seconds=60.0)
        return sequence
    except Exception as e:
        # Fallback to simple random generation if something goes wrong
        print(f"Evolutionary search failed: {e}")
        return generate_random_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
