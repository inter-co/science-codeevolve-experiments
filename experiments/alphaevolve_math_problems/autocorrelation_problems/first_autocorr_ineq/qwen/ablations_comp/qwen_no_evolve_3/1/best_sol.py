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
    Compute the autocorrelation constant C₁ and its reciprocal 1/C₁ for a sequence.
    
    Args:
        sequence: List of non-negative real numbers representing step heights
        
    Returns:
        Tuple of (C₁, 1/C₁) where C₁ = 2n * max(convolution) / (sum(sequence))²
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure all values are non-negative and clip to reasonable range
    sequence = [max(0.0, min(x, 1000.0)) for x in sequence]
    
    # Skip if sequence is all zeros
    sum_seq = sum(sequence)
    if sum_seq < 0.01:
        return float('inf'), 0.0
    
    n = len(sequence)
    
    # Use FFT for efficient convolution
    # Convert to numpy array for FFT operations
    seq_array = np.array(sequence)
    
    # Compute convolution using FFT (faster than direct computation)
    # For autoconvolution: a * a
    conv_result = convolve(seq_array, seq_array, mode='full')
    
    # Extract the valid convolution part (the middle portion)
    # For autoconvolution of length n, we get 2n-1 elements
    # The peak occurs at the center
    max_conv = np.max(conv_result)
    
    # Calculate C₁ = 2n * max(convolution) / (sum(sequence))²
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    
    # Return both C₁ and its reciprocal 1/C₁
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    
    return c1, inv_c1

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with specified length."""
    # Generate sequence with some randomness but ensure it's meaningful
    return [random.uniform(0.1, 10.0) for _ in range(length)]

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Create a mutated version of the sequence."""
    mutated = sequence.copy()
    
    # Randomly change some elements
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate by adding/subtracting a small random value
            delta = random.uniform(-1.0, 1.0)
            mutated[i] = max(0.0, mutated[i] + delta)
    
    # Occasionally add/remove elements to explore different sequence lengths
    if random.random() < 0.05 and len(mutated) > 1:
        # Remove an element
        idx = random.randint(0, len(mutated) - 1)
        mutated.pop(idx)
    elif random.random() < 0.05:
        # Add an element
        idx = random.randint(0, len(mutated))
        mutated.insert(idx, random.uniform(0.1, 10.0))
    
    return mutated

def evolutionary_search(max_time_seconds: float = 60.0) -> List[float]:
    """
    Use evolutionary algorithm to search for the best sequence.
    """
    start_time = time.time()
    
    # Initialize population with diverse sequences
    population_size = 50
    population = []
    
    # Create initial diverse sequences
    for _ in range(population_size):
        length = random.randint(10, 500)  # Vary sequence lengths
        individual = generate_random_sequence(length)
        population.append(individual)
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    generation = 0
    while time.time() - start_time < max_time_seconds:
        generation += 1
        
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            c1, inv_c1 = compute_autocorrelation_constant(individual)
            fitness_scores.append((inv_c1, individual))
        
        # Sort by fitness (descending order)
        fitness_scores.sort(reverse=True)
        
        # Update best sequence found so far
        if fitness_scores and fitness_scores[0][0] > best_inv_c1:
            best_inv_c1 = fitness_scores[0][0]
            best_sequence = fitness_scores[0][1].copy()
        
        # Keep top performers
        top_performers = [ind for _, ind in fitness_scores[:population_size//2]]
        
        # Create new population through crossover and mutation
        new_population = top_performers.copy()
        
        # Generate offspring through mutation and crossover
        while len(new_population) < population_size:
            # Select parents
            parent1 = random.choice(top_performers)
            parent2 = random.choice(top_performers)
            
            # Simple crossover: take first half from parent1, second half from parent2
            if len(parent1) > 0 and len(parent2) > 0:
                crossover_point = min(len(parent1), len(parent2)) // 2
                child = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                child = parent1 if len(parent1) > 0 else parent2
                
            # Mutate child
            mutated_child = mutate_sequence(child, mutation_rate=0.2)
            new_population.append(mutated_child)
        
        population = new_population
        
        # Occasionally restart with random sequences to avoid local optima
        if generation % 20 == 0:
            for i in range(0, population_size // 10):
                if random.random() < 0.5:
                    length = random.randint(10, 500)
                    population[random.randint(0, population_size-1)] = generate_random_sequence(length)
    
    return best_sequence if best_sequence is not None else generate_random_sequence(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    # Use evolutionary algorithm to find good sequences
    try:
        sequence = evolutionary_search(max_time_seconds=55.0)  # Leave some buffer time
        
        # Validate and refine the result
        c1, inv_c1 = compute_autocorrelation_constant(sequence)
        
        # If we didn't get a good result, fallback to a simple approach
        if inv_c1 < 0.6:  # If not good enough, try a more systematic approach
            # Try a few specific patterns that often work well
            best_pattern = None
            best_score = 0.0
            
            # Try uniform sequences
            for length in [50, 100, 200, 300]:
                uniform_seq = [1.0] * length
                _, score = compute_autocorrelation_constant(uniform_seq)
                if score > best_score:
                    best_score = score
                    best_pattern = uniform_seq
                    
            # Try sequences with peaks
            for length in [50, 100, 200]:
                peak_seq = [0.0] * length
                if length > 2:
                    peak_seq[length//2] = 10.0
                    peak_seq[length//2 - 1] = 5.0
                    peak_seq[length//2 + 1] = 5.0
                _, score = compute_autocorrelation_constant(peak_seq)
                if score > best_score:
                    best_score = score
                    best_pattern = peak_seq
            
            if best_pattern is not None:
                sequence = best_pattern
        
        return sequence
    except Exception as e:
        # Fallback to simple approach if anything fails
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
