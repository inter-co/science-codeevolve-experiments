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
    
    # Compute convolution using FFT for efficiency
    # Using 'full' mode to get complete convolution
    conv_result = convolve(a, a, mode='full')
    
    # The convolution result has length 2*n - 1
    # We're interested in the maximum value (excluding the zero padding)
    max_conv = np.max(conv_result)
    
    # Sum of sequence
    sum_a = np.sum(a)
    
    # Avoid division by zero
    if sum_a < 1e-10:
        return float('inf'), 0.0
    
    # Compute C₁
    n = len(a)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    
    # Return both C₁ and its reciprocal
    return c1, 1.0 / c1

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with positive values."""
    # Generate random heights in [0.1, 100] range
    return [random.uniform(0.1, 100.0) for _ in range(length)]

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply mutation to a sequence."""
    mutated = sequence.copy()
    
    # Randomly change some elements
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply Gaussian perturbation
            mutated[i] = max(0.01, mutated[i] + random.gauss(0, mutated[i] * 0.1))
    
    # Occasionally add/remove steps
    if random.random() < 0.05 and len(mutated) > 1:
        # Remove a random element
        idx = random.randint(0, len(mutated) - 1)
        mutated.pop(idx)
    elif random.random() < 0.05:
        # Add a random element
        idx = random.randint(0, len(mutated))
        mutated.insert(idx, random.uniform(0.1, 100.0))
    
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if seq1 else seq2
    
    # Simple uniform crossover
    min_len = min(len(seq1), len(seq2))
    crossover_point = random.randint(0, min_len)
    
    child = seq1[:crossover_point] + seq2[crossover_point:]
    
    # Ensure minimum length
    if len(child) < 1:
        child = [random.uniform(0.1, 100.0)]
        
    return child

def evolutionary_search(max_time_seconds: float = 60.0) -> List[float]:
    """
    Evolutionary algorithm to find optimal step function.
    Uses tournament selection, crossover, and mutation.
    """
    start_time = time.time()
    
    # Initial population
    population_size = 50
    population = []
    
    # Generate initial diverse sequences
    for _ in range(population_size):
        length = random.randint(10, 500)
        sequence = generate_random_sequence(length)
        population.append(sequence)
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Evolutionary loop
    generation = 0
    while time.time() - start_time < max_time_seconds:
        generation += 1
        
        # Evaluate fitness (inverse of C₁)
        fitness_scores = []
        for seq in population:
            _, inv_c1 = compute_autocorrelation_constant(seq)
            fitness_scores.append(inv_c1)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_inv_c1:
            best_inv_c1 = fitness_scores[max_fitness_idx]
            best_sequence = population[max_fitness_idx].copy()
        
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
        new_population.append(best_sequence.copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Mutation
            child = mutate_sequence(child)
            
            # Clip values to reasonable range
            child = [max(0.01, min(1000.0, x)) for x in child]
            
            new_population.append(child)
        
        population = new_population
        
        # Occasionally introduce completely new individuals
        if generation % 10 == 0:
            for i in range(0, population_size, 5):
                if len(population) > i:
                    length = random.randint(10, 500)
                    population[i] = generate_random_sequence(length)
    
    return best_sequence if best_sequence is not None else generate_random_sequence(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use evolutionary approach
        sequence = evolutionary_search(max_time_seconds=55.0)
        
        # Validate the sequence
        if len(sequence) == 0:
            sequence = [1.0]
        
        # Ensure minimum sum constraint
        if sum(sequence) < 0.01:
            sequence = [x + 0.01 for x in sequence]
            
        return sequence
    except Exception as e:
        # Fallback to simple approach if something fails
        print(f"Evolutionary search failed: {e}")
        return generate_random_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
