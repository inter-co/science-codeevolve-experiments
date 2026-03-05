# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
from scipy.fft import fft, ifft
import random
import time
from typing import List, Tuple

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    
    Returns:
        tuple: (C₁, 1/C₁) where C₁ = 2n * max(convolution) / (sum(sequence))²
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Convert to numpy array
    a = np.array(sequence)
    n = len(a)
    
    # Compute convolution using FFT for efficiency
    # Using 'full' mode to get complete convolution
    conv_result = convolve(a, a, mode='full')
    
    # The convolution result has length 2*n - 1
    # We want the maximum value among the middle part (the actual convolution)
    max_conv = np.max(conv_result)
    
    # Sum of the sequence
    sum_a = np.sum(a)
    
    # Avoid division by zero
    if sum_a < 1e-10:
        return float('inf'), 0.0
    
    # Compute C₁
    c1 = 2 * n * max_conv / (sum_a ** 2)
    
    # Return both C₁ and its reciprocal
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    
    return c1, inv_c1

def generate_random_sequence(length: int, min_height: float = 0.0, max_height: float = 1000.0) -> List[float]:
    """Generate a random sequence with specified length and height bounds."""
    return [random.uniform(min_height, max_height) for _ in range(length)]

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1, 
                   max_mutation: float = 100.0) -> List[float]:
    """Create a mutated version of the sequence."""
    new_sequence = sequence.copy()
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            # Apply random mutation
            mutation = random.uniform(-max_mutation, max_mutation)
            new_sequence[i] = max(0.0, new_sequence[i] + mutation)
    return new_sequence

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    if len(seq1) != len(seq2):
        # If lengths differ, take the longer one and pad the shorter one
        if len(seq1) > len(seq2):
            seq2.extend([0.0] * (len(seq1) - len(seq2)))
        else:
            seq1.extend([0.0] * (len(seq2) - len(seq1)))
    
    # Simple uniform crossover
    new_sequence = []
    for i in range(len(seq1)):
        if random.random() < 0.5:
            new_sequence.append(seq1[i])
        else:
            new_sequence.append(seq2[i])
    return new_sequence

def evolutionary_search(max_time_seconds: float = 60.0) -> Tuple[List[float], float, float]:
    """
    Evolve sequences using genetic algorithm approach to maximize 1/C₁.
    
    Returns:
        tuple: (best_sequence, best_C₁, best_inv_c1)
    """
    start_time = time.time()
    
    # Parameters for evolution
    population_size = 50
    generations = 1000
    mutation_rate = 0.1
    elite_size = 5
    tournament_size = 3
    
    # Initialize population
    population = []
    for _ in range(population_size):
        length = random.randint(10, 500)  # Random sequence length
        individual = generate_random_sequence(length)
        population.append(individual)
    
    best_inv_c1 = 0.0
    best_sequence = None
    best_c1 = float('inf')
    
    # Evolution loop
    for generation in range(generations):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Evaluate fitness (1/C₁) for all individuals
        fitness_scores = []
        for individual in population:
            _, inv_c1 = compute_autocorrelation_constant(individual)
            fitness_scores.append(inv_c1)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_inv_c1:
            best_inv_c1 = fitness_scores[max_fitness_idx]
            best_sequence = population[max_fitness_idx].copy()
            _, best_c1 = compute_autocorrelation_constant(best_sequence)
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individuals
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring through selection and crossover
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = tournament_selection(population, fitness_scores, tournament_size)
            parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
            
            # Crossover
            child = crossover_sequences(population[parent1_idx], population[parent2_idx])
            
            # Mutation
            child = mutate_sequence(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population[:population_size]
    
    return best_sequence, best_c1, best_inv_c1

def tournament_selection(population: List[List[float]], fitness_scores: List[float], 
                        tournament_size: int) -> int:
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return winner_idx

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Run evolutionary search
        best_sequence, best_c1, best_inv_c1 = evolutionary_search(max_time_seconds=59.0)
        
        # Ensure we have a valid result
        if best_sequence is None:
            # Fallback to simple approach if evolution fails
            best_sequence = generate_random_sequence(100)
        
        return best_sequence
    except Exception as e:
        # Fallback in case of errors
        print(f"Evolutionary search failed: {e}")
        return generate_random_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
