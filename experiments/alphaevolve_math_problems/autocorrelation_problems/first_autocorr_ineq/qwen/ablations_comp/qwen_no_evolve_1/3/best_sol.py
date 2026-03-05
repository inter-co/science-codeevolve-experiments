# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import convolve
from scipy.fft import fft, ifft
import time
from typing import List, Tuple

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the autocorrelation constant C₁ and 1/C₁ for a given sequence.
    Returns (C₁, 1/C₁)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure sequence has positive sum
    total_sum = sum(sequence)
    if total_sum < 0.01:
        return float('inf'), 0.0
    
    # Compute autoconvolution using FFT for efficiency
    # Convert to numpy array
    a = np.array(sequence)
    
    # Autoconvolution: a * a
    # Using FFT-based convolution for O(n log n) complexity
    b = convolve(a, a, mode='full')
    
    # Take only the relevant part (the actual convolution)
    # For autoconvolution, we want indices from 0 to 2*n-2
    # But we're interested in the peak value of the convolution
    max_conv = np.max(b)
    
    # Calculate C₁ = 2n * max(b) / (sum(a))^2
    n = len(sequence)
    c1 = 2 * n * max_conv / (total_sum ** 2)
    
    # Return both C₁ and 1/C₁
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    
    return c1, inv_c1

def generate_random_step_function(n_steps: int) -> List[float]:
    """Generate a random step function with specified number of steps."""
    # Generate random heights for steps
    heights = [random.uniform(0.1, 100.0) for _ in range(n_steps)]
    return heights

def mutate_step_function(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate a step function by randomly changing some heights."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Randomly adjust the height
            change_factor = random.uniform(0.5, 2.0)
            mutated[i] = max(0.01, mutated[i] * change_factor)
    return mutated

def crossover_step_functions(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two step functions."""
    # Simple uniform crossover
    min_len = min(len(seq1), len(seq2))
    crossover_point = random.randint(0, min_len)
    
    child = seq1[:crossover_point] + seq2[crossover_point:]
    
    # Make sure we don't create empty sequences
    if len(child) == 0:
        child = [random.uniform(0.1, 100.0)]
    
    return child

def evolve_step_functions(max_time_seconds: float = 60.0) -> List[float]:
    """
    Evolve step functions using evolutionary algorithm to maximize 1/C₁.
    """
    start_time = time.time()
    
    # Initialize population
    population_size = 50
    population = []
    
    # Generate initial diverse population
    for _ in range(population_size):
        n_steps = random.randint(10, 500)  # Vary sequence lengths
        individual = generate_random_step_function(n_steps)
        population.append(individual)
    
    best_individual = None
    best_inv_c1 = 0.0
    
    generation = 0
    while time.time() - start_time < max_time_seconds:
        generation += 1
        
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
            best_individual = population[max_fitness_idx].copy()
        
        # Create new population through selection, crossover, and mutation
        new_population = []
        
        # Keep top performers (elitism)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_count = population_size // 4
        for i in range(min(elite_count, len(sorted_indices))):
            new_population.append(population[sorted_indices[i]].copy())
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = random.choice(sorted_indices[:population_size//2])
            parent2_idx = random.choice(sorted_indices[:population_size//2])
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child = crossover_step_functions(parent1, parent2)
            
            # Mutation
            child = mutate_step_function(child, mutation_rate=0.2)
            
            # Ensure minimum size
            if len(child) < 2:
                child.extend([random.uniform(0.1, 100.0)] * (2 - len(child)))
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Occasionally introduce diversity by adding random individuals
        if generation % 10 == 0:
            for _ in range(5):
                n_steps = random.randint(10, 500)
                random_individual = generate_random_step_function(n_steps)
                # Replace worst performer
                worst_idx = np.argmin(fitness_scores)
                population[worst_idx] = random_individual
    
    return best_individual if best_individual is not None else generate_random_step_function(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    # Try multiple strategies to find the best solution
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Run evolution multiple times with different seeds
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            sequence = evolve_step_functions(max_time_seconds=10.0)
            _, inv_c1 = compute_autocorrelation_constant(sequence)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = sequence
        except Exception as e:
            continue
    
    # If no good sequence found, return a reasonable default
    if best_sequence is None:
        best_sequence = generate_random_step_function(100)
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
