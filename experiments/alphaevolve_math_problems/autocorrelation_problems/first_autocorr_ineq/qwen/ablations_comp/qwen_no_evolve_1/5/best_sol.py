# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
from scipy.optimize import differential_evolution, minimize
import random
from typing import List
import time

def compute_c1(sequence: List[float]) -> float:
    """
    Compute C1 for a given sequence.
    C1 = 2n * max(convolution) / (sum(sequence))^2
    We want to maximize 1/C1, which means minimize C1.
    """
    if len(sequence) == 0:
        return float('inf')
    
    seq_sum = sum(sequence)
    if seq_sum < 0.01:
        return float('inf')
        
    # Compute convolution using FFT for efficiency
    conv = convolve(sequence, sequence, mode='full')
    max_conv = max(conv)
    
    n = len(sequence)
    if max_conv <= 0:
        return float('inf')
    
    c1 = 2 * n * max_conv / (seq_sum ** 2)
    return c1

def compute_inv_c1(sequence: List[float]) -> float:
    """
    Compute 1/C1 for a given sequence.
    This is what we want to maximize.
    """
    c1 = compute_c1(sequence)
    if c1 <= 0:
        return 0
    return 1.0 / c1

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with specified length."""
    return [random.uniform(0.1, 100.0) for _ in range(length)]

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate a sequence by randomly changing some elements."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = max(0.01, mutated[i] + random.gauss(0, 0.1 * mutated[i]))
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    if len(seq1) != len(seq2):
        # If lengths differ, truncate to shorter length
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    crossover_point = random.randint(1, len(seq1) - 1)
    child = seq1[:crossover_point] + seq2[crossover_point:]
    return child

def evolutionary_search(max_time: float = 50.0) -> List[float]:
    """
    Use evolutionary algorithm to find optimal sequence.
    """
    start_time = time.time()
    
    # Initialize population
    population_size = 50
    population = []
    
    # Generate initial diverse population
    for _ in range(population_size):
        length = random.randint(10, 200)
        individual = generate_random_sequence(length)
        population.append(individual)
    
    best_sequence = None
    best_inv_c1 = 0
    
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            inv_c1 = compute_inv_c1(individual)
            fitness_scores.append(inv_c1)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = individual.copy()
        
        # Selection: keep top 50% based on fitness
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        selected_indices = sorted_indices[:population_size // 2]
        selected_population = [population[i] for i in selected_indices]
        
        # Create new population through crossover and mutation
        new_population = selected_population.copy()
        
        # Add offspring
        while len(new_population) < population_size:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Mutation
            if random.random() < 0.7:  # 70% chance of mutation
                child = mutate_sequence(child)
            
            # Ensure minimum length and non-negative values
            if len(child) == 0:
                child = generate_random_sequence(random.randint(10, 50))
            elif len(child) < 5:
                # Extend short sequences
                child.extend([random.uniform(0.1, 10.0) for _ in range(5 - len(child))])
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Occasionally introduce random diversity
        if generation % 10 == 0:
            for i in range(min(5, len(population))):
                if random.random() < 0.3:
                    population[i] = generate_random_sequence(random.randint(10, 200))
    
    return best_sequence if best_sequence is not None else generate_random_sequence(50)

def gradient_based_refinement(initial_sequence: List[float], max_iter: int = 100) -> List[float]:
    """
    Use gradient-based refinement to fine-tune the sequence.
    """
    def objective(x):
        # Convert to list and ensure non-negativity
        seq = [max(0.01, val) for val in x]
        return -compute_inv_c1(seq)  # Negative because we want to maximize
    
    def constraint_func(x):
        # Constraint: sum must be greater than 0.01
        return sum(x) - 0.01
    
    # Use scipy minimize with bounds
    bounds = [(0.01, 1000.0) for _ in range(len(initial_sequence))]
    
    try:
        result = minimize(objective, 
                         initial_sequence,
                         method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': max_iter})
        
        if result.success:
            refined = [max(0.01, val) for val in result.x]
            return refined
    except:
        pass
    
    return initial_sequence

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence.
    Uses evolutionary search followed by gradient refinement.
    """
    # First, do evolutionary search to find good candidates
    best_sequence = evolutionary_search(max_time=45.0)
    
    # Then refine with gradient-based optimization
    refined_sequence = gradient_based_refinement(best_sequence, max_iter=50)
    
    # Final validation
    final_inv_c1 = compute_inv_c1(refined_sequence)
    if final_inv_c1 < 0.6653:  # Benchmark check
        # Try another approach if needed
        alternative_sequence = generate_random_sequence(100)
        alt_inv_c1 = compute_inv_c1(alternative_sequence)
        if alt_inv_c1 > final_inv_c1:
            refined_sequence = alternative_sequence
    
    return refined_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
