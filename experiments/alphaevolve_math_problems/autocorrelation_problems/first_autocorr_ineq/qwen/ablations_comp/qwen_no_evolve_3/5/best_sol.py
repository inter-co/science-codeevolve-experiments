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
    Returns (C₁, 1/C₁) where we want to maximize 1/C₁
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Use fast convolution with FFT for efficiency
    conv = signal.convolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    
    sum_seq = sum(sequence)
    if sum_seq < 1e-10:
        return float('inf'), 0.0
    
    n = len(sequence)
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    
    return c1, inv_c1

def generate_step_function(n: int) -> List[float]:
    """Generate a random step function with n steps"""
    # Generate random heights with some structure
    heights = []
    for i in range(n):
        # Use exponential decay pattern to create better distributions
        heights.append(max(0, 1000 * np.exp(-i * 0.1) * random.random()))
    return heights

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Create a mutated version of the sequence"""
    new_sequence = sequence.copy()
    n = len(new_sequence)
    
    # Randomly modify some elements
    for i in range(n):
        if random.random() < mutation_rate:
            # Add small perturbation
            change = random.gauss(0, 0.1 * max(1, new_sequence[i]))
            new_sequence[i] = max(0, new_sequence[i] + change)
    
    # Occasionally add/remove steps
    if random.random() < 0.05 and n > 1:
        # Remove a random element
        idx = random.randint(0, n-1)
        new_sequence.pop(idx)
    elif random.random() < 0.05:
        # Add a new element
        idx = random.randint(0, n)
        new_sequence.insert(idx, random.random() * 100)
        
    return new_sequence

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Create offspring from two parent sequences"""
    n1, n2 = len(seq1), len(seq2)
    min_len = min(n1, n2)
    
    # Take first half from seq1 and second half from seq2
    split_point = min_len // 2
    child = seq1[:split_point] + seq2[split_point:]
    
    # Ensure minimum length
    if len(child) < 2:
        child.extend([random.random() * 100] * (2 - len(child)))
        
    return child

def optimize_with_evolutionary_algorithm(max_time: float = 60.0) -> List[float]:
    """
    Evolutionary algorithm to maximize 1/C₁
    """
    start_time = time.time()
    
    # Initialize population
    population_size = 50
    population = []
    
    # Create initial diverse population
    for _ in range(population_size):
        n = random.randint(50, 500)  # Variable sequence lengths
        individual = generate_step_function(n)
        # Normalize to prevent extreme values
        total = sum(individual)
        if total > 1e-10:
            individual = [x / total * 100 for x in individual]
        population.append(individual)
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    generation = 0
    while time.time() - start_time < max_time * 0.95:  # Leave some time for final processing
        generation += 1
        
        # Evaluate fitness (1/C₁) for entire population
        fitness_scores = []
        for individual in population:
            _, inv_c1 = compute_autocorrelation_constant(individual)
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
            # Select two parents
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Mutation
            child = mutate_sequence(child, mutation_rate=0.1)
            
            # Normalize
            total = sum(child)
            if total > 1e-10:
                child = [x / total * 100 for x in child]
            
            new_population.append(child)
        
        population = new_population
        
        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Gen {generation}: Best 1/C₁ = {best_inv_c1:.6f}")
    
    return best_sequence

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence"""
    try:
        # Use evolutionary algorithm with time constraint
        best_sequence = optimize_with_evolutionary_algorithm(max_time=55.0)
        return best_sequence
    except Exception as e:
        print(f"Error in evolutionary optimization: {e}")
        # Fallback to simple random generation
        return generate_step_function(random.randint(100, 500))

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
