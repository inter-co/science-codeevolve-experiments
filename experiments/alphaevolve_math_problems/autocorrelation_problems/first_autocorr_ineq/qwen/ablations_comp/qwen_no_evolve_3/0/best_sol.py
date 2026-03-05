# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
from scipy.optimize import differential_evolution
import random
import time

def compute_c1(sequence):
    """Compute C₁ for a given sequence."""
    if len(sequence) == 0:
        return float('inf')
    
    # Ensure sequence has positive sum
    seq_sum = sum(sequence)
    if seq_sum < 0.01:
        return float('inf')
    
    # Compute autoconvolution
    conv = convolve(sequence, sequence, mode='full')
    # Take only the relevant part (the middle part where convolution is meaningful)
    mid_idx = len(conv) // 2
    max_conv = max(conv[mid_idx - len(sequence) + 1:mid_idx + len(sequence)])
    
    # Compute C₁ = 2n * max(b) / (sum(a))²
    n = len(sequence)
    c1 = 2 * n * max_conv / (seq_sum ** 2)
    
    return c1

def compute_inv_c1(sequence):
    """Compute 1/C₁ for a given sequence."""
    c1 = compute_c1(sequence)
    if c1 == float('inf'):
        return 0
    return 1.0 / c1

def fitness_function(sequence):
    """Fitness function to maximize (1/C₁)."""
    return compute_inv_c1(sequence)

def generate_random_sequence(length_range=(10, 500)):
    """Generate a random valid sequence."""
    n = random.randint(*length_range)
    # Generate sequence with random positive values, but bounded
    sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1):
    """Create a mutated version of the sequence."""
    new_sequence = sequence.copy()
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            # Apply small random perturbation
            new_sequence[i] *= random.uniform(0.8, 1.2)
            # Ensure non-negative
            new_sequence[i] = max(0.0, new_sequence[i])
    return new_sequence

def crossover_sequences(seq1, seq2):
    """Perform crossover between two sequences."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Single point crossover
    crossover_point = random.randint(1, min_len - 1)
    new_seq = seq1[:crossover_point] + seq2[crossover_point:]
    
    # Make sure we don't exceed typical bounds
    if len(new_seq) > 1000:
        new_seq = new_seq[:1000]
    elif len(new_seq) < 1:
        new_seq = [1.0]
        
    return new_seq

def evolutionary_search(max_time_seconds=60):
    """Evolutionary algorithm to find the best sequence."""
    start_time = time.time()
    
    # Initial population
    population_size = 50
    population = [generate_random_sequence() for _ in range(population_size)]
    
    best_fitness = 0
    best_sequence = None
    
    generation = 0
    while time.time() - start_time < max_time_seconds:
        generation += 1
        
        # Evaluate fitness
        fitness_scores = [fitness_function(individual) for individual in population]
        
        # Track best
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_sequence = population[max_fitness_idx].copy()
        
        # Selection - tournament selection
        selected = []
        for _ in range(population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
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
            
            # Ensure minimum length and positive sum
            if len(child) == 0:
                child = [1.0]
            elif sum(child) < 0.01:
                child = [max(0.01, x) for x in child]
                
            new_population.append(child)
        
        population = new_population
        
        # Occasionally introduce completely new individuals
        if generation % 10 == 0:
            for i in range(0, population_size, 5):
                if i < len(population):
                    population[i] = generate_random_sequence()
    
    return best_sequence, best_fitness

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence."""
    try:
        sequence, fitness = evolutionary_search(max_time_seconds=55)
        return sequence
    except Exception as e:
        # Fallback to simple random search if evolution fails
        print(f"Evolutionary search failed: {e}")
        sequence = generate_random_sequence()
        return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
