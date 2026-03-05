# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import convolve
from scipy.optimize import differential_evolution
from numba import jit
import time

@jit(nopython=True)
def compute_autocorrelation_fast(a):
    """Compute autocorrelation efficiently using Numba"""
    n = len(a)
    b = np.zeros(2*n - 1)
    
    # Compute convolution manually for speed
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < 2*n - 1:
                b[idx] += a[i] * a[j]
    
    return b

def compute_c1_value(a):
    """Compute C1 value for a given sequence"""
    if len(a) == 0:
        return float('inf')
    
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')
        
    # Using FFT for faster convolution
    b = convolve(a, a, mode='full')[:len(a)*2-1]
    max_b = np.max(b)
    
    # C1 = 2n * max(b) / (sum(a))^2
    n = len(a)
    c1 = (2 * n * max_b) / (sum_a ** 2)
    
    # We want to maximize 1/C1, so we return the inverse
    return 1.0 / c1 if c1 > 0 else 0

def generate_random_sequence(length_range=(50, 500)):
    """Generate a random valid sequence"""
    n = random.randint(*length_range)
    # Generate sequence with some randomness but keep it reasonable
    sequence = [random.uniform(0.1, 10.0) for _ in range(n)]
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1):
    """Create a mutated version of the sequence"""
    new_seq = sequence.copy()
    n = len(new_seq)
    
    # Randomly modify some elements
    for i in range(n):
        if random.random() < mutation_rate:
            # Apply small random perturbation
            change = random.uniform(-0.5, 0.5)
            new_seq[i] = max(0.0, new_seq[i] + change)
    
    # Occasionally add/remove elements
    if random.random() < 0.1 and n > 10:
        # Remove element
        idx = random.randint(0, n-1)
        new_seq.pop(idx)
    elif random.random() < 0.1 and n < 1000:
        # Add element
        idx = random.randint(0, n)
        new_seq.insert(idx, random.uniform(0.1, 10.0))
    
    return new_seq

def evolutionary_search(max_time=50.0):
    """Use evolutionary algorithm to find good sequences"""
    start_time = time.time()
    
    # Initialize population
    population_size = 50
    population = []
    
    # Generate initial diverse population
    for _ in range(population_size):
        seq = generate_random_sequence()
        if np.sum(seq) > 0.01:  # Ensure valid sum
            population.append(seq)
    
    best_score = 0
    best_sequence = None
    
    # Evolutionary iterations
    generation = 0
    while time.time() - start_time < max_time and generation < 1000:
        generation += 1
        
        # Evaluate fitness for all individuals
        fitness_scores = []
        for seq in population:
            score = compute_c1_value(seq)
            fitness_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_sequence = seq.copy()
        
        # Selection: keep top 30%
        sorted_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], reverse=True)
        elite_count = max(1, population_size // 3)
        elite_indices = sorted_indices[:elite_count]
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Keep elites
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = random.choice(elite_indices)
            parent2_idx = random.choice(elite_indices)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover: combine parts of two sequences
            child = []
            min_len = min(len(parent1), len(parent2))
            
            # Take alternating segments
            for i in range(min_len):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            
            # Add remaining elements from longer parent
            if len(parent1) > min_len:
                child.extend(parent1[min_len:])
            elif len(parent2) > min_len:
                child.extend(parent2[min_len:])
            
            # Mutate
            child = mutate_sequence(child, mutation_rate=0.1)
            
            # Ensure valid sum
            if np.sum(child) > 0.01:
                new_population.append(child)
        
        population = new_population[:population_size]
        
        # Occasionally introduce completely new sequences
        if random.random() < 0.1:
            for _ in range(5):
                new_seq = generate_random_sequence()
                if np.sum(new_seq) > 0.01:
                    # Replace worst individual
                    worst_idx = min(range(len(fitness_scores)), 
                                  key=lambda i: fitness_scores[i])
                    population[worst_idx] = new_seq
    
    return best_sequence if best_sequence is not None else generate_random_sequence()

def search_for_best_sequence() -> list[float]:
    """Main function to search for the best coefficient sequence"""
    try:
        # Use evolutionary approach for better results
        sequence = evolutionary_search(max_time=55.0)
        
        # Final verification and refinement
        if sequence is None or len(sequence) == 0:
            sequence = generate_random_sequence()
            
        # Ensure minimum requirements
        if np.sum(sequence) < 0.01:
            # Add small amount to make it valid
            sequence[0] = max(sequence[0], 0.01)
            
        return sequence
        
    except Exception as e:
        # Fallback to simple random generation
        return generate_random_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
