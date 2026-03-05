# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import convolve
from scipy.optimize import differential_evolution, minimize
import time
from numba import jit

@jit(nopython=True)
def compute_convolution_fast(a):
    """Compute convolution of a with itself using fast algorithm"""
    n = len(a)
    # Using the fact that convolution of a with itself has size 2*n-1
    result = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            result[i + j] += a[i] * a[j]
    return result

def compute_c1(sequence):
    """Compute C1 constant for given sequence"""
    if len(sequence) == 0:
        return float('inf')
    
    # Ensure no negative values and clip to reasonable range
    seq = np.clip(sequence, 0, 1000)
    sum_a = np.sum(seq)
    
    if sum_a < 0.01:
        return float('inf')
    
    # Compute convolution (auto-correlation)
    conv = convolve(seq, seq, mode='full')
    
    # Maximum value in convolution
    max_conv = np.max(conv)
    
    # C1 = 2n * max(conv) / (sum(a))^2
    n = len(seq)
    c1 = 2 * n * max_conv / (sum_a ** 2)
    
    return c1

def compute_inv_c1(sequence):
    """Compute 1/C1 for given sequence"""
    c1 = compute_c1(sequence)
    if c1 == float('inf'):
        return 0
    return 1.0 / c1

def generate_random_sequence(length_range=(10, 500)):
    """Generate a random sequence with specified length"""
    n = random.randint(*length_range)
    # Generate sequence with mostly small values but some larger ones
    sequence = [random.uniform(0, 10) for _ in range(n)]
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1):
    """Create a mutated version of the sequence"""
    new_seq = sequence.copy()
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            # Add some noise to the element
            new_seq[i] = max(0, new_seq[i] + random.gauss(0, 0.5))
    return new_seq

def crossover_sequences(seq1, seq2):
    """Create offspring from two parent sequences"""
    min_len = min(len(seq1), len(seq2))
    crossover_point = random.randint(0, min_len)
    
    child = seq1[:crossover_point] + seq2[crossover_point:]
    
    # Make sure child has reasonable length
    if len(child) < 5:
        # Extend with random elements
        child.extend([random.uniform(0, 10) for _ in range(5 - len(child))])
    elif len(child) > 1000:
        # Truncate if too long
        child = child[:1000]
        
    return child

def evolutionary_search(max_time=60):
    """Use evolutionary algorithm to find good sequences"""
    start_time = time.time()
    
    # Initialize population
    population_size = 50
    population = [generate_random_sequence() for _ in range(population_size)]
    
    best_individual = None
    best_fitness = 0
    
    generation = 0
    while time.time() - start_time < max_time - 1:  # Leave 1 second for cleanup
        generation += 1
        
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = compute_inv_c1(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness (descending)
        fitness_scores.sort(reverse=True)
        
        # Update best individual
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_individual = fitness_scores[0][1].copy()
        
        # Select top performers (tournament selection)
        selected = []
        tournament_size = 5
        for _ in range(population_size):
            tournament = random.sample(fitness_scores, tournament_size)
            winner = max(tournament, key=lambda x: x[0])
            selected.append(winner[1])
        
        # Create new population through crossover and mutation
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % population_size]
            
            # Crossover
            child1 = crossover_sequences(parent1, parent2)
            child2 = crossover_sequences(parent2, parent1)
            
            # Mutation
            child1 = mutate_sequence(child1)
            child2 = mutate_sequence(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
    
    # Final evaluation of best individual
    if best_individual is not None:
        final_fitness = compute_inv_c1(best_individual)
        if final_fitness > best_fitness:
            best_fitness = final_fitness
    
    return best_individual if best_individual is not None else generate_random_sequence()

def local_optimization_search(initial_sequence, max_time=30):
    """Use local optimization around promising solutions"""
    start_time = time.time()
    
    def objective(x):
        # Convert to proper sequence format
        seq = np.clip(x, 0, 1000)
        c1 = compute_c1(seq)
        if c1 == float('inf'):
            return float('inf')
        return c1  # We want to minimize C1
    
    # Use differential evolution for global search first
    bounds = [(0, 1000) for _ in range(min(100, len(initial_sequence) + 20))]
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            disp=False
        )
        
        if result.success:
            optimized_seq = np.clip(result.x, 0, 1000)
            return optimized_seq.tolist()
    except:
        pass
    
    # Fallback to simple gradient-based method if needed
    return initial_sequence

def search_for_best_sequence():
    """Main search function using multiple strategies"""
    # Strategy 1: Evolutionary search
    print("Starting evolutionary search...")
    best_sequence = evolutionary_search(max_time=50)
    
    # Strategy 2: Local optimization around best found
    print("Applying local optimization...")
    optimized_sequence = local_optimization_search(best_sequence, max_time=10)
    
    # Final evaluation
    final_fitness = compute_inv_c1(optimized_sequence)
    
    # If we got a better solution, return it
    if final_fitness > compute_inv_c1(best_sequence):
        return optimized_sequence
    else:
        return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
