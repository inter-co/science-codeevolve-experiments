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
    """Fast computation of autocorrelation using Numba"""
    n = len(a)
    b = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            b[i+j] += a[i] * a[j]
    return b

def compute_c1_fast(a):
    """Fast computation of C1 value"""
    if len(a) == 0:
        return float('inf')
    
    sum_a = np.sum(a)
    if sum_a < 1e-10:
        return float('inf')
    
    # Use fast autocorrelation computation
    b = compute_autocorrelation_fast(a)
    max_b = np.max(b)
    n = len(a)
    
    # C1 = 2n * max(b) / (sum(a))^2
    c1 = 2 * n * max_b / (sum_a * sum_a)
    return c1

def evaluate_sequence(a):
    """Evaluate a sequence for the inverse C1 metric"""
    if len(a) == 0:
        return 0.0
    
    sum_a = np.sum(a)
    if sum_a < 1e-10:
        return 0.0
    
    # Compute autocorrelation efficiently
    b = compute_autocorrelation_fast(a)
    max_b = np.max(b)
    
    if max_b < 1e-10:
        return 0.0
    
    n = len(a)
    c1 = 2 * n * max_b / (sum_a * sum_a)
    
    # Return inverse of C1 (we want to maximize this)
    return 1.0 / c1 if c1 > 0 else 0.0

def generate_step_function(n_steps, heights=None):
    """Generate a step function with specified number of steps"""
    if heights is None:
        # Generate random heights in [0.1, 10]
        heights = np.random.uniform(0.1, 10, n_steps)
    else:
        heights = np.array(heights[:n_steps])
        if len(heights) < n_steps:
            # Pad with zeros
            heights = np.pad(heights, (0, n_steps - len(heights)), mode='constant')
    
    return heights

def create_mutation_operator():
    """Create a mutation operator for evolutionary search"""
    def mutate_individual(individual, mutation_rate=0.1):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Random change to height
                mutated[i] = max(0.01, mutated[i] + random.gauss(0, 0.5))
        return mutated
    return mutate_individual

def create_crossover_operator():
    """Create a crossover operator for evolutionary search"""
    def crossover(parent1, parent2):
        # Uniform crossover
        child = []
        for i in range(min(len(parent1), len(parent2))):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return np.array(child)
    return crossover

def evolutionary_search(max_time=50.0):
    """Evolutionary algorithm to find optimal step functions"""
    start_time = time.time()
    
    # Initial population parameters
    pop_size = 50
    max_generations = 1000
    min_steps = 10
    max_steps = 1000
    
    # Initialize population
    population = []
    fitness_scores = []
    
    # Create initial diverse population
    for _ in range(pop_size):
        n_steps = random.randint(min_steps, max_steps)
        individual = generate_step_function(n_steps)
        population.append(individual)
    
    best_fitness = 0.0
    best_individual = None
    
    generation = 0
    while time.time() - start_time < max_time and generation < max_generations:
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            fitness = evaluate_sequence(individual)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
        
        # Selection (tournament selection)
        selected_indices = []
        tournament_size = 3
        for _ in range(pop_size):
            tournament_indices = random.sample(range(pop_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices.append(winner_idx)
        
        # Create next generation through crossover and mutation
        new_population = []
        for i in range(0, pop_size, 2):
            parent1_idx = selected_indices[i]
            parent2_idx = selected_indices[(i + 1) % pop_size]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            if random.random() < 0.7:
                # Create offspring with different lengths
                child1 = create_crossover_operator()(parent1, parent2)
                child2 = create_crossover_operator()(parent2, parent1)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            # Mutation
            child1 = create_mutation_operator()(child1)
            child2 = create_mutation_operator()(child2)
            
            # Ensure minimum size
            if len(child1) < min_steps:
                child1 = np.pad(child1, (0, min_steps - len(child1)), mode='constant')
            if len(child2) < min_steps:
                child2 = np.pad(child2, (0, min_steps - len(child2)), mode='constant')
            
            new_population.extend([child1, child2])
        
        # Keep only the population size
        population = new_population[:pop_size]
        generation += 1
    
    return best_individual, best_fitness

def search_for_best_sequence() -> list[float]:
    """Main function to search for the best coefficient sequence"""
    try:
        # Run evolutionary search
        best_sequence, best_fitness = evolutionary_search(max_time=55.0)
        
        # Ensure we have a valid sequence
        if best_sequence is None or len(best_sequence) == 0:
            # Fallback to simple construction
            return [1.0] * 100
        
        # Make sure all elements are non-negative
        best_sequence = np.maximum(best_sequence, 0)
        
        # Normalize to prevent extreme values
        sum_seq = np.sum(best_sequence)
        if sum_seq > 0:
            best_sequence = best_sequence / sum_seq * 100
        
        return best_sequence.tolist()
    
    except Exception as e:
        # Fallback to simple approach
        print(f"Evolutionary search failed: {e}")
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
