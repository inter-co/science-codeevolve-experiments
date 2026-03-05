# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy import signal
from scipy.optimize import differential_evolution
import time

def compute_autocorrelation_constant(sequence):
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    C₁ = 2n * max(convolution) / (sum(sequence))²
    We want to maximize 1/C₁, which means minimizing C₁.
    """
    if len(sequence) == 0:
        return float('inf')
    
    seq_sum = sum(sequence)
    if seq_sum < 0.01:
        return float('inf')
        
    # Use FFT for efficient convolution
    conv = signal.convolve(sequence, sequence, mode='full')
    max_conv = max(conv)
    
    n = len(sequence)
    c1 = (2 * n * max_conv) / (seq_sum ** 2)
    
    return c1

def evaluate_fitness(sequence):
    """
    Evaluate fitness as inverse of C₁ (we want to maximize this).
    """
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0  # Invalid sequence gets very low fitness
    return 1.0 / c1

def generate_random_sequence(length_range=(50, 500)):
    """Generate a random valid sequence."""
    n = random.randint(*length_range)
    # Generate sequence with some randomness but ensure it's not all zeros
    sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1):
    """Create a mutated version of the sequence."""
    new_seq = sequence.copy()
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            # Mutate by adding small random change
            new_seq[i] = max(0.0, new_seq[i] + random.gauss(0, 0.5))
    return new_seq

def crossover_sequences(seq1, seq2):
    """Perform crossover between two sequences."""
    if len(seq1) != len(seq2):
        # If lengths differ, truncate to shorter length
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    # Single-point crossover
    point = random.randint(1, len(seq1) - 1)
    new_seq = seq1[:point] + seq2[point:]
    return new_seq

def genetic_algorithm_search(max_time=55.0):
    """Use genetic algorithm to find optimal sequence."""
    start_time = time.time()
    
    # Initialize population
    population_size = 50
    population = [generate_random_sequence() for _ in range(population_size)]
    
    best_fitness = 0.0
    best_sequence = None
    
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        
        # Evaluate fitness for entire population
        fitness_scores = [evaluate_fitness(seq) for seq in population]
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_sequence = population[max_fitness_idx].copy()
        
        # Selection (tournament selection)
        tournament_size = 3
        selected = []
        for _ in range(population_size):
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
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
        
        # Occasionally introduce completely new individuals
        if generation % 10 == 0:
            for i in range(5):  # Add 5 new random individuals
                idx = random.randint(0, population_size - 1)
                population[idx] = generate_random_sequence()
    
    return best_sequence, best_fitness

def gradient_based_optimization(initial_sequence, max_time=55.0):
    """Use gradient-based optimization with constraints."""
    start_time = time.time()
    
    # Convert to numpy array for easier manipulation
    initial_array = np.array(initial_sequence)
    
    # Define objective function to minimize (negative of 1/C₁)
    def objective(x):
        # Ensure all values are non-negative
        x = np.maximum(x, 0.0)
        if np.sum(x) < 0.01:
            return float('inf')
        
        c1 = compute_autocorrelation_constant(x)
        if c1 == float('inf'):
            return float('inf')
        return -1.0 / c1  # Negative because we want to maximize 1/C₁
    
    # Simple gradient descent approach
    learning_rate = 0.01
    current_x = initial_array.copy()
    
    best_x = current_x.copy()
    best_value = objective(current_x)
    
    iterations = 0
    while time.time() - start_time < max_time:
        iterations += 1
        current_value = objective(current_x)
        
        if current_value > best_value:
            best_value = current_value
            best_x = current_x.copy()
        
        # Simple gradient approximation using finite differences
        eps = 1e-6
        grad = np.zeros_like(current_x)
        
        for i in range(len(current_x)):
            x_plus = current_x.copy()
            x_minus = current_x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        
        # Update with gradient ascent (since we're maximizing)
        new_x = current_x + learning_rate * grad
        new_x = np.maximum(new_x, 0.0)  # Keep non-negative
        
        current_x = new_x
        
        # Occasionally perturb to escape local optima
        if iterations % 100 == 0:
            current_x += np.random.normal(0, 0.01, len(current_x))
            current_x = np.maximum(current_x, 0.0)
    
    return best_x.tolist(), best_value

def search_for_best_sequence():
    """Main search function that tries multiple approaches."""
    # Try genetic algorithm first
    try:
        ga_sequence, ga_fitness = genetic_algorithm_search(max_time=50.0)
        if ga_fitness > 0.6:
            return ga_sequence
    except Exception as e:
        pass
    
    # Fall back to gradient-based approach
    try:
        initial_seq = generate_random_sequence()
        grad_sequence, grad_fitness = gradient_based_optimization(initial_seq, max_time=50.0)
        if grad_fitness > 0.6:
            return grad_sequence
    except Exception as e:
        pass
    
    # Last resort: return a simple well-known good sequence
    return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
