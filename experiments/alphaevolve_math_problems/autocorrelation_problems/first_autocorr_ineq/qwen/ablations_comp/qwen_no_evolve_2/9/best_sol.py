# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
import random
from typing import List, Tuple
import time

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the C1 constant for a given sequence.
    Returns (C1_value, inv_c1_value)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure we have a valid sequence with sufficient sum
    sum_a = sum(sequence)
    if sum_a < 0.01:
        return float('inf'), 0.0
    
    # Compute autoconvolution (convolution of sequence with itself)
    # Using fast convolution for efficiency
    conv_result = convolve(sequence, sequence, mode='full')
    
    # The convolution result has length 2*n - 1
    # We want the maximum value among the middle elements (the actual convolution)
    max_conv = max(conv_result)
    
    # Compute C1 = 2n * max(b) / (sum(a))^2
    n = len(sequence)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    
    # Return both C1 and its reciprocal
    return c1, 1.0 / c1 if c1 > 0 else 0.0

def evaluate_sequence(sequence: List[float]) -> float:
    """
    Evaluate how good a sequence is for maximizing 1/C1.
    Returns negative of 1/C1 (since we want to maximize 1/C1).
    """
    c1, inv_c1 = compute_c1_constant(sequence)
    if c1 == float('inf'):
        return -1e10  # Very bad score
    return inv_c1

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with specified length."""
    return [random.uniform(0, 100) for _ in range(length)]

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Create a mutated version of the sequence."""
    new_seq = sequence.copy()
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            # Add small random change
            new_seq[i] += random.gauss(0, 0.1 * new_seq[i] if new_seq[i] > 0 else 1.0)
            new_seq[i] = max(0, new_seq[i])  # Ensure non-negative
    return new_seq

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    if len(seq1) != len(seq2):
        # If lengths differ, use the shorter one for crossover
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    crossover_point = random.randint(1, len(seq1) - 1)
    return seq1[:crossover_point] + seq2[crossover_point:]

def evolutionary_search(max_time: float = 60.0) -> List[float]:
    """
    Use evolutionary algorithm to find the best sequence.
    """
    start_time = time.time()
    
    # Initial population
    population_size = 50
    population = []
    
    # Generate initial diverse sequences
    for _ in range(population_size):
        length = random.randint(10, 500)  # Variable sequence lengths
        seq = generate_random_sequence(length)
        population.append(seq)
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    generation = 0
    
    while time.time() - start_time < max_time * 0.95:  # Leave some time for final evaluation
        generation += 1
        
        # Evaluate fitness of all individuals
        fitness_scores = []
        for seq in population:
            fitness = evaluate_sequence(seq)
            fitness_scores.append(fitness)
            
            if fitness > best_inv_c1:
                best_inv_c1 = fitness
                best_sequence = seq.copy()
        
        # Selection: keep top 50% 
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
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
            child = mutate_sequence(child, mutation_rate=0.1)
            
            # Ensure minimum length and valid values
            if len(child) < 5:
                child.extend([random.uniform(0, 100) for _ in range(5 - len(child))])
            elif len(child) > 1000:
                child = child[:1000]
                
            new_population.append(child)
        
        population = new_population[:population_size]
    
    # Final evaluation of best sequence found
    if best_sequence is not None:
        _, final_inv_c1 = compute_c1_constant(best_sequence)
        if final_inv_c1 > best_inv_c1:
            best_inv_c1 = final_inv_c1
    
    return best_sequence if best_sequence is not None else generate_random_sequence(100)

def local_optimization_search(initial_sequence: List[float], max_time: float = 10.0) -> List[float]:
    """
    Apply local optimization to refine the best sequence found.
    """
    start_time = time.time()
    
    # Use scipy's differential evolution for local optimization
    def objective(x):
        # Convert to list with same length as initial sequence
        seq = list(x)
        # Pad or truncate to match initial length
        if len(seq) < len(initial_sequence):
            seq.extend([0.0] * (len(initial_sequence) - len(seq)))
        elif len(seq) > len(initial_sequence):
            seq = seq[:len(initial_sequence)]
        
        # Make sure all values are non-negative
        seq = [max(0, val) for val in seq]
        
        inv_c1 = evaluate_sequence(seq)
        return -inv_c1  # Minimize negative because we want to maximize 1/C1
    
    # Set bounds (0 to 1000 for each element)
    bounds = [(0, 1000) for _ in range(len(initial_sequence))]
    
    try:
        # Use differential evolution which works well for this type of problem
        result = differential_evolution(
            objective, 
            bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            disp=False
        )
        
        # Convert back to sequence
        optimized_seq = list(result.x)
        # Ensure proper length
        if len(optimized_seq) < len(initial_sequence):
            optimized_seq.extend([0.0] * (len(initial_sequence) - len(optimized_seq)))
        elif len(optimized_seq) > len(initial_sequence):
            optimized_seq = optimized_seq[:len(initial_sequence)]
        
        # Ensure non-negative values
        optimized_seq = [max(0, val) for val in optimized_seq]
        
        return optimized_seq
    except Exception:
        # If optimization fails, return the original
        return initial_sequence

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence.
    Uses evolutionary search followed by local optimization.
    """
    # First, do evolutionary search
    evolutionary_sequence = evolutionary_search(max_time=50.0)
    
    # Then refine with local optimization
    refined_sequence = local_optimization_search(evolutionary_sequence, max_time=10.0)
    
    return refined_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
