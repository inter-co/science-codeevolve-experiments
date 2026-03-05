# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy import signal
from scipy.optimize import differential_evolution
import time

def compute_c1(sequence):
    """Compute C1 for a given sequence."""
    if len(sequence) == 0:
        return float('inf')
    
    # Compute convolution using FFT for efficiency
    conv = signal.convolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    
    # Sum of sequence squared
    sum_sq = np.sum(sequence)**2
    
    if sum_sq == 0:
        return float('inf')
    
    # C1 = 2n * max(conv) / (sum(sequence))^2
    n = len(sequence)
    c1 = 2 * n * max_conv / sum_sq
    
    return c1

def compute_inv_c1(sequence):
    """Compute 1/C1 for a given sequence (what we want to maximize)."""
    c1 = compute_c1(sequence)
    if c1 == 0:
        return float('inf')
    return 1.0 / c1

def generate_random_step_function(length=None):
    """Generate a random step function with specified length or random length."""
    if length is None:
        length = random.randint(10, 500)
    
    # Generate random heights between 0 and 1000
    heights = [random.uniform(0, 1000) for _ in range(length)]
    return heights

def mutate_step_function(sequence, mutation_rate=0.1):
    """Mutate a step function by randomly changing some heights."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Change the height by a small amount
            change = random.uniform(-100, 100)
            mutated[i] = max(0, mutated[i] + change)
    return mutated

def crossover_step_functions(seq1, seq2):
    """Perform crossover between two step functions."""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Simple uniform crossover
    min_len = min(len(seq1), len(seq2))
    crossover_point = random.randint(0, min_len)
    
    child = seq1[:crossover_point] + seq2[crossover_point:]
    
    # Ensure we don't create empty sequences
    if len(child) == 0:
        child = [seq1[0]] if seq1 else [seq2[0]]
        
    return child

def evaluate_sequence(sequence):
    """Evaluate a sequence by computing 1/C1."""
    # Filter out invalid sequences
    if len(sequence) == 0 or sum(sequence) < 0.01:
        return 0.0
    
    inv_c1 = compute_inv_c1(sequence)
    return inv_c1

def evolutionary_search(max_time=60):
    """Use evolutionary algorithm to find the best sequence."""
    start_time = time.time()
    
    # Initialize population
    population_size = 50
    population = []
    
    # Create initial diverse population
    for _ in range(population_size):
        seq = generate_random_step_function()
        population.append(seq)
    
    best_score = 0
    best_sequence = None
    
    generation = 0
    while time.time() - start_time < max_time - 1:  # Leave 1 second for finalization
        generation += 1
        
        # Evaluate population
        scores = [evaluate_sequence(seq) for seq in population]
        
        # Track best
        max_score_idx = np.argmax(scores)
        if scores[max_score_idx] > best_score:
            best_score = scores[max_score_idx]
            best_sequence = population[max_score_idx].copy()
        
        # Selection: tournament selection
        selected = []
        for _ in range(population_size):
            tournament_size = 3
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_scores = [scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
            selected.append(population[winner_idx].copy())
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Elitism: keep best individual
        new_population.append(best_sequence.copy() if best_sequence is not None else generate_random_step_function())
        
        while len(new_population) < population_size:
            # Select parents
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Crossover
            child = crossover_step_functions(parent1, parent2)
            
            # Mutation
            child = mutate_step_function(child, mutation_rate=0.2)
            
            # Ensure minimum length and valid values
            if len(child) == 0:
                child = generate_random_step_function()
            
            new_population.append(child)
        
        population = new_population
    
    return best_sequence if best_sequence is not None else generate_random_step_function()

def optimized_search():
    """Run the optimized search with better heuristics."""
    # Try different approaches with different parameters
    best_result = None
    best_inv_c1 = 0
    
    # Run evolutionary search
    try:
        sequence = evolutionary_search(max_time=55)  # Leave 5 seconds for cleanup
        inv_c1 = compute_inv_c1(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_result = sequence
    except Exception as e:
        print(f"Evolutionary search failed: {e}")
        # Fallback to simple random search
        sequence = generate_random_step_function()
        inv_c1 = compute_inv_c1(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_result = sequence
    
    return best_result if best_result is not None else generate_random_step_function()

# Main search function
def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence using evolutionary approach."""
    return optimized_search()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
