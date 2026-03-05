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
    Compute the autocorrelation constant C₁ and its reciprocal 1/C₁
    For a sequence a, C₁ = 2n * max(convolution) / (sum(a))^2
    """
    if len(sequence) == 0:
        return 0.0, 0.0
    
    # Ensure sequence has at least some positive values
    if sum(sequence) < 0.01:
        return 0.0, 0.0
    
    # Using FFT for efficient convolution
    # Convolution using FFT
    n = len(sequence)
    # Pad to avoid circular convolution effects
    padded_length = 2 * n - 1
    seq_padded = np.pad(sequence, (0, padded_length - n), mode='constant')
    
    # Compute convolution using FFT
    conv_result = ifft(fft(seq_padded) * fft(seq_padded.conj())).real[:padded_length]
    
    # Find maximum value in convolution (excluding trivial zero case)
    max_conv = np.max(conv_result)
    
    # Avoid division by zero
    sum_sq = sum(sequence) ** 2
    if sum_sq < 1e-12:
        return 0.0, 0.0
    
    # Compute C₁
    c1 = 2 * n * max_conv / sum_sq
    
    # Return both C₁ and its reciprocal
    return c1, 1.0 / c1 if c1 > 0 else 0.0

def generate_random_step_function(n: int) -> List[float]:
    """Generate a random step function with n steps"""
    # Generate random heights (clipped to [0, 1000])
    heights = [max(0, min(1000, random.gauss(100, 50))) for _ in range(n)]
    # Normalize to avoid extreme values
    total = sum(heights)
    if total > 0:
        heights = [h / total * 100 for h in heights]
    return heights

def generate_fibonacci_like_step_function(n: int) -> List[float]:
    """Generate a step function based on Fibonacci-like pattern"""
    if n <= 0:
        return []
    elif n == 1:
        return [1.0]
    elif n == 2:
        return [1.0, 1.0]
    
    # Generate Fibonacci-like sequence
    heights = [1.0, 1.0]
    for i in range(2, n):
        heights.append(heights[i-1] + heights[i-2])
    
    # Normalize
    total = sum(heights)
    if total > 0:
        heights = [h / total * 100 for h in heights]
    return heights

def generate_geometric_step_function(n: int, ratio: float = 0.7) -> List[float]:
    """Generate a geometric decay step function"""
    if n <= 0:
        return []
    
    heights = [ratio ** i for i in range(n)]
    
    # Normalize
    total = sum(heights)
    if total > 0:
        heights = [h / total * 100 for h in heights]
    return heights

def generate_peak_step_function(n: int, peak_position: int = None, peak_height: float = 100.0) -> List[float]:
    """Generate a step function with a single dominant peak"""
    if n <= 0:
        return []
    
    if peak_position is None:
        peak_position = n // 2
    
    heights = [1.0] * n
    # Make the peak much larger
    heights[peak_position] = peak_height
    
    # Normalize
    total = sum(heights)
    if total > 0:
        heights = [h / total * 100 for h in heights]
    return heights

def generate_multimodal_step_function(n: int) -> List[float]:
    """Generate a multimodal step function with multiple peaks"""
    if n <= 0:
        return []
    
    heights = [0.0] * n
    # Place multiple peaks
    num_peaks = min(3, n // 4)
    peak_positions = [random.randint(0, n-1) for _ in range(num_peaks)]
    
    for pos in peak_positions:
        heights[pos] = random.uniform(50, 100)
    
    # Normalize
    total = sum(heights)
    if total > 0:
        heights = [h / total * 100 for h in heights]
    return heights

def evaluate_population(population: List[List[float]]) -> List[Tuple[List[float], float, float]]:
    """Evaluate all sequences in the population"""
    results = []
    for seq in population:
        try:
            c1, inv_c1 = compute_autocorrelation_constant(seq)
            if c1 > 0:
                results.append((seq, c1, inv_c1))
        except Exception:
            # Skip invalid sequences
            continue
    return results

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate a sequence by randomly modifying elements"""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Randomly change the value with some noise
            mutated[i] = max(0, mutated[i] + random.gauss(0, 0.1 * mutated[i]))
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Create offspring from two parent sequences"""
    if len(seq1) != len(seq2):
        # If lengths differ, make them same length
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    # Simple uniform crossover
    offspring = []
    for i in range(len(seq1)):
        if random.random() < 0.5:
            offspring.append(seq1[i])
        else:
            offspring.append(seq2[i])
    
    # Normalize to keep values reasonable
    total = sum(offspring)
    if total > 0:
        offspring = [x / total * 100 for x in offspring]
    
    return offspring

def search_for_best_sequence() -> List[float]:
    """Main evolutionary search function"""
    start_time = time.time()
    max_time = 55.0  # Leave some buffer for cleanup
    
    # Initialize population
    population_size = 100
    population = []
    
    # Generate diverse initial population
    for i in range(population_size):
        # Mix different generation strategies
        strategy = random.choice([
            generate_random_step_function,
            generate_fibonacci_like_step_function,
            generate_geometric_step_function,
            generate_peak_step_function,
            generate_multimodal_step_function
        ])
        
        n = random.randint(50, 500)  # Variable sequence lengths
        sequence = strategy(n)
        if len(sequence) > 0:
            population.append(sequence)
    
    best_inv_c1 = 0.0
    best_sequence = []
    
    # Evolutionary loop
    generations = 0
    while time.time() - start_time < max_time and generations < 1000:
        # Evaluate current population
        evaluated = evaluate_population(population)
        
        if evaluated:
            # Sort by inv_c1 descending
            evaluated.sort(key=lambda x: x[2], reverse=True)
            
            # Update best solution
            current_best_seq, _, current_best_inv_c1 = evaluated[0]
            if current_best_inv_c1 > best_inv_c1:
                best_inv_c1 = current_best_inv_c1
                best_sequence = current_best_seq.copy()
        
        # Create new population through selection, crossover, and mutation
        new_population = []
        
        # Keep top 20% (elitism)
        evaluated.sort(key=lambda x: x[2], reverse=True)
        elite_count = max(1, population_size // 5)
        for i in range(elite_count):
            new_population.append(evaluated[i][0].copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 5
            tournament = random.sample(evaluated, min(tournament_size, len(evaluated)))
            tournament.sort(key=lambda x: x[2], reverse=True)
            parent1 = tournament[0][0]
            parent2 = tournament[min(1, len(tournament)-1)][0]
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Mutation
            child = mutate_sequence(child, mutation_rate=0.1)
            
            # Ensure minimum size and valid values
            if len(child) < 10:
                child.extend([1.0] * (10 - len(child)))
            
            new_population.append(child)
        
        population = new_population
        generations += 1
    
    # Final evaluation of best sequence found
    if best_sequence:
        _, final_inv_c1 = compute_autocorrelation_constant(best_sequence)
        if final_inv_c1 > best_inv_c1:
            best_inv_c1 = final_inv_c1
    
    # Return the best sequence found
    return best_sequence if best_sequence else generate_random_step_function(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
