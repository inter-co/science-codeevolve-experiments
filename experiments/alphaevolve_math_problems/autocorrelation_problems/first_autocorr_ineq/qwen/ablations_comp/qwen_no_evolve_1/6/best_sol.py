# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
import random
from typing import List, Tuple
import time
from numba import jit

@jit(nopython=True)
def compute_convolution_fast(a: np.ndarray) -> np.ndarray:
    """Fast computation of autoconvolution using FFT"""
    n = len(a)
    # Use FFT for efficient convolution
    fft_size = 2 * n - 1
    a_padded = np.pad(a, (0, fft_size - n), mode='constant')
    b_padded = np.pad(a, (0, fft_size - n), mode='constant')
    
    # FFT convolution
    a_fft = np.fft.fft(a_padded)
    b_fft = np.fft.fft(b_padded)
    conv_result = np.fft.ifft(a_fft * np.conj(b_fft))
    
    return np.real(conv_result[:fft_size])

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the C1 constant for a given sequence.
    Returns (C1_value, 1/C1_value)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    a = np.array(sequence)
    sum_a = np.sum(a)
    
    if sum_a < 0.01:
        return float('inf'), 0.0
    
    # Compute autoconvolution
    b = convolve(a, a, mode='full')[:len(a)*2-1]
    max_b = np.max(b)
    
    # C1 = 2n * max(b) / (sum(a))^2
    n = len(a)
    c1 = 2 * n * max_b / (sum_a ** 2)
    
    return c1, 1.0 / c1

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random valid sequence"""
    # Generate random sequence with some positive values
    sequence = [random.uniform(0.1, 10.0) for _ in range(length)]
    # Ensure at least one element is significant
    sequence[random.randint(0, length-1)] += random.uniform(5.0, 15.0)
    return sequence

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.3) -> List[float]:
    """Create a mutated version of the sequence"""
    new_sequence = sequence.copy()
    
    # Randomly modify some elements
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            # Add small random perturbation
            new_sequence[i] = max(0.0, new_sequence[i] + random.gauss(0, 0.5))
    
    # Occasionally add/remove elements to explore different sequence lengths
    if random.random() < 0.1 and len(new_sequence) > 1:
        # Remove an element
        idx = random.randint(0, len(new_sequence)-1)
        new_sequence.pop(idx)
    elif random.random() < 0.1 and len(new_sequence) < 1000:
        # Add an element
        idx = random.randint(0, len(new_sequence))
        new_sequence.insert(idx, random.uniform(0.1, 5.0))
        
    return new_sequence

def create_step_function_from_sequence(sequence: List[float]) -> List[float]:
    """Convert a general sequence to a step function with integer heights"""
    # Normalize to [0,1000] range and round to nearest integer
    max_val = max(sequence) if sequence else 1.0
    if max_val < 1e-10:
        max_val = 1.0
    scaled = [max(0.0, min(1000.0, x/max_val * 1000.0)) for x in sequence]
    return [round(x) for x in scaled]

def evolutionary_search(max_time_seconds: int = 60) -> List[float]:
    """
    Evolutionary algorithm to find optimal sequence for maximizing 1/C1
    """
    start_time = time.time()
    
    # Initialize population with diverse sequences
    population_size = 50
    population = []
    
    # Generate initial diverse population
    for _ in range(population_size):
        length = random.randint(10, 500)
        seq = generate_random_sequence(length)
        population.append(seq)
    
    best_sequence = None
    best_inv_c1 = 0.0
    generation = 0
    
    while time.time() - start_time < max_time_seconds:
        generation += 1
        
        # Evaluate fitness for all sequences in population
        fitness_scores = []
        for seq in population:
            try:
                c1, inv_c1 = compute_c1_constant(seq)
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = seq.copy()
                fitness_scores.append((seq, inv_c1))
            except:
                fitness_scores.append((seq, 0.0))
        
        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top performers
        top_performers = [seq for seq, score in fitness_scores[:population_size//2]]
        
        # Create new population through crossover and mutation
        new_population = top_performers.copy()
        
        # Generate offspring through mutation
        while len(new_population) < population_size:
            parent = random.choice(top_performers)
            child = mutate_sequence(parent)
            new_population.append(child)
        
        population = new_population
        
        # Occasionally introduce completely new sequences
        if generation % 10 == 0:
            for _ in range(5):
                length = random.randint(10, 500)
                seq = generate_random_sequence(length)
                if random.random() < 0.5:
                    population.append(seq)
    
    return best_sequence if best_sequence is not None else [1.0]

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence"""
    try:
        # Run evolutionary search
        sequence = evolutionary_search(max_time_seconds=55)
        
        # Validate and refine
        if not sequence or len(sequence) == 0:
            sequence = [1.0] * 10
            
        # Ensure minimum sum
        total_sum = sum(sequence)
        if total_sum < 0.01:
            sequence = [x + 0.1 for x in sequence]
            
        return sequence
    except Exception as e:
        # Fallback to simple approach
        print(f"Evolutionary search failed: {e}")
        return [random.uniform(0.1, 10.0) for _ in range(50)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
