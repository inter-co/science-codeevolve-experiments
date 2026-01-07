# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import convolve
from scipy.fft import fft, ifft
import time
from typing import List, Tuple
from numba import jit
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def fast_autoconvolve(a):
    """Fast autoconvolution using FFT - JIT compiled for speed"""
    n = len(a)
    # Use FFT for efficient convolution
    fa = fft(a, 2*n-1)
    result = ifft(fa * fa.conj()).real
    return result[:n]

def compute_c1(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute C1 and 1/C1 for a given sequence.
    Returns (C1, 1/C1)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure sequence is numpy array
    a = np.array(sequence, dtype=np.float64)
    
    # Compute autoconvolution using FFT for efficiency
    try:
        # Use the fast autoconvolve function
        conv_result = fast_autoconvolve(a)
        # The maximum value in the autoconvolution
        max_conv = np.max(conv_result)
    except:
        # Fallback to standard convolution if FFT fails
        conv_result = convolve(a, a, mode='full')[:2*len(a)-1]
        max_conv = np.max(conv_result)
    
    # Sum of sequence
    sum_a = np.sum(a)
    
    if sum_a < 1e-10:
        return float('inf'), 0.0
    
    # Compute C1 = 2n * max(convolution) / (sum(a))^2
    n = len(a)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    
    # Return both C1 and 1/C1
    return c1, 1.0 / c1 if c1 > 0 else 0.0

def evaluate_sequence(sequence: List[float]) -> float:
    """
    Evaluate a sequence by returning 1/C1 (higher is better).
    This is our objective function to maximize.
    """
    _, inv_c1 = compute_c1(sequence)
    return inv_c1

def generate_bell_pattern_sequence(length: int) -> List[float]:
    """Generate a bell-shaped sequence with peak in the middle"""
    sequence = []
    center = length // 2
    for i in range(length):
        # Create a Gaussian-like pattern centered in the middle
        distance_from_center = abs(i - center)
        # Gaussian decay with a slight bump at the center
        value = 100.0 * np.exp(-0.003 * distance_from_center * distance_from_center)
        # Add a peak at center
        if distance_from_center <= 2:
            value += 50.0
        # Add some randomness
        sequence.append(max(0.01, value * random.uniform(0.9, 1.1)))
    return sequence

def generate_spiral_sequence(length: int) -> List[float]:
    """Generate a spiral-like pattern that tends to reduce convolution peaks"""
    sequence = []
    for i in range(length):
        # Create a pattern that oscillates and decays
        oscillation = 1.0 + 0.5 * np.sin(i * 0.2)
        decay = np.exp(-i * 0.01)
        value = 100.0 * oscillation * decay
        sequence.append(max(0.01, min(1000.0, value * random.uniform(0.85, 1.15))))
    return sequence

def generate_peaky_sequence(length: int) -> List[float]:
    """Generate a sequence with high peaks at strategic locations"""
    sequence = []
    # Create peaks at positions 0, length//3, 2*length//3, length-1
    peak_positions = [0, length//3, 2*length//3, length-1]
    for i in range(length):
        value = 0.0
        for pos in peak_positions:
            distance = abs(i - pos)
            # Gaussian peak at each position
            peak_height = 100.0 * np.exp(-0.02 * distance * distance)
            value += peak_height
        sequence.append(max(0.01, min(1000.0, value * random.uniform(0.9, 1.1))))
    return sequence

def generate_balanced_sequence(length: int) -> List[float]:
    """Generate a balanced sequence with moderate variance"""
    sequence = []
    base_val = 50.0
    for i in range(length):
        # Slight variation around base value
        variation = random.gauss(0, 10.0)
        val = base_val + variation
        sequence.append(max(0.01, min(1000.0, val)))
    return sequence

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with specified length."""
    # Use exponential distribution for better spread
    sequence = []
    for i in range(length):
        val = random.expovariate(0.1) * random.choice([0.1, 0.5, 1.0, 2.0, 5.0])
        sequence.append(min(val, 1000.0))
    return sequence

def generate_geometric_sequence(length: int) -> List[float]:
    """Generate a geometric decay sequence"""
    sequence = []
    decay_factor = 0.93  # Slightly faster decay than previous
    for i in range(length):
        value = 100.0 * (decay_factor ** i)
        sequence.append(max(0.01, min(1000.0, value)))
    return sequence

def generate_power_law_sequence(length: int) -> List[float]:
    """Generate a power-law decay sequence"""
    sequence = []
    alpha = 1.5  # Power law exponent
    for i in range(length):
        if i == 0:
            value = 100.0
        else:
            value = 100.0 / (i ** alpha)
        sequence.append(max(0.01, min(1000.0, value)))
    return sequence

def generate_fibonacci_sequence(length: int) -> List[float]:
    """Generate a Fibonacci-like sequence"""
    sequence = []
    if length <= 0:
        return sequence
    elif length == 1:
        sequence.append(100.0)
    elif length == 2:
        sequence.extend([100.0, 100.0])
    else:
        sequence.extend([100.0, 100.0])
        for i in range(2, length):
            next_val = sequence[i-1] + sequence[i-2] * 0.8  # Slight damping
            sequence.append(max(0.01, min(1000.0, next_val)))
    return sequence

def generate_triangular_sequence(length: int) -> List[float]:
    """Generate a triangular sequence"""
    sequence = []
    for i in range(length):
        if i <= length // 2:
            sequence.append(2000.0 * i / length)
        else:
            sequence.append(2000.0 * (length - i) / length)
    return sequence

def generate_gaussian_peak_sequence(length: int) -> List[float]:
    """Generate a sequence with a sharp Gaussian peak in the center"""
    sequence = []
    center = length // 2
    sigma = length / 10.0  # Narrower peak
    for i in range(length):
        value = 1000.0 * np.exp(-((i - center)**2) / (2 * sigma**2))
        sequence.append(value)
    return sequence

def generate_multiscale_sequence(length: int) -> List[float]:
    """Generate a multi-scale pattern with harmonic components"""
    sequence = []
    for i in range(length):
        # Combines multiple frequencies for better autocorrelation properties
        val = 1000.0 * (0.5 + 0.3 * np.sin(2 * np.pi * i / (length/3.0)) + 
                       0.2 * np.sin(2 * np.pi * i / (length/7.0)) + 
                       0.1 * np.sin(2 * np.pi * i / (length/11.0)))
        sequence.append(max(0.1, val))
    return sequence

def generate_sparse_peak_sequence(length: int) -> List[float]:
    """Generate a sequence with sparse, strategically placed high peaks"""
    sequence = [0.0] * length
    # Place peaks sparsely to reduce convolution interference
    num_peaks = max(1, length // 20)  # Fewer peaks for longer sequences
    peak_positions = random.sample(range(length), min(num_peaks, length))
    for pos in peak_positions:
        # Peaks are high but not too concentrated
        sequence[pos] = 1000.0 + random.uniform(0, 500.0)
    # Smooth the transitions to reduce sharp peaks that cause high convolution
    for i in range(length):
        if sequence[i] > 0:
            # Apply smoothing to reduce sharp peaks
            smoothed = sequence[i] * 0.9
            # Apply some randomness to break perfect patterns
            sequence[i] = max(0.01, smoothed * random.uniform(0.95, 1.05))
    return sequence

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.25) -> List[float]:
    """Mutate a sequence by randomly modifying some elements."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply multiplicative mutation with better control
            # Use log-normal distribution for more stable changes
            if mutated[i] > 0.1:
                log_factor = random.gauss(0, 0.3)
                mutation_factor = np.exp(log_factor)
                new_value = mutated[i] * mutation_factor
            else:
                # For very small values, use additive mutation with careful scaling
                new_value = mutated[i] + random.uniform(-0.1, 0.1)
            # Ensure non-negativity and reasonable bounds
            mutated[i] = max(0.01, min(1000.0, new_value))
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> Tuple[List[float], List[float]]:
    """Perform uniform crossover between two sequences"""
    if len(seq1) != len(seq2):
        # If lengths differ, make them the same length by padding or truncating
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    # Blend crossover with weighted averaging for better mixing
    child1, child2 = [], []
    for i in range(len(seq1)):
        # Use blend with different weights to encourage exploration
        weight1 = random.uniform(0.3, 0.7)
        weight2 = 1.0 - weight1
        child1.append(weight1 * seq1[i] + weight2 * seq2[i])
        child2.append(weight2 * seq1[i] + weight1 * seq2[i])
    
    return child1, child2

def adaptive_genetic_algorithm_search(max_time_seconds: float = 85.0) -> List[float]:
    """
    Enhanced genetic algorithm with adaptive parameters and better initialization
    """
    start_time = time.time()
    
    # Adaptive parameters based on time available
    population_size = 180
    generations = int(400 + max_time_seconds * 0.25)
    elite_size = max(5, population_size // 12)
    min_length = 20
    max_length = 1000
    
    # Initialize population with diverse strategies
    population = []
    
    # Add mathematical sequences (more varied)
    for _ in range(15):
        n = random.randint(50, 200)
        individual = generate_gaussian_peak_sequence(n)
        population.append(individual)
    
    for _ in range(15):
        n = random.randint(50, 200)
        individual = generate_multiscale_sequence(n)
        population.append(individual)
    
    for _ in range(15):
        n = random.randint(50, 200)
        individual = generate_sparse_peak_sequence(n)
        population.append(individual)
    
    # Add some known good patterns as starting points
    for _ in range(10):
        # Uniform pattern
        length = random.randint(50, 200)
        individual = [1.0] * length
        population.append(individual)
    
    for _ in range(10):
        # Bell pattern
        length = random.randint(50, 200)
        individual = generate_bell_pattern_sequence(length)
        population.append(individual)
    
    for _ in range(10):
        # Geometric pattern
        length = random.randint(50, 200)
        individual = generate_geometric_sequence(length)
        population.append(individual)
    
    # Add some pattern-based sequences
    for _ in range(25):
        length = random.randint(50, 200)
        individual = generate_power_law_sequence(length)
        population.append(individual)
    
    # Add some Fibonacci sequences
    for _ in range(15):
        length = random.randint(50, 200)
        individual = generate_fibonacci_sequence(length)
        population.append(individual)
    
    # Add triangular patterns
    for _ in range(10):
        length = random.randint(50, 200)
        individual = generate_triangular_sequence(length)
        population.append(individual)
    
    # Add some random sequences
    for _ in range(population_size - 100):
        # Mix of different initialization strategies
        strategy = random.random()
        if strategy < 0.15:
            # Random sequences
            length = random.randint(min_length, max_length)
            individual = generate_random_sequence(length)
        elif strategy < 0.3:
            # Pattern-based sequences
            length = random.randint(min_length, max_length)
            individual = generate_power_law_sequence(length)
        elif strategy < 0.45:
            # Fibonacci sequences
            length = random.randint(min_length, max_length)
            individual = generate_fibonacci_sequence(length)
        elif strategy < 0.6:
            # Geometric sequences
            length = random.randint(min_length, max_length)
            individual = generate_geometric_sequence(length)
        elif strategy < 0.75:
            # Spiral sequences
            length = random.randint(min_length, max_length)
            individual = generate_spiral_sequence(length)
        elif strategy < 0.9:
            # Triangular sequences
            length = random.randint(min_length, max_length)
            individual = generate_triangular_sequence(length)
        else:
            # Sparse peak sequences
            length = random.randint(min_length, max_length)
            individual = generate_sparse_peak_sequence(length)
        population.append(individual)
    
    best_individual = None
    best_fitness = 0.0
    
    # Evolution loop
    for generation in range(generations):
        if time.time() - start_time > max_time_seconds * 0.95:  # Leave some buffer
            break
            
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            fitness = evaluate_sequence(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness (descending)
        fitness_scores.sort(reverse=True)
        
        # Update best individual
        current_best_fitness, current_best_individual = fitness_scores[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()
        
        # Selection using tournament selection
        new_population = []
        
        # Elitism - keep best individuals
        for i in range(elite_size):
            new_population.append(fitness_scores[i][1].copy())
        
        # Generate offspring with adaptive mutation rate
        # Decrease mutation rate over time for exploitation
        mutation_rate = max(0.05, 0.25 * (1 - generation / generations))
        
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = min(8, max(3, population_size // 12))
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i][0] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            
            parent1 = fitness_scores[winner_idx][1]
            
            # Select second parent
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i][0] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            parent2 = fitness_scores[winner_idx][1]
            
            # Crossover
            child1, child2 = crossover_sequences(parent1, parent2)
            
            # Mutate
            child1 = mutate_sequence(child1, mutation_rate)
            child2 = mutate_sequence(child2, mutation_rate)
            
            # Occasionally add new random sequences for diversity
            if random.random() < 0.12:
                length = random.randint(min_length, max_length)
                strategy = random.random()
                if strategy < 0.15:
                    child1 = generate_bell_pattern_sequence(length)
                elif strategy < 0.3:
                    child1 = generate_geometric_sequence(length)
                elif strategy < 0.45:
                    child1 = generate_power_law_sequence(length)
                elif strategy < 0.6:
                    child1 = generate_fibonacci_sequence(length)
                elif strategy < 0.75:
                    child1 = generate_spiral_sequence(length)
                elif strategy < 0.9:
                    child1 = generate_triangular_sequence(length)
                else:
                    child1 = generate_sparse_peak_sequence(length)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
        
        # Add diversity periodically
        if generation % 15 == 0 and len(new_population) > 0:
            # Add some random individuals occasionally
            for _ in range(12):
                length = random.randint(min_length, max_length)
                strategy = random.random()
                if strategy < 0.15:
                    individual = generate_bell_pattern_sequence(length)
                elif strategy < 0.3:
                    individual = generate_geometric_sequence(length)
                elif strategy < 0.45:
                    individual = generate_power_law_sequence(length)
                elif strategy < 0.6:
                    individual = generate_fibonacci_sequence(length)
                elif strategy < 0.75:
                    individual = generate_spiral_sequence(length)
                elif strategy < 0.9:
                    individual = generate_triangular_sequence(length)
                else:
                    individual = generate_sparse_peak_sequence(length)
                if len(population) < population_size:
                    population.append(individual)
    
    # Final validation
    if best_individual is None:
        # Fallback to a good known pattern
        best_individual = [1.0] * 100
    
    return best_individual

def enhanced_local_improvement_search(initial_sequence: List[float], max_time_seconds: float = 90.0) -> List[float]:
    """
    Enhanced local search with multiple improvement strategies and better cooling
    """
    start_time = time.time()
    
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)
    
    best_sequence = current_sequence.copy()
    best_fitness = current_fitness
    
    iteration = 0
    max_iterations = 2000  # Increased iterations for better local search
    
    # Simulated annealing with better temperature schedule
    temp = 1.0
    cooling_rate = 0.999  # Even slower cooling for better exploration
    
    # Track recent improvements to detect stagnation
    recent_improvements = []
    max_stagnation = 60
    
    while iteration < max_iterations and time.time() - start_time < max_time_seconds * 0.95:
        iteration += 1
        
        # Gradually decrease temperature for simulated annealing
        temp = max(0.001, temp * cooling_rate)
        
        # Try different types of mutations
        mutated = False
        
        # Try changing a few elements - adapt number based on sequence length
        num_changes = min(10, max(1, len(current_sequence) // 12))
        for _ in range(num_changes):
            if random.random() < 0.85:  # 85% chance to change an element
                idx = random.randint(0, len(current_sequence) - 1)
                # Try different mutation strategies
                strategy = random.random()
                if strategy < 0.35:
                    # Multiplicative mutation with log-normal
                    if current_sequence[idx] > 0.1:
                        log_factor = random.gauss(0, 0.2)
                        current_sequence[idx] *= np.exp(log_factor)
                    else:
                        current_sequence[idx] += random.uniform(-0.3, 0.3)
                elif strategy < 0.7:
                    # Additive mutation with Gaussian
                    delta = random.gauss(0, 0.1 * current_sequence[idx])
                    current_sequence[idx] += delta
                else:
                    # Large jump mutation with bounded range
                    current_sequence[idx] = random.uniform(0.01, 1000.0)
                current_sequence[idx] = max(0.01, min(1000.0, current_sequence[idx]))
                mutated = True
        
        # If we made changes, evaluate and possibly accept
        if mutated:
            new_fitness = evaluate_sequence(current_sequence)
            delta = new_fitness - current_fitness
            
            if delta > 0:
                # Always accept better solutions
                current_fitness = new_fitness
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_sequence = current_sequence.copy()
                    recent_improvements = []  # Reset stagnation counter
            else:
                # Accept worse solutions with probability based on temperature and delta
                if random.random() < np.exp(delta / max(temp, 1e-10)):
                    current_fitness = new_fitness
                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        best_sequence = current_sequence.copy()
                        recent_improvements = []
            
            # Check for stagnation and take corrective action
            recent_improvements.append(delta)
            if len(recent_improvements) > max_stagnation:
                recent_improvements.pop(0)
                # If no significant improvement in recent iterations, add some noise
                if len(recent_improvements) > 1 and \
                   abs(sum(recent_improvements[-15:])) < 1e-8:
                    # Add small random perturbations to escape local minimum
                    for _ in range(5):
                        idx = random.randint(0, len(current_sequence) - 1)
                        current_sequence[idx] *= random.uniform(0.97, 1.03)
        
        # Occasionally do a complete restart with better pattern
        if random.random() < 0.02:
            # Try different pattern-based restarts
            pattern_type = random.random()
            if pattern_type < 0.15:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_bell_pattern_sequence(new_length)
            elif pattern_type < 0.3:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_geometric_sequence(new_length)
            elif pattern_type < 0.45:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_power_law_sequence(new_length)
            elif pattern_type < 0.6:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_fibonacci_sequence(new_length)
            elif pattern_type < 0.75:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_spiral_sequence(new_length)
            elif pattern_type < 0.9:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_triangular_sequence(new_length)
            else:
                # Restart with sparse peak sequence
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_sparse_peak_sequence(new_length)
            current_fitness = evaluate_sequence(current_sequence)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_sequence = current_sequence.copy()
    
    return best_sequence

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence
    Uses hybrid approach: adaptive genetic algorithm + enhanced local improvement
    """
    # First, run adaptive genetic algorithm to get a good starting point
    ga_solution = adaptive_genetic_algorithm_search(40.0)  # Use 40 seconds for GA
    
    # Then refine with enhanced local improvement
    final_solution = enhanced_local_improvement_search(ga_solution, 40.0)  # Use remaining time for local search
    
    return final_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
