# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.fft import fft, ifft
import time
from typing import Tuple, List

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Computes the autocorrelation constant C1 and its reciprocal 1/C1.
    Returns (C1, 1/C1) where C1 = 2n * max(convolution) / (sum(sequence))^2
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Convert to numpy array
    a = np.array(sequence)
    n = len(a)
    
    # Compute autoconvolution using FFT for efficiency
    # Pad to 2*n-1 for proper linear convolution
    padded_length = 2 * n - 1
    a_padded = np.pad(a, (0, padded_length - n), mode='constant')
    
    # FFT-based linear convolution (autoconvolution)
    fft_a = fft(a_padded)
    # For autoconvolution, we multiply FFT by itself (not conjugate)
    conv_fft = fft_a * fft_a
    convolution = ifft(conv_fft).real[:padded_length]
    
    # Max value in convolution (valid part)
    max_conv = np.max(convolution)
    
    # Sum of sequence
    sum_a = np.sum(a)
    
    # Avoid division by zero
    if sum_a < 1e-10:
        return float('inf'), 0.0
    
    # Compute C1
    C1 = 2 * n * max_conv / (sum_a ** 2)
    
    # Compute 1/C1
    inv_C1 = 1.0 / C1 if C1 > 0 else 0.0
    
    return C1, inv_C1

def generate_optimized_sequence(n: int) -> List[float]:
    """Generate highly optimized mathematical sequences."""
    # Try multiple patterns and select the best
    patterns = []
    
    # Pattern 1: Modified geometric decay with better spread
    decay_base = 0.85
    pattern1 = [decay_base ** i for i in range(n)]
    total = sum(pattern1)
    if total > 0:
        pattern1 = [val * 1000 / total for val in pattern1]
    patterns.append(pattern1)
    
    # Pattern 2: Gaussian-like peak with better normalization
    center = n // 2
    sigma = n / 6.0
    pattern2 = [np.exp(-((i - center)**2) / (2 * sigma**2)) for i in range(n)]
    total = sum(pattern2)
    if total > 0:
        pattern2 = [val * 1000 / total for val in pattern2]
    patterns.append(pattern2)
    
    # Pattern 3: Multi-scale pattern with alternating high/low
    pattern3 = []
    for i in range(n):
        if i % 4 < 2:
            pattern3.append(2.0)
        else:
            pattern3.append(0.5)
    total = sum(pattern3)
    if total > 0:
        pattern3 = [val * 1000 / total for val in pattern3]
    patterns.append(pattern3)
    
    # Pattern 4: Triangular with peak
    pattern4 = []
    for i in range(n):
        if i <= n//2:
            pattern4.append(i / (n//2))
        else:
            pattern4.append(1.0 - (i - n//2) / (n//2))
    total = sum(pattern4)
    if total > 0:
        pattern4 = [val * 1000 / total for val in pattern4]
    patterns.append(pattern4)
    
    # Pattern 5: Harmonic with modification
    pattern5 = [1.0 / (i + 1) for i in range(n)]
    total = sum(pattern5)
    if total > 0:
        pattern5 = [val * 1000 / total for val in pattern5]
    patterns.append(pattern5)
    
    # Find the best pattern
    best_pattern = patterns[0]
    best_score = 0
    
    for pattern in patterns:
        score = compute_autocorrelation_constant(pattern)[1]  # Get 1/C1
        if score > best_score:
            best_score = score
            best_pattern = pattern
    
    return best_pattern

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.15) -> List[float]:
    """Enhanced mutation with multiple strategies."""
    mutated = sequence.copy()
    
    # Apply mutations to multiple elements
    num_mutations = max(1, int(len(mutated) * mutation_rate))
    
    for _ in range(num_mutations):
        idx = random.randint(0, len(mutated) - 1)
        
        # Choose mutation type based on value magnitude
        if mutated[idx] < 5:
            # Very small values - additive with larger range
            noise = random.uniform(-3, 3)
            mutated[idx] = max(0.01, mutated[idx] + noise)
        elif mutated[idx] < 50:
            # Small-medium values - multiplicative
            scale = random.uniform(0.8, 1.2)
            mutated[idx] = max(0.01, mutated[idx] * scale)
        elif mutated[idx] < 500:
            # Medium values - multiplicative with wider range
            scale = random.uniform(0.7, 1.3)
            mutated[idx] = max(0.01, mutated[idx] * scale)
        else:
            # Large values - additive with smaller range
            noise = random.uniform(-10, 10)
            mutated[idx] = max(0.01, mutated[idx] + noise)
    
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Enhanced crossover with more sophisticated mixing."""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    min_len = min(len(seq1), len(seq2))
    
    # Use multiple crossover points for better mixing
    num_crossover_points = max(2, min_len // 4)
    crossover_points = sorted(random.sample(range(1, min_len), num_crossover_points))
    
    # Create offspring by alternating segments
    child = []
    last_point = 0
    take_from_first = True
    
    for point in crossover_points + [min_len]:
        if take_from_first:
            child.extend(seq1[last_point:point])
        else:
            child.extend(seq2[last_point:point])
        last_point = point
        take_from_first = not take_from_first
    
    # Ensure correct length
    if len(child) < max(len(seq1), len(seq2)):
        # Fill with values from the longer parent or random
        remaining = max(len(seq1), len(seq2)) - len(child)
        for i in range(remaining):
            if len(seq1) > len(child) + i:
                child.append(seq1[len(child) + i])
            elif len(seq2) > len(child) + i:
                child.append(seq2[len(child) + i])
            else:
                child.append(random.uniform(0.01, 1000.0))
    elif len(child) > max(len(seq1), len(seq2)):
        child = child[:max(len(seq1), len(seq2))]
    
    # Ensure bounds
    child = [max(0.01, min(1000.0, val)) for val in child]
    return child

def aggressive_local_search(sequence: List[float], max_iterations: int = 300) -> List[float]:
    """Very aggressive local search with multiple refinement strategies."""
    current = sequence.copy()
    _, current_fitness = compute_autocorrelation_constant(current)
    
    for iteration in range(max_iterations):
        # Strategy 1: Random perturbations
        candidate = current.copy()
        num_changes = max(1, len(candidate) // 10)  # 10% of elements
        
        for _ in range(num_changes):
            idx = random.randint(0, len(candidate) - 1)
            # Adaptive perturbation based on value
            if candidate[idx] < 10:
                noise = random.uniform(-2, 2)
                candidate[idx] = max(0.01, candidate[idx] + noise)
            elif candidate[idx] < 100:
                noise = random.uniform(-5, 5)
                candidate[idx] = max(0.01, candidate[idx] + noise)
            else:
                scale = random.uniform(0.95, 1.05)
                candidate[idx] = max(0.01, candidate[idx] * scale)
        
        _, candidate_fitness = compute_autocorrelation_constant(candidate)
        if candidate_fitness > current_fitness:
            current = candidate
            current_fitness = candidate_fitness
            continue
        
        # Strategy 2: Global scaling
        if random.random() < 0.15:  # 15% chance
            scale_factor = random.uniform(0.9, 1.1)
            scaled = [max(0.01, val * scale_factor) for val in current]
            _, scaled_fitness = compute_autocorrelation_constant(scaled)
            if scaled_fitness > current_fitness:
                current = scaled
                current_fitness = scaled_fitness
                continue
        
        # Strategy 3: Pair swaps
        if len(current) >= 2 and random.random() < 0.08:  # 8% chance
            idx1, idx2 = random.sample(range(len(current)), 2)
            swapped = current.copy()
            swapped[idx1], swapped[idx2] = swapped[idx2], swapped[idx1]
            _, swapped_fitness = compute_autocorrelation_constant(swapped)
            if swapped_fitness > current_fitness:
                current = swapped
                current_fitness = swapped_fitness
                continue
        
        # Strategy 4: Window-based adjustments
        if len(current) >= 5 and random.random() < 0.05:  # 5% chance
            window_start = random.randint(0, len(current) - 5)
            window_end = window_start + 5
            window = current[window_start:window_end]
            # Adjust window by averaging with neighbors
            if window_start > 0 and window_end < len(current):
                avg_neighbor = (current[window_start-1] + current[window_end]) / 2
                # Apply adjustment
                for i in range(window_start, window_end):
                    if current[i] < avg_neighbor:
                        current[i] = min(1000.0, current[i] * 1.1)
                    else:
                        current[i] = max(0.01, current[i] * 0.9)
    
    return current

def comprehensive_search_approach(max_time_seconds: float = 85.0) -> List[float]:
    """
    Comprehensive search approach that combines mathematical construction,
    evolutionary search, and intensive local optimization.
    """
    start_time = time.time()
    
    # Strategy 1: Mathematical construction with multiple patterns
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Test various mathematical patterns extensively
    test_patterns = [
        ("geometric", lambda n: [0.85 ** i for i in range(n)]),
        ("gaussian", lambda n: [np.exp(-((i - n//2)**2) / (2 * (n/6)**2)) for i in range(n)]),
        ("triangular", lambda n: [i/(n//2) if i <= n//2 else 1.0 - (i-n//2)/(n//2) for i in range(n)]),
        ("bimodal", lambda n: [2.0 if i < n//2 else 0.5 for i in range(n)]),
        ("harmonic", lambda n: [1.0/(i+1) for i in range(n)])
    ]
    
    # Test different lengths for each pattern
    lengths_to_test = [30, 50, 75, 100, 150, 200, 300, 400]
    
    for pattern_name, pattern_func in test_patterns:
        for length in lengths_to_test:
            if time.time() - start_time > max_time_seconds * 0.6:
                break
            try:
                # Create pattern
                pattern = pattern_func(length)
                # Normalize
                total = sum(pattern)
                if total > 0:
                    pattern = [val * 1000 / total for val in pattern]
                # Evaluate
                _, inv_c1 = compute_autocorrelation_constant(pattern)
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = pattern.copy()
            except Exception:
                continue
    
    # Strategy 2: Intensive evolutionary search with better parameters
    if time.time() - start_time < max_time_seconds * 0.8:
        # Population-based approach with better parameters
        population_size = 120
        generations = 500
        elite_size = 15
        
        # Initialize with better mathematical patterns
        population = []
        for i in range(population_size):
            length = random.choice(lengths_to_test)
            pattern_name = random.choice(["geometric", "gaussian", "triangular", "bimodal"])
            if pattern_name == "geometric":
                pattern = [0.85 ** i for i in range(length)]
            elif pattern_name == "gaussian":
                pattern = [np.exp(-((i - length//2)**2) / (2 * (length/6)**2)) for i in range(length)]
            elif pattern_name == "triangular":
                pattern = [i/(length//2) if i <= length//2 else 1.0 - (i-length//2)/(length//2) for i in range(length)]
            else:  # bimodal
                pattern = [2.0 if i < length//2 else 0.5 for i in range(length)]
            
            # Normalize
            total = sum(pattern)
            if total > 0:
                pattern = [val * 1000 / total for val in pattern]
            population.append(pattern)
        
        for generation in range(generations):
            if time.time() - start_time > max_time_seconds:
                break
                
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                _, inv_c1 = compute_autocorrelation_constant(individual)
                fitness_scores.append(inv_c1)
            
            # Update best solution
            max_fitness = max(fitness_scores)
            if max_fitness > best_inv_c1:
                best_inv_c1 = max_fitness
                best_idx = fitness_scores.index(max_fitness)
                best_sequence = population[best_idx].copy()
            
            # Create next generation
            new_population = []
            
            # Elitism - keep top performers
            sorted_pairs = sorted(zip(population, fitness_scores), 
                                key=lambda x: x[1], reverse=True)
            elite_individuals = [pair[0] for pair in sorted_pairs[:elite_size]]
            new_population.extend(elite_individuals)
            
            # Generate offspring through crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection with size 5
                tournament_size = 5
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                parent1 = population[winner_idx]
                
                # Second parent
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                parent2 = population[winner_idx]
                
                # Crossover
                child = crossover_sequences(parent1, parent2)
                
                # Mutation with higher rate
                child = mutate_sequence(child, 0.2)  # Higher mutation rate
                
                # Local improvement
                if random.random() < 0.3:
                    child = aggressive_local_search(child, 50)
                
                new_population.append(child)
            
            population = new_population
    
    # Final aggressive refinement
    if best_sequence is not None and time.time() - start_time < max_time_seconds:
        best_sequence = aggressive_local_search(best_sequence, 400)
    
    # Return best found
    if best_sequence is not None:
        return best_sequence
    else:
        # Fallback to a well-tested pattern
        return generate_optimized_sequence(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        sequence = comprehensive_search_approach(85.0)
        _, inv_c1 = compute_autocorrelation_constant(sequence)
        
        # Additional aggressive refinement
        if inv_c1 > 0.5:
            refined_sequence = aggressive_local_search(sequence, 500)
            _, refined_inv_c1 = compute_autocorrelation_constant(refined_sequence)
            if refined_inv_c1 > inv_c1:
                sequence = refined_sequence
                
        return sequence
    except Exception as e:
        print(f"Comprehensive search failed: {e}")
        # Fallback to simple approach
        return generate_optimized_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
