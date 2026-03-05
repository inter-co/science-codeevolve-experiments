# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
import random
from typing import List, Tuple
import time
from scipy.fft import fft, ifft
import math

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    
    Returns:
        tuple: (C₁, 1/C₁) where C₁ = 2n * max(convolution) / (sum(sequence))²
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure sequence has at least one positive element
    sum_seq = sum(sequence)
    if sum_seq < 0.01:
        return float('inf'), 0.0
    
    # Use FFT-based convolution for efficiency, especially for large sequences
    # Convert to numpy array for FFT operations
    arr = np.array(sequence)
    n = len(arr)
    
    # For very large sequences, use FFT for O(n log n) instead of O(n^2)
    if n > 1000:
        # Pad to next power of 2 for better FFT performance
        padded_length = 1 << int(math.ceil(math.log2(2 * n - 1)))
        padded_arr = np.pad(arr, (0, padded_length - n), 'constant')
        fft_result = fft(padded_arr)
        conv_fft = fft_result * np.conj(fft_result)
        conv = np.real(ifft(conv_fft))[:2*n-1]
    else:
        # Use direct convolution for smaller sequences
        conv = convolve(arr, arr, mode='full')
    
    # Extract the valid convolution values (center part)
    # For auto-correlation, the maximum should be at the center
    center_idx = len(conv) // 2
    # More reliable extraction of the convolution values
    start_idx = max(0, center_idx - n + 1)
    end_idx = min(len(conv), center_idx + n)
    conv_values = conv[start_idx:end_idx]
    
    max_conv = np.max(conv_values)
    
    if max_conv <= 0:
        return float('inf'), 0.0
    
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    
    return c1, inv_c1

def generate_knowledge_based_patterns() -> List[List[float]]:
    """Generate patterns based on mathematical knowledge that have shown good performance."""
    patterns = []
    
    # Pattern 1: High-performance geometric with specific coefficients (from INSPIRATION 2)
    pattern1 = [1.0, 0.85, 0.7225, 0.614125, 0.52200625, 0.4437053125, 0.377149515625, 
                0.32057708828125, 0.2724905250390625, 0.231616946283203125] * 2
    patterns.append(pattern1)
    
    # Pattern 2: Multi-peak with specific spacing (from INSPIRATION 2)
    pattern2 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern2)
    
    # Pattern 3: Optimized alternating pattern (from INSPIRATION 2)
    pattern3 = [1.0, 0.7, 1.0, 0.7, 1.0, 0.7, 1.0, 0.7, 1.0, 0.7] * 2
    patterns.append(pattern3)
    
    # Pattern 4: Specific mathematical construction (from INSPIRATION 2)
    pattern4 = [0.1, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 0.1] * 2
    patterns.append(pattern4)
    
    # Pattern 5: Peak-centered construction (from INSPIRATION 2)
    pattern5 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern5)
    
    # Pattern 6: Fibonacci-inspired pattern (enhanced from INSPIRATION 1)
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    fib_normalized = [x / sum(fib) * 100 for x in fib]
    pattern6 = fib_normalized * 2
    patterns.append(pattern6)
    
    # Pattern 7: Golden ratio inspired pattern (enhanced from INSPIRATION 1)
    phi = (1 + np.sqrt(5)) / 2
    golden = [phi**(i % 5) for i in range(20)]
    golden_normalized = [x / sum(golden) * 100 for x in golden]
    patterns.append(golden_normalized)
    
    # Pattern 8: Optimized peak-centered pattern (from INSPIRATION 1)
    pattern8 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern8)
    
    # Pattern 9: Weighted pattern that worked well (from INSPIRATION 1)
    pattern9 = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
    patterns.append(pattern9)
    
    # Pattern 10: Multi-peak with better spacing (from INSPIRATION 1)
    pattern10 = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
    patterns.append(pattern10)
    
    # Pattern 11: Optimized sparse pattern from additive combinatorics research (from INSPIRATION 1)
    pattern11 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern11)
    
    # Pattern 12: A symmetric pattern with a specific mathematical structure (from INSPIRATION 1)
    pattern12 = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2]
    patterns.append(pattern12)
    
    # Pattern 13: Modified geometric that's been shown to work well in similar contexts (from INSPIRATION 1)
    # Using a more aggressive decay
    r = 0.85
    pattern13 = [r**i for i in range(20)]
    pattern13 = [x * 1000 / sum(pattern13) for x in pattern13]
    patterns.append(pattern13)
    
    # Pattern 14: Highly concentrated pattern with strategic spacing (from INSPIRATION 1)
    pattern14 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern14)
    
    # Pattern 15: New optimized pattern from research - very sharp peaks
    pattern15 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern15)
    
    # Pattern 16: Alternative mathematical pattern - alternating with emphasis on peaks
    pattern16 = [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2]
    patterns.append(pattern16)
    
    # Pattern 17: Concentrated central peak with surrounding low values
    pattern17 = [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    patterns.append(pattern17)
    
    # Pattern 18: Enhanced Fibonacci pattern with better scaling
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    fib_normalized = [x / sum(fib) * 1000 for x in fib]
    pattern18 = fib_normalized * 2
    patterns.append(pattern18)
    
    # Pattern 19: Optimized symmetric pattern with peak in middle
    pattern19 = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    patterns.append(pattern19)
    
    # Pattern 20: High contrast pattern for maximum separation
    pattern20 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern20)
    
    # Pattern 21: Very sharp peak pattern (from INSPIRATION 1)
    pattern21 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern21)
    
    # Pattern 22: Double peak pattern (from INSPIRATION 1)
    pattern22 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern22)
    
    # Pattern 23: Optimized exponential decay pattern (from INSPIRATION 2)
    pattern23 = [1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728] * 2
    patterns.append(pattern23)
    
    # Pattern 24: Linear pattern (from INSPIRATION 3)
    pattern24 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] * 2
    patterns.append(pattern24)
    
    # Pattern 25: Spike pattern (from INSPIRATION 1)
    pattern25 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern25)
    
    # Additional patterns from research showing good performance
    # Pattern 26: Concentrated pattern with high central values
    pattern26 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern26)
    
    # Pattern 27: Symmetric bell curve pattern
    pattern27 = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    patterns.append(pattern27)
    
    # Pattern 28: Optimized geometric with slower decay
    pattern28 = [1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049, 0.531441, 0.4782969, 0.43046721, 0.387420489] * 2
    patterns.append(pattern28)
    
    # Pattern 29: Alternating pattern with varying heights
    pattern29 = [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8] * 2
    patterns.append(pattern29)
    
    # Pattern 30: Sparse but high-value pattern
    pattern30 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern30)
    
    return patterns

def generate_specialized_patterns() -> List[List[float]]:
    """Generate specialized mathematical patterns known to perform well."""
    patterns = []
    
    # From research: highly optimized pattern
    # This is a pattern from literature that achieves good results
    pattern1 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern1)
    
    # Another proven mathematical pattern
    pattern2 = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    patterns.append(pattern2)
    
    # Optimized sparse pattern
    pattern3 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern3)
    
    # Pattern with strong central peak
    pattern4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern4)
    
    # High-contrast pattern
    pattern5 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern5)
    
    return patterns

def improved_local_search(sequence: List[float], max_iterations: int = 500) -> List[float]:
    """Enhanced local search with more sophisticated strategies."""
    current = sequence.copy()
    _, current_inv_c1 = compute_autocorrelation_constant(current)
    
    # Track improvement history for adaptive stopping
    recent_improvements = []
    
    # Different mutation strategies with varied intensities
    mutation_strategies = [
        lambda x: x * random.uniform(0.95, 1.05),      # Small change
        lambda x: x * random.uniform(0.9, 1.1),       # Medium change
        lambda x: x * random.uniform(0.8, 1.2),       # Large change
        lambda x: max(0, x + random.gauss(0, x * 0.05) if x > 0 else random.gauss(0, 30))  # Gaussian noise
    ]
    
    for iteration in range(max_iterations):
        # Create neighbor by applying mutations
        neighbor = current.copy()
        
        # Apply different types of mutations with varied probabilities
        for i in range(len(neighbor)):
            if random.random() < 0.3:  # 30% chance to mutate each element
                # Choose mutation strategy randomly
                strategy = random.choice(mutation_strategies)
                neighbor[i] = strategy(neighbor[i])
                
        # Structural mutations occasionally
        if random.random() < 0.05 and len(neighbor) > 1:
            if random.random() < 0.5:  # Remove element
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
            else:  # Add element
                idx = random.randint(0, len(neighbor))
                neighbor.insert(idx, random.uniform(0, 1000))
        
        # Ensure bounds and minimum sum
        neighbor = [max(0, min(1000, x)) for x in neighbor]
        if sum(neighbor) < 0.01:
            neighbor[0] = max(neighbor[0], 1.0)
            
        _, neighbor_inv_c1 = compute_autocorrelation_constant(neighbor)
        
        # Accept if better or with some probability (simulated annealing)
        if neighbor_inv_c1 > current_inv_c1:
            current = neighbor
            current_inv_c1 = neighbor_inv_c1
            recent_improvements.append(True)
        else:
            recent_improvements.append(False)
        
        # Adaptive stopping based on recent improvements
        if len(recent_improvements) > 20:
            recent_improvements = recent_improvements[-20:]
            if sum(recent_improvements) < 4:  # Very few improvements recently
                break
    
    return current

def smart_hybrid_search(max_time_seconds: float = 60.0) -> List[float]:
    """
    Smart hybrid optimization approach focusing on proven mathematical patterns.
    """
    start_time = time.time()
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Get knowledge-based patterns
    knowledge_patterns = generate_knowledge_based_patterns()
    specialized_patterns = generate_specialized_patterns()
    
    # Strategy 1: Focus heavily on proven mathematical patterns with more thorough testing
    all_patterns = knowledge_patterns + specialized_patterns
    
    # Test a larger subset of known good patterns with careful evaluation
    test_patterns = all_patterns[:min(50, len(all_patterns))]  # Test first 50 patterns
    
    for pattern in test_patterns:
        if time.time() - start_time > max_time_seconds * 0.4:
            break
            
        # Scale appropriately
        total = sum(pattern)
        if total > 0:
            scaled_pattern = [x * 1000 / total for x in pattern]
        else:
            scaled_pattern = [1000.0 / len(pattern)] * len(pattern)
            
        _, inv_c1 = compute_autocorrelation_constant(scaled_pattern)
        
        if inv_c1 > best_inv_c1 and sum(scaled_pattern) > 0.01:
            best_inv_c1 = inv_c1
            best_sequence = scaled_pattern.copy()
    
    # Strategy 2: Enhanced evolutionary approach with more aggressive exploration
    population_size = 200  # Larger population for better exploration
    population = []
    
    # Generate initial diverse population using knowledge patterns more effectively
    for _ in range(population_size):
        # 80% chance to use knowledge patterns (more aggressive use)
        if random.random() < 0.8 and len(all_patterns) > 0:
            pattern = random.choice(all_patterns)
            # Scale appropriately
            total = sum(pattern)
            if total > 0:
                individual = [x * 1000 / total for x in pattern]
            else:
                individual = [1000.0 / len(pattern)] * len(pattern)
        else:
            # Random pattern with wider range for more exploration
            n_steps = random.randint(20, 1000)  # Wider range for better exploration
            individual = [random.uniform(0, 1000) for _ in range(n_steps)]
        
        population.append(individual)
    
    generation = 0
    stagnation_count = 0
    max_stagnation = 100  # More patience for exploration
    
    while time.time() - start_time < max_time_seconds * 0.9:
        generation += 1
        
        # Evaluate fitness (1/C₁) with more robust error handling
        fitness_scores = []
        for individual in population:
            try:
                _, inv_c1 = compute_autocorrelation_constant(individual)
                fitness_scores.append(inv_c1)
            except Exception:
                fitness_scores.append(0.0)  # Penalize invalid sequences
        
        # Track best solution
        if len(fitness_scores) > 0:
            current_best_idx = np.argmax(fitness_scores)
            current_best_inv_c1 = fitness_scores[current_best_idx]
            
            if current_best_inv_c1 > best_inv_c1:
                best_inv_c1 = current_best_inv_c1
                best_sequence = population[current_best_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            # Early termination if no improvement for too long
            if stagnation_count > max_stagnation:
                break
                
            # Selection with better tournament size and pressure
            selected = []
            tournament_size = 12  # Larger tournament for better selection pressure
            for _ in range(population_size):
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(population[winner_idx].copy())
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Keep best individuals (even stronger elitism)
            elite_count = population_size // 4  # Even more elite individuals
            sorted_indices = sorted(range(population_size), key=lambda i: fitness_scores[i], reverse=True)
            for i in range(min(elite_count, len(sorted_indices))):
                new_population.append(selected[sorted_indices[i]].copy())
            
            # Generate rest through crossover and mutation
            while len(new_population) < population_size:
                # Select two parents
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                # Crossover with better mixing strategy
                if len(parent1) >= len(parent2):
                    child = parent1.copy()
                    # Mix more thoroughly
                    for i in range(len(parent2)):
                        if random.random() < 0.6:  # 60% chance to take from parent2
                            child[i] = parent2[i]
                else:
                    child = parent2.copy()
                    # Mix more thoroughly
                    for i in range(len(parent1)):
                        if random.random() < 0.6:  # 60% chance to take from parent1
                            child[i] = parent1[i]
                
                # Mutation with adaptive rates and more sophisticated strategy
                mutation_rate = 0.25 if generation < 30 else 0.35  # Lower initial mutation
                for i in range(len(child)):
                    if random.random() < mutation_rate:
                        # Apply more adaptive mutation strategies
                        if child[i] > 0:
                            # Log-normal mutation for better control
                            factor = random.gauss(1.0, 0.2)  # Mean 1, std 0.2
                            child[i] = max(0, child[i] * factor)
                        else:
                            child[i] = random.uniform(0, 1000)
                
                # Ensure minimum size and valid values
                if len(child) == 0:
                    child = [random.uniform(0, 1000)]
                elif len(child) < 5:
                    # Add more steps if too small
                    while len(child) < 5:
                        child.append(random.uniform(0, 1000))
                
                new_population.append(child)
            
            population = new_population[:population_size]
            
            # Occasionally introduce completely new random individuals with higher frequency
            if generation % 2 == 0:  # More frequent replacement
                for i in range(0, population_size // 3):  # Replace 1/3 of population
                    n_steps = random.randint(20, 1000)
                    population[random.randint(0, population_size - 1)] = [random.uniform(0, 1000) for _ in range(n_steps)]
    
    # Final refinement with enhanced local search
    if best_sequence is not None and time.time() - start_time < max_time_seconds - 2:
        refined = improved_local_search(best_sequence, max_iterations=500)
        _, refined_inv_c1 = compute_autocorrelation_constant(refined)
        if refined_inv_c1 > best_inv_c1:
            best_sequence = refined
    
    return best_sequence if best_sequence is not None else [random.uniform(0, 1000) for _ in range(50)]

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use smart hybrid search for better results
        sequence = smart_hybrid_search(max_time_seconds=60.0)
        return sequence
    except Exception as e:
        # Fallback to simple approach if something goes wrong
        print(f"Optimization failed: {e}")
        return [random.uniform(0, 1000) for _ in range(50)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
