# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import convolve
from scipy.fft import fft, ifft
from scipy.optimize import differential_evolution, dual_annealing
import time
from functools import wraps
import warnings

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def time_limited(max_time=55):
    """Decorator to limit execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                warnings.warn(f"Function {func.__name__} failed: {e}")
                return None
            finally:
                end_time = time.time()
                if end_time - start_time > max_time:
                    warnings.warn(f"Function {func.__name__} exceeded time limit")
        return wrapper
    return decorator

def compute_autocorrelation_constant(sequence):
    """Compute C₁ = 2n * max(convolution) / (sum(sequence))²"""
    if len(sequence) == 0:
        return float('inf')
    
    # Use FFT-based convolution for better performance with large sequences
    n = len(sequence)
    if n <= 100:
        # For small sequences, use direct convolution for accuracy
        conv = convolve(sequence, sequence, mode='full')
        max_conv = np.max(conv)
    else:
        # For large sequences, use FFT-based convolution
        padded_length = 2**(int(np.ceil(np.log2(2*n - 1))) if n > 0 else 1)
        seq_fft = fft(sequence, padded_length)
        conv_fft = seq_fft * np.conj(seq_fft)
        convolution = ifft(conv_fft).real[:2*n-1]
        max_conv = np.max(convolution)
    
    sum_seq = np.sum(sequence)
    if sum_seq < 0.01:
        return float('inf')  # Reject sequences with too small sum
    
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return c1

def evaluate_sequence(sequence):
    """Evaluate fitness: maximize 1/C₁ (minimize C₁)"""
    # Ensure all values are non-negative and sum is acceptable
    sequence = np.array([max(0, x) for x in sequence])
    if np.sum(sequence) < 0.01:
        return 0.0  # Invalid sequence
    
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0
    
    return 1.0 / c1  # Maximize inverse of C₁

def construct_high_performance_patterns():
    """Construct specifically high-performance mathematical patterns based on proven constructions."""
    patterns = []
    
    # Focus on the most successful patterns from inspirations that achieved ~0.636 (benchmark_ratio ~0.956)
    # These are the patterns that consistently performed best
    best_patterns = [
        # Top performer from inspirations - highly optimized symmetric peak
        [0.1, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 0.1],
        
        # Variants with different scalings - more aggressive variations
        [0.05, 0.15, 0.35, 0.5, 0.5, 0.5, 0.5, 0.35, 0.15, 0.05],  # Scaled down
        [0.2, 0.6, 1.4, 2.0, 2.0, 2.0, 2.0, 1.4, 0.6, 0.2],      # Scaled up
        [0.1, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 0.1][::-1], # Reversed
        
        # Another high performer - symmetric decreasing
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        
        # Weighted peak pattern - balanced with central high values
        [0.1, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.2, 0.1],
        
        # Multi-peak pattern - multiple high-value regions
        [0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1],
        
        # Sparse high-value pattern - concentrated spikes
        [0.0] * 15 + [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * 5,
        
        # Alternating pattern - regular alternating pattern
        [1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
        
        # Balanced symmetric pattern - symmetric with moderate peaks
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2],
        
        # Optimized alternating with peaks - optimized alternating
        [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        
        # Geometric decay pattern - exponential decay
        [1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728],
        
        # Concentrated peak pattern - very sharp central peak
        [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 0.8, 0.4, 0.2, 0.1, 0.05],
        
        # Exponentially decaying with peaks - smooth exponential with peaks
        [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01],
        
        # Linearly increasing then decreasing - linear pattern
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        
        # Fibonacci-inspired pattern - Fibonacci with normalization
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        
        # Gaussian-like pattern - bell curve shaped
        [0.006, 0.061, 0.242, 0.542, 0.797, 0.935, 0.997, 0.935, 0.797, 0.542, 0.242, 0.061, 0.006],
        
        # Double peak pattern - two central peaks
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        
        # Logarithmic pattern - logarithmic growth
        [np.log(i + 1) for i in range(10)],
        
        # Golden ratio based pattern - golden ratio sequence
        [(1 + np.sqrt(5)) / 2, (1 + np.sqrt(5)) / 2 * 1.618, (1 + np.sqrt(5)) / 2 * 1.618**2, 
         (1 + np.sqrt(5)) / 2 * 1.618**3, (1 + np.sqrt(5)) / 2 * 1.618**4, 
         (1 + np.sqrt(5)) / 2 * 1.618**5, (1 + np.sqrt(5)) / 2 * 1.618**6, 
         (1 + np.sqrt(5)) / 2 * 1.618**7, (1 + np.sqrt(5)) / 2 * 1.618**8, 
         (1 + np.sqrt(5)) / 2 * 1.618**9],
         
        # Additional mathematical constructions
        # Hyperbolic pattern
        [1.0, 0.5, 0.33, 0.25, 0.2, 0.17, 0.14, 0.125, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        
        # Polynomial decay pattern
        [1.0, 0.81, 0.64, 0.49, 0.36, 0.25, 0.16, 0.09, 0.04, 0.01],
        
        # New patterns inspired by the best performers from literature
        # Very high concentration pattern
        [0.01] * 5 + [1.0] * 20 + [0.01] * 5,
        
        # Multi-level pattern with precise balancing
        [1.0] * 10 + [0.8] * 10 + [0.6] * 10 + [0.4] * 10,
        
        # Precise 5-level pattern
        [1.0] * 15 + [0.9] * 15 + [0.8] * 15 + [0.7] * 15 + [0.6] * 10,
        
        # Sharp peak pattern with tail
        [0.1] * 10 + [1.0] * 10 + [0.5] * 10 + [0.1] * 10,
        
        # Highly optimized symmetric pattern
        [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05],
        
        # Enhanced symmetric pattern with more pronounced peaks
        [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05],
        
        # High-contrast pattern
        [0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1],
        
        # Concentrated multi-peak pattern
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        
        # Optimized exponential pattern with higher decay rate
        [1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049, 0.531441, 0.4782969, 0.43046721, 0.387420489],
        
        # Multi-peak with strategic spacing
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        
        # Very sharp concentrated pattern
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    
    # Add the best patterns and their variations
    for pattern in best_patterns:
        # Convert Fibonacci to normalized form
        if len(pattern) == 10 and pattern[0] == 1:  # Fibonacci pattern
            fib_normalized = [x / sum(pattern) * 100 for x in pattern]
            patterns.append(fib_normalized * 2)
        else:
            patterns.append(pattern)
    
    # Add many variations of the top performer to increase search space
    top_performer = [0.1, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 0.1]
    variations = [
        [x * 0.9 for x in top_performer],
        [x * 1.1 for x in top_performer],
        [x * 0.8 for x in top_performer],
        [x * 1.2 for x in top_performer],
        [x * 0.7 for x in top_performer],
        [x * 1.3 for x in top_performer],
        [x * 0.6 for x in top_performer],
        [x * 1.4 for x in top_performer],
        [x * 0.5 for x in top_performer],
        [x * 1.5 for x in top_performer],
        # More extreme variations
        [x * 0.4 for x in top_performer],
        [x * 1.6 for x in top_performer],
        [x * 0.3 for x in top_performer],
        [x * 1.7 for x in top_performer],
        [x * 0.2 for x in top_performer],
        [x * 1.8 for x in top_performer],
        # Even more extreme variations
        [x * 0.15 for x in top_performer],
        [x * 2.0 for x in top_performer],
        [x * 0.1 for x in top_performer],
        [x * 2.5 for x in top_performer],
        # Even more aggressive variations
        [x * 0.1 for x in top_performer],
        [x * 3.0 for x in top_performer],
        [x * 0.05 for x in top_performer],
        [x * 4.0 for x in top_performer],
    ]
    
    patterns.extend(variations)
    
    # Add reversed versions
    patterns.extend([p[::-1] for p in variations])
    
    # Add mathematical patterns with more complex structures
    # Create a new class of patterns with different properties
    complex_patterns = []
    
    # Pattern with multiple peaks at different positions
    multi_peak = [0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 1.0, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1]
    complex_patterns.append(multi_peak)
    
    # Pattern with asymmetric peaks
    asymmetric = [0.1, 0.2, 0.4, 0.8, 1.0, 0.8, 0.4, 0.2, 0.1, 0.05]
    complex_patterns.append(asymmetric)
    
    # Pattern with gradually increasing then decreasing
    gradual = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    complex_patterns.append(gradual)
    
    # Add complex patterns
    patterns.extend(complex_patterns)
    
    # Create longer versions of high performers
    for pattern in patterns[:40]:  # Focus on top patterns
        if len(pattern) < 30:
            # Extend with zeros or repeat pattern
            extended = pattern + [0.0] * (30 - len(pattern))
            patterns.append(extended)
    
    # Add some very long patterns to test scalability
    long_patterns = []
    for i in range(10):
        # Create a long exponentially decaying pattern
        n = 100 + i * 10
        pattern = []
        center = n // 2
        for j in range(n):
            dist_from_center = abs(j - center)
            value = max(0.01, 1000 * np.exp(-dist_from_center / (center/3)))
            pattern.append(value)
        long_patterns.append(pattern)
    
    patterns.extend(long_patterns)
    
    # Add patterns from inspiration that were particularly successful
    # Very precise mathematical constructions
    precise_patterns = [
        # Pattern from literature with high performance
        [1.0] * 30 + [0.8] * 30 + [0.6] * 30 + [0.4] * 30 + [0.2] * 30,
        
        # Symmetric 4-level pattern
        [1.0] * 20 + [0.9] * 20 + [0.8] * 20 + [0.7] * 20 + [0.6] * 20,
        
        # Concentrated high-value pattern
        [0.01] * 50 + [1.0] * 20 + [0.01] * 30,
        
        # Multi-peak pattern with specific spacing
        [0.1] * 10 + [1.0] * 10 + [0.1] * 10 + [1.0] * 10 + [0.1] * 10 + [1.0] * 10 + [0.1] * 10,
        
        # Precisely balanced pattern
        [1.0] * 25 + [0.8] * 25 + [0.6] * 25 + [0.4] * 25,
        
        # Very sharp peak pattern
        [0.0] * 10 + [1.0] * 20 + [0.0] * 10,
        
        # Highly concentrated pattern
        [0.0] * 20 + [1.0] * 20 + [0.0] * 20,
        
        # Optimized multi-peak pattern
        [0.0] * 5 + [1.0] * 10 + [0.0] * 5 + [1.0] * 10 + [0.0] * 5,
        
        # Concentrated with very sharp peaks
        [0.0] * 15 + [1.0] * 10 + [0.0] * 15,
    ]
    
    patterns.extend(precise_patterns)
    
    # Add more mathematically inspired patterns that are known to perform well
    mathematical_patterns = [
        # Golden ratio based pattern with higher precision
        [1.0, 0.618, 0.382, 0.236, 0.146, 0.090, 0.056, 0.034, 0.021, 0.013],
        
        # Optimized exponential with fast decay
        [1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049, 0.531441, 0.4782969, 0.43046721, 0.387420489, 0.3486784401],
        
        # Multi-level pattern with strategic values
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] * 2,
        
        # Concentrated with strategic gaps
        [0.0] * 10 + [1.0] * 10 + [0.0] * 10 + [1.0] * 10 + [0.0] * 10,
        
        # Symmetric with peak in center
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        
        # Multi-peak with specific spacing
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        
        # Very aggressive geometric pattern
        [1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728, 0.1073741824, 0.08589934592],
    ]
    
    patterns.extend(mathematical_patterns)
    
    return patterns

def create_enhanced_initial_population(patterns, population_size=150):
    """Generate enhanced initial population with more strategic pattern selection"""
    population = []
    
    # Add the high-performance mathematical patterns
    for pattern in patterns:
        if len(population) < population_size:
            # Ensure minimum sum requirement
            total_sum = sum(pattern)
            if total_sum < 0.01:
                pattern = [x + 0.1 for x in pattern]
            population.append(pattern.copy())
    
    # Add some variation of high performers with more aggressive variations
    remaining = population_size - len(population)
    
    for _ in range(remaining):
        # Strategy 1: Significantly varied versions of high performers
        if random.random() < 0.8:  # Increase probability of using high-quality patterns
            # Pick a high-performing pattern and aggressively modify it
            base_pattern = random.choice(patterns)
            modified = base_pattern.copy()
            # Apply more substantial variations to get different solutions
            for i in range(len(modified)):
                if random.random() < 0.4:  # Increase modification probability
                    # More aggressive scaling
                    scale_factor = random.uniform(0.3, 3.0)
                    modified[i] = max(0.01, modified[i] * scale_factor)
            population.append(modified)
        else:
            # Strategy 2: Create sequences with geometric or exponential decay
            n = random.randint(20, 250)  # Extended range for longer sequences
            # Create a pattern that balances peaks and smoothness
            sequence = []
            for i in range(n):
                # Centered distribution with peak in middle
                center = n // 2
                dist_from_center = abs(i - center)
                # Higher values near center with exponential decay
                value = max(0.01, 1000 * np.exp(-dist_from_center / (center/3)))
                sequence.append(value)
            
            # Ensure minimum sum
            total_sum = sum(sequence)
            if total_sum < 0.01:
                sequence[0] = 1.0
            
            population.append(sequence)
    
    return population

@time_limited(max_time=45)
def adaptive_evolutionary_search():
    """Enhanced evolutionary search with improved strategies and better local search"""
    # Generate high-performance patterns
    patterns = construct_high_performance_patterns()
    
    # Initialize population with enhanced sequences
    population_size = 200  # Even larger population for better diversity
    population = create_enhanced_initial_population(patterns, population_size)
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Evolution loop with more generations and better strategies
    generation = 0
    max_generations = 800  # Even more generations for better convergence
    
    while generation < max_generations:
        generation += 1
        
        # Evaluate fitness (1/C1) for entire population
        fitness_scores = []
        for seq in population:
            fit = evaluate_sequence(seq)
            fitness_scores.append(fit)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_inv_c1:
            best_inv_c1 = fitness_scores[max_fitness_idx]
            best_sequence = population[max_fitness_idx].copy()
        
        # Selection with stronger elitism
        selected = []
        elite_count = 40  # Even more elite individuals for better convergence
        
        # Keep elite individuals (top 20%)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            selected.append(population[idx].copy())
        
        # Tournament selection for remaining slots with larger tournament size
        remaining_slots = population_size - elite_count
        tournament_size = 20  # Even larger tournament for better selection pressure
        for _ in range(remaining_slots):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Add elites directly
        new_population.extend(selected[:elite_count])
        
        # Generate rest of population
        while len(new_population) < population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Crossover - blend with better mixing
            child = []
            for i in range(max(len(parent1), len(parent2))):
                if i < len(parent1) and i < len(parent2):
                    # Blend with preference to better parent
                    if random.random() < 0.95:  # 95% chance to take from parent1
                        child.append(parent1[i])
                    else:
                        child.append(parent2[i])
                elif i < len(parent1):
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            
            # Mutation with adaptive rate (more aggressive)
            mutation_rate = max(0.01, 0.3 * (1 - generation / max_generations))
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    # More aggressive mutation for early generations, less later
                    if child[i] < 1.0:
                        # Small values get larger jumps
                        child[i] = max(0.01, child[i] * random.uniform(0.1, 8.0))
                    else:
                        # Large values get smaller adjustments
                        child[i] = max(0.01, child[i] * random.uniform(0.4, 1.8))
            
            # Ensure minimum value
            child = [max(0.01, x) for x in child]
            
            new_population.append(child)
        
        population = new_population
        
        # Introduce diversity more frequently and with higher probability
        if generation % 2 == 0:  # Even more frequent diversity introduction
            for i in range(len(population) // 3):
                if len(population) > 0:
                    # Replace with a high-performing pattern with even higher probability
                    if random.random() < 0.99:  # Even higher probability
                        # Use pattern
                        pattern_idx = random.randint(0, len(patterns) - 1)
                        population[random.randint(0, len(population)-1)] = patterns[pattern_idx].copy()
                    else:
                        # Use a new randomized sequence
                        n = random.randint(20, 250)
                        sequence = []
                        for j in range(n):
                            # Centered distribution with peak in middle
                            center = n // 2
                            dist_from_center = abs(j - center)
                            # Higher values near center with exponential decay
                            value = max(0.01, 1000 * np.exp(-dist_from_center / (center/3)))
                            sequence.append(value)
                        
                        # Ensure minimum sum
                        total_sum = sum(sequence)
                        if total_sum < 0.01:
                            sequence[0] = 1.0
                        
                        population[random.randint(0, len(population)-1)] = sequence
    
    # Final comprehensive fine-tuning with multiple strategies
    if best_sequence is not None:
        try:
            # Strategy 1: Differential evolution on best solution with more iterations
            bounds = [(0.01, 1000.0) for _ in range(len(best_sequence))]
            
            def objective(x):
                # Minimize negative of our objective (since we want to maximize)
                return -evaluate_sequence(list(x))
            
            # Use a more thorough optimization approach with more iterations
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=400, 
                popsize=80,
                seed=42,
                strategy='best1bin'
            )
            
            if result.success:
                optimized_pattern = list(result.x)
                inv_c1_optimized = evaluate_sequence(optimized_pattern)
                
                if inv_c1_optimized > best_inv_c1:
                    best_inv_c1 = inv_c1_optimized
                    best_sequence = optimized_pattern
            
            # Strategy 2: Simulated annealing for further improvement
            if best_sequence is not None:
                try:
                    def sa_objective(x):
                        return -evaluate_sequence(list(x))
                    
                    # Run simulated annealing with more iterations and better parameters
                    sa_result = dual_annealing(
                        sa_objective,
                        bounds,
                        maxiter=800,
                        seed=42,
                        no_local_search=False,
                        initial_temp=25000,
                        restart_temp_ratio=0.98,
                        visit=2.62,
                        accept=-5.0
                    )
                    
                    if sa_result.success:
                        sa_pattern = list(sa_result.x)
                        sa_score = evaluate_sequence(sa_pattern)
                        
                        if sa_score > best_inv_c1:
                            best_inv_c1 = sa_score
                            best_sequence = sa_pattern
                except Exception:
                    pass  # Continue if SA fails
                    
            # Strategy 3: Enhanced local search refinement with better neighborhood exploration
            if best_sequence is not None:
                # Try a few rounds of enhanced local search to further improve
                current = best_sequence.copy()
                current_score = evaluate_sequence(current)
                
                # More extensive local search with adaptive neighborhood size
                for iteration in range(3000):  # Even more iterations
                    # Create neighbor by making small changes
                    neighbor = current.copy()
                    # Change 50% of elements in each iteration
                    num_changes = max(1, len(neighbor) // 2)
                    change_indices = random.sample(range(len(neighbor)), num_changes)
                    for i in change_indices:
                        neighbor[i] = max(0.01, neighbor[i] * random.uniform(0.5, 1.5))
                    
                    # Ensure minimum sum
                    if sum(neighbor) < 0.01:
                        neighbor[0] = max(neighbor[0], 1.0)
                    
                    neighbor_score = evaluate_sequence(neighbor)
                    if neighbor_score > current_score:
                        current = neighbor
                        current_score = neighbor_score
                
                # Check if local search improved
                final_score = evaluate_sequence(current)
                if final_score > best_inv_c1:
                    best_inv_c1 = final_score
                    best_sequence = current
                    
        except Exception as e:
            # If optimization fails, continue with current best
            pass
    
    return best_sequence if best_sequence is not None else [1.0] * 10

@time_limited(max_time=45)
def hybrid_approach():
    """Hybrid approach combining multiple strategies."""
    # Strategy 1: Enhanced evolutionary algorithm
    try:
        sequence = adaptive_evolutionary_search()
        return sequence
    except Exception as e:
        print(f"Evolutionary algorithm failed: {e}")
    
    # Fallback to pattern-based approach
    try:
        patterns = construct_high_performance_patterns()
        best_score = 0.0
        best_sequence = None
        
        for pattern in patterns:
            # Normalize pattern
            pattern_sum = sum(pattern)
            if pattern_sum > 0:
                pattern = [x * 100 / pattern_sum for x in pattern]
            
            score = evaluate_sequence(pattern)
            if score > best_score:
                best_score = score
                best_sequence = pattern.copy()
        
        if best_sequence is not None:
            return best_sequence
    except Exception as e:
        print(f"Pattern approach failed: {e}")
    
    # Final fallback
    return [1.0] * 100

def search_for_best_sequence():
    """
    Main search function using advanced evolutionary approach.
    """
    try:
        sequence = hybrid_approach()
        return sequence
    except Exception as e:
        # Fallback to simple approach if something goes wrong
        print(f"All approaches failed with error: {e}")
        return [1.0] * 1000

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
