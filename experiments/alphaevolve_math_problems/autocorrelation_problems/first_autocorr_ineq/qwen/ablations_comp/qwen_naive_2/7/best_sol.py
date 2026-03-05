# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.fft import fft, ifft
from scipy.optimize import minimize
import time
from collections import deque
import math
from numba import jit
import warnings

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True)
def fast_convolution_numba(a):
    """Fast convolution using numba for small sequences"""
    n = len(a)
    result = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            result[i + j] += a[i] * a[j]
    return result

def compute_convolution_fft(sequence):
    """Compute convolution using FFT with better numerical stability"""
    n = len(sequence)
    if n == 0:
        return np.array([])
    
    # Use proper padding for linear convolution
    padded_length = 2 * n - 1
    padded_seq = np.pad(sequence, (0, padded_length - n), mode='constant')
    
    # Compute FFT and convolution
    conv_fft = fft(padded_seq)
    # For auto-correlation, we multiply by conjugate
    conv_result = ifft(conv_fft * np.conj(conv_fft)).real[:padded_length]
    
    # Add small epsilon to avoid numerical issues
    conv_result = np.maximum(conv_result, 1e-15)
    
    return conv_result

def compute_autocorrelation_constant_fast(sequence):
    """
    Optimized version using efficient convolution computation
    """
    if len(sequence) == 0:
        return 0.0
    
    seq_sum = sum(sequence)
    if seq_sum < 0.01:
        return 0.0
    
    n = len(sequence)
    
    # For small sequences, use direct computation with numba
    if n <= 50:
        try:
            # Convert to numpy array for faster computation
            arr = np.array(sequence, dtype=np.float64)
            conv_result = fast_convolution_numba(arr)
            max_conv = np.max(conv_result)
        except Exception:
            # Fallback to Python implementation if numba fails
            conv = [0] * (2 * n - 1)
            for i in range(n):
                for j in range(n):
                    conv[i + j] += sequence[i] * sequence[j]
            max_conv = max(conv)
    else:
        # Use FFT for larger sequences with better error handling
        try:
            conv_result = compute_convolution_fft(sequence)
            max_conv = np.max(conv_result)
            
            # Additional numerical stability check
            if np.isnan(max_conv) or np.isinf(max_conv):
                # Fallback to manual calculation
                conv = [0] * (2 * n - 1)
                for i in range(n):
                    for j in range(n):
                        conv[i + j] += sequence[i] * sequence[j]
                max_conv = max(conv)
        except Exception:
            # Fallback for numerical issues
            conv = [0] * (2 * n - 1)
            for i in range(n):
                for j in range(n):
                    conv[i + j] += sequence[i] * sequence[j]
            max_conv = max(conv)
    
    # Calculate C₁
    if max_conv <= 0 or seq_sum <= 0:
        return 0.0
    
    c1 = (2 * n * max_conv) / (seq_sum ** 2)
    
    # Return 1/C₁ (what we want to maximize)
    return 1.0 / c1 if c1 > 0 else 0.0

def compute_autocorrelation_constant(sequence):
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    C₁ = 2n * max(convolution) / (sum(sequence))²
    We want to maximize 1/C₁, which means minimizing C₁.
    """
    return compute_autocorrelation_constant_fast(sequence)

# Simplified version without caching since it's causing overhead
def compute_autocorrelation_constant_simple(sequence):
    """
    Simple and fast version for optimization purposes
    """
    if len(sequence) == 0:
        return 0.0
    
    seq_sum = sum(sequence)
    if seq_sum < 0.01:
        return 0.0
    
    n = len(sequence)
    
    # Use FFT for all sequences (more consistent performance)
    try:
        conv_result = compute_convolution_fft(sequence)
        max_conv = np.max(conv_result)
        
        if np.isnan(max_conv) or np.isinf(max_conv) or max_conv <= 0 or seq_sum <= 0:
            return 0.0
        
        c1 = (2 * n * max_conv) / (seq_sum ** 2)
        return 1.0 / c1 if c1 > 0 else 0.0
    except Exception:
        return 0.0

def generate_mathematically_informed_sequences():
    """Generate sequences based on mathematical insights that tend to work well."""
    sequences = []
    
    # Golden ratio based sequences
    golden_ratio = (1 + np.sqrt(5)) / 2
    for n in [25, 30, 40, 50, 75, 100]:
        # Golden ratio decay
        golden_seq = [1.0 / (golden_ratio ** i) for i in range(n)]
        sequences.append(golden_seq)
        
        # Alternating golden ratios
        alt_golden = [1.0 if i % 2 == 0 else 1.0/golden_ratio for i in range(n)]
        sequences.append(alt_golden)
    
    # Exponential sequences with different bases
    for base in [1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]:
        exp_seq = [1.0 / (base ** i) for i in range(100)]
        sequences.append(exp_seq)
    
    # Concentrated energy sequences (better for minimizing max convolution)
    for n in [40, 50, 75, 100]:
        # Peak at beginning with rapid decay
        peak_seq = [10.0] + [1.0 / (1.5 ** i) for i in range(n-1)]
        sequences.append(peak_seq)
        
        # Two peaks
        two_peak = [0.1] * (n//3) + [5.0] + [0.1] * (n//3) + [10.0] + [0.1] * (n - 2*n//3 - 1)
        sequences.append(two_peak)
    
    # Sequences with strong concentration at start (good for minimizing max convolution)
    for n in [40, 50, 75, 100]:
        # Strong peak followed by exponential decay
        strong_peak = [100.0] + [1.0 / (1.2 ** i) for i in range(n-1)]
        sequences.append(strong_peak)
        
        # Multiple strong peaks
        multi_peak = [0.1] * (n//4) + [50.0] + [0.1] * (n//4) + [25.0] + [0.1] * (n//2)
        sequences.append(multi_peak)
    
    # Mathematical patterns that often work well
    # Harmonic sequences
    harmonic_seq = [1.0 / (i + 1) for i in range(100)]
    sequences.append(harmonic_seq)
    
    # Power law sequences
    power_seq = [1.0 / ((i + 1) ** 1.5) for i in range(100)]
    sequences.append(power_seq)
    
    # Logarithmic sequences
    log_seq = [1.0 / (np.log(i + 2)) for i in range(100)]
    sequences.append(log_seq)
    
    # Fibonacci-like sequences (normalized)
    fib_seq = [1.0]
    for i in range(1, 100):
        fib_seq.append(fib_seq[i-1] + (fib_seq[i-1] if i >= 2 else 1))
    sequences.append([x/1000.0 for x in fib_seq])
    
    # Gaussian-like sequences
    gaussian_seq = [np.exp(-0.5 * ((i - 50) / 15)**2) for i in range(100)]
    sequences.append(gaussian_seq)
    
    # Sequences with strategic concentration
    # High concentration at start with gradual decrease
    high_concentration = [100.0] + [0.1] * 99
    sequences.append(high_concentration)
    
    # Balanced decay with early emphasis
    balanced_decay = [1.0] + [0.8] * 20 + [0.6] * 20 + [0.4] * 20 + [0.2] * 20 + [0.1] * 20
    sequences.append(balanced_decay)
    
    # Special sequences from literature
    # Power-law with different exponents
    for alpha in [1.2, 1.3, 1.4, 1.6, 1.8]:
        power_alpha = [1.0 / ((i + 1) ** alpha) for i in range(100)]
        sequences.append(power_alpha)
    
    # Modified geometric sequences
    for r in [1.1, 1.15, 1.2, 1.25, 1.3]:
        geom_mod = [1.0 / (r ** i) for i in range(100)]
        sequences.append(geom_mod)
    
    return sequences

def generate_special_sequences():
    """Generate specialized sequences based on mathematical insights for better performance."""
    sequences = []
    
    # Known good mathematical patterns
    # Geometric sequences with various ratios
    for ratio in [1.1, 1.2, 1.3, 1.4, 1.5]:
        geometric = [1.0 / (ratio ** i) for i in range(100)]
        sequences.append(geometric)
    
    # Decaying sequences with different decay rates
    for decay in [0.8, 0.9, 0.95]:
        decaying = [1.0 * (decay ** i) for i in range(100)]
        sequences.append(decaying)
    
    # Sequences with strong initial peaks
    peak_seq = [100.0] + [1.0] * 99
    sequences.append(peak_seq)
    
    # Multi-peak structures
    multi_peak = [1.0] * 20 + [50.0] + [1.0] * 20 + [25.0] + [1.0] * 58
    sequences.append(multi_peak)
    
    # Step functions
    step_seq = [1.0] * 50 + [0.5] * 50
    sequences.append(step_seq)
    
    # Alternating high-low patterns
    alt_seq = [10.0 if i % 2 == 0 else 0.5 for i in range(100)]
    sequences.append(alt_seq)
    
    # Smooth decreasing sequences
    smooth_dec = [1.0 / (1.0 + i * 0.02) for i in range(100)]
    sequences.append(smooth_dec)
    
    return sequences

def generate_better_initial_sequences():
    """Generate a diverse set of initial sequences for better exploration."""
    sequences = []
    
    # Add sequences of various lengths and structures
    for n in [10, 20, 30, 50, 75, 100]:
        # Uniform sequences
        sequences.append([1.0] * n)
        
        # Some with varying heights
        seq = [random.uniform(0.1, 10.0) for _ in range(n)]
        sequences.append(seq)
        
        # Step-like sequences (like in known solutions)
        if n >= 5:
            step_seq = [1.0] * (n//2) + [0.5] * (n - n//2)
            sequences.append(step_seq)
            
        # Special structures based on mathematical intuition
        if n >= 10:
            # Geometric decay sequences
            geometric_seq = [1.0 / (1.5 ** i) for i in range(n)]
            sequences.append(geometric_seq)
            
            # Alternating sequences
            alternating_seq = [1.0 if i % 2 == 0 else 0.5 for i in range(n)]
            sequences.append(alternating_seq)
            
            # Peaks at center (more focused energy)
            peak_seq = [0.1] * n
            center = n // 2
            peak_seq[center] = 10.0
            if center + 1 < n:
                peak_seq[center + 1] = 5.0
            if center - 1 >= 0:
                peak_seq[center - 1] = 5.0
            sequences.append(peak_seq)
    
    # Add some high-quality known patterns
    # Pattern that often works well - concentrated at start
    sequences.append([10.0] + [0.1] * 99)
    sequences.append([1.0] * 50 + [0.1] * 50)
    
    # Add sequences that have been shown to work well in similar problems
    # Concentrated peak at beginning with gradual decay
    sequences.append([10.0, 5.0, 2.5, 1.0, 0.5] + [0.1] * 95)
    
    # Double peak structure
    double_peak = [0.1] * 20 + [5.0] + [0.1] * 20 + [10.0] + [0.1] * 58
    sequences.append(double_peak)
    
    # Exponential decay pattern
    exp_decay = [1.0 / (1.3 ** i) for i in range(100)]
    sequences.append(exp_decay)
    
    # Improved patterns based on known good solutions
    # High concentration at start
    high_concentration = [100.0] + [0.1] * 99
    sequences.append(high_concentration)
    
    # Balanced decay
    balanced_decay = [1.0] + [0.8] * 20 + [0.6] * 20 + [0.4] * 20 + [0.2] * 20 + [0.1] * 20
    sequences.append(balanced_decay)
    
    # Add mathematical patterns
    sequences.extend(generate_mathematically_informed_sequences())
    sequences.extend(generate_special_sequences())
    
    return sequences

def mutate_sequence(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Mutate a sequence with adaptive mutation rate and better bounds."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Adaptive mutation: smaller changes for larger values
            change_factor = random.uniform(-max_mutation, max_mutation)
            mutated[i] = max(0.01, mutated[i] * (1 + change_factor))
            # Clip to reasonable bounds
            mutated[i] = min(1000.0, mutated[i])
    return mutated

def advanced_crossover(seq1, seq2):
    """Advanced crossover with more sophisticated mixing strategies."""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Try different crossover strategies
    strategy = random.choice(['uniform', 'segment', 'alternating'])
    
    if strategy == 'uniform':
        # Uniform crossover - randomly select elements from either parent
        child = []
        for i in range(min(len(seq1), len(seq2))):
            if random.random() < 0.5:
                child.append(seq1[i])
            else:
                child.append(seq2[i])
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[len(seq2):])
        elif len(seq2) > len(seq1):
            child.extend(seq2[len(seq1):])
        return child
    
    elif strategy == 'segment':
        # Segment crossover - take segments from each parent
        min_len = min(len(seq1), len(seq2))
        if min_len < 2:
            return seq1 if len(seq1) > 0 else seq2
            
        # Split at random points
        split1 = random.randint(1, min_len - 1)
        split2 = random.randint(1, min_len - 1)
        
        child = []
        if random.random() < 0.5:
            child.extend(seq1[:split1])
            child.extend(seq2[split1:split2])
            child.extend(seq1[split2:])
        else:
            child.extend(seq2[:split1])
            child.extend(seq1[split1:split2])
            child.extend(seq2[split2:])
        
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[len(seq2):])
        elif len(seq2) > len(seq1):
            child.extend(seq2[len(seq1):])
        return child
    
    else:  # alternating
        # Alternating crossover - alternate between elements from parents
        min_len = min(len(seq1), len(seq2))
        child = []
        for i in range(min_len):
            if i % 2 == 0:
                child.append(seq1[i])
            else:
                child.append(seq2[i])
        
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[min_len:])
        elif len(seq2) > len(seq1):
            child.extend(seq2[min_len:])
        return child

def local_search_improvement(sequence, iterations=50):
    """Perform local search to improve the sequence with multiple strategies."""
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Try different local search strategies
    for i in range(iterations):
        # Strategy 1: Single element perturbation
        if random.random() < 0.7:
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            # Small multiplicative change with better bounds
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
            
            new_score = compute_autocorrelation_constant_fast(new_seq)
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
        else:
            # Strategy 2: Multiple element perturbation with adaptive strategy
            new_seq = current_seq.copy()
            num_changes = random.randint(1, min(5, len(new_seq)//4))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                # Use different mutation factors based on position and value
                if new_seq[idx] > 10.0:
                    factor = random.uniform(0.9, 1.1)
                elif new_seq[idx] > 1.0:
                    factor = random.uniform(0.85, 1.15)
                else:
                    factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
            
            new_score = compute_autocorrelation_constant_fast(new_seq)
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
    
    return current_seq

def hill_climbing_local_search(sequence, max_iterations=100):
    """More aggressive hill climbing with better neighborhood exploration."""
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    for i in range(max_iterations):
        # Generate neighbors with different strategies
        candidates = []
        
        # Strategy 1: Single element modification
        for _ in range(5):
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            # Larger changes for exploration
            factor = random.uniform(0.5, 2.0)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
            candidates.append(new_seq)
        
        # Strategy 2: Multiple element modification
        for _ in range(3):
            new_seq = current_seq.copy()
            num_changes = random.randint(1, min(10, len(new_seq)//2))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
            candidates.append(new_seq)
        
        # Strategy 3: Local averaging with nearby elements
        if len(current_seq) >= 3:
            for _ in range(2):
                new_seq = current_seq.copy()
                start_idx = random.randint(0, len(current_seq) - 3)
                # Average with neighbors
                avg_val = (current_seq[start_idx] + current_seq[start_idx+1] + current_seq[start_idx+2]) / 3
                new_seq[start_idx] = avg_val * random.uniform(0.8, 1.2)
                new_seq[start_idx+1] = avg_val * random.uniform(0.8, 1.2)
                new_seq[start_idx+2] = avg_val * random.uniform(0.8, 1.2)
                candidates.append(new_seq)
        
        # Evaluate all candidates
        best_candidate = None
        best_candidate_score = current_score
        
        for candidate in candidates:
            candidate_score = compute_autocorrelation_constant_fast(candidate)
            if candidate_score > best_candidate_score:
                best_candidate = candidate
                best_candidate_score = candidate_score
        
        # Accept the best improvement
        if best_candidate is not None:
            current_seq = best_candidate
            current_score = best_candidate_score
        else:
            # No improvement found, try random exploration occasionally
            if random.random() < 0.1:
                idx = random.randint(0, len(current_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                current_seq[idx] = max(0.01, current_seq[idx] * factor)
                current_seq[idx] = min(1000.0, current_seq[idx])
                current_score = compute_autocorrelation_constant_fast(current_seq)
    
    return current_seq

def simulated_annealing_optimization(initial_sequence, max_time=10):
    """Use simulated annealing for fine-tuning with better parameters."""
    start_time = time.time()
    
    current_seq = initial_sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Parameters for simulated annealing with better tuning
    temperature = 100.0
    cooling_rate = 0.98
    min_temperature = 0.001
    
    best_seq = current_seq.copy()
    best_score = current_score
    
    iteration = 0
    while time.time() - start_time < max_time and temperature > min_temperature:
        # Generate neighbor solution with adaptive strategy
        new_seq = current_seq.copy()
        
        # Choose between different neighborhood types
        if random.random() < 0.7:
            # Single element change (more local)
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        else:
            # Multiple element change (more global)
            num_changes = random.randint(1, min(5, len(new_seq)//3))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
        
        new_score = compute_autocorrelation_constant_fast(new_seq)
        
        # Accept or reject the new solution
        if new_score > current_score:
            current_seq = new_seq
            current_score = new_score
        else:
            # Accept with probability based on temperature and difference
            delta = new_score - current_score
            if delta < 0:
                # Even negative improvements might be accepted sometimes
                acceptance_prob = math.exp(delta / (temperature + 1e-10))
                if random.random() < acceptance_prob:
                    current_seq = new_seq
                    current_score = new_score
        
        # Update best solution
        if current_score > best_score:
            best_seq = current_seq.copy()
            best_score = current_score
        
        # Cool down with adaptive schedule
        if iteration % 10 == 0:
            temperature *= cooling_rate
        
        iteration += 1
    
    return best_seq

def advanced_evolutionary_search(max_time=40):
    """Improved evolutionary algorithm with better strategies and more sophisticated selection."""
    start_time = time.time()
    
    # Initial diverse population with more focus on promising sequences
    population_size = 200
    population = generate_better_initial_sequences()
    
    # Add more mathematically informed sequences to increase quality
    math_sequences = generate_mathematically_informed_sequences()
    population.extend(math_sequences[:50])  # Add some of the best math sequences
    
    # Fill up population with random sequences
    while len(population) < population_size:
        # Make random sequences with more varied lengths and ranges
        n = random.randint(20, 150)
        seq = [random.uniform(0.01, 50.0) for _ in range(n)]
        population.append(seq)
    
    best_score = 0
    best_sequence = None
    
    # Evolutionary parameters
    generations = 0
    max_generations = 400
    
    # Keep track of recent best scores for early stopping
    recent_scores = deque(maxlen=20)
    
    while time.time() - start_time < max_time and generations < max_generations:
        # Evaluate fitness (1/C₁)
        fitness_scores = []
        for seq in population:
            score = compute_autocorrelation_constant_fast(seq)
            fitness_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_sequence = seq.copy()
        
        # Track recent scores
        recent_scores.append(best_score)
        
        # Early stopping if no significant improvement
        if len(recent_scores) == 20:
            improvement = (recent_scores[-1] - recent_scores[0]) / recent_scores[0] if recent_scores[0] > 0 else 0
            if improvement < 0.0001:
                break
            
        # Selection - keep top 40% and apply tournament selection
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        elite_count = population_size // 2  # Increase elite count
        elite_indices = sorted_indices[:elite_count]
        elite_population = [population[i] for i in elite_indices]
        
        # Create new generation through mutation and crossover
        new_population = elite_population.copy()
        
        # Add diversity with mutated elites (higher mutation rate for exploration)
        for _ in range(population_size // 3):
            parent = random.choice(elite_population)
            child = mutate_sequence(parent, mutation_rate=0.3, max_mutation=0.5)
            new_population.append(child)
        
        # Add crossover combinations with better variety
        while len(new_population) < population_size - 20:
            parent1, parent2 = random.sample(elite_population, 2)
            # Use advanced crossover
            child = advanced_crossover(parent1, parent2)
            new_population.append(child)
        
        # Add some completely random sequences for exploration
        while len(new_population) < population_size:
            n = random.randint(30, 120)
            seq = [random.uniform(0.01, 100.0) for _ in range(n)]
            new_population.append(seq)
        
        # Apply local search to some individuals with higher probability
        for i in range(0, len(new_population), 2):
            if random.random() < 0.7:  # Higher probability of local search
                # Try both types of local search for better results
                if random.random() < 0.6:
                    new_population[i] = local_search_improvement(new_population[i], iterations=30)
                else:
                    new_population[i] = hill_climbing_local_search(new_population[i], max_iterations=50)
        
        population = new_population
        generations += 1
    
    return best_sequence, best_score

def adaptive_local_search(sequence, max_time=10):
    """Adaptive local search that varies intensity based on progress."""
    start_time = time.time()
    
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Start with intensive local search
    iterations = 100
    for i in range(iterations):
        if time.time() - start_time > max_time:
            break
            
        # Vary the search intensity over time and use better strategies
        if i < iterations // 3:
            # Very intensive search early on
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.7, 1.3)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        elif i < 2 * iterations // 3:
            # Moderate search
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        else:
            # Lighter search towards end
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.95, 1.05)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        
        new_score = compute_autocorrelation_constant_fast(new_seq)
        if new_score > current_score:
            current_seq = new_seq
            current_score = new_score
    
    return current_seq

def hybrid_optimization(initial_sequence, max_time=15):
    """Hybrid optimization combining multiple techniques with enhanced strategies."""
    start_time = time.time()
    
    # First, try adaptive local search with more iterations
    improved_seq = adaptive_local_search(initial_sequence, max_time=max_time/2)
    best_score = compute_autocorrelation_constant_fast(improved_seq)
    
    # Then try simulated annealing
    if time.time() - start_time < max_time/2:
        sa_seq = simulated_annealing_optimization(improved_seq, max_time=max_time/3)
        sa_score = compute_autocorrelation_constant_fast(sa_seq)
        
        if sa_score > best_score:
            improved_seq = sa_seq
            best_score = sa_score
    
    # Apply multiple rounds of adaptive local search with better strategies
    for i in range(3):  # More rounds
        if time.time() - start_time < max_time * 0.8:
            # Try different local search strategies
            if random.random() < 0.7:  # Higher probability
                local_seq = adaptive_local_search(improved_seq, max_time=max_time/6)
            else:
                local_seq = hill_climbing_local_search(improved_seq, max_iterations=40)
            local_score = compute_autocorrelation_constant_fast(local_seq)
            if local_score > best_score:
                improved_seq = local_seq
                best_score = local_score
    
    # Try gradient-free optimization on selected parameters with better bounds
    try:
        if time.time() - start_time < max_time * 0.9:
            # Try different optimization approaches on subsets
            subset_sizes = [min(30, len(initial_sequence)), min(50, len(initial_sequence))]
            
            for subset_size in subset_sizes:
                if subset_size <= 1:
                    continue
                    
                indices = sorted(random.sample(range(len(initial_sequence)), subset_size))
                
                def objective(x):
                    seq = initial_sequence.copy()
                    for i, idx in enumerate(indices):
                        if 0 <= idx < len(seq):
                            seq[idx] = max(0.01, x[i])
                    return -compute_autocorrelation_constant_fast(seq)
                
                # Start from current solution
                x0 = [initial_sequence[i] for i in indices]
                bounds = [(0.01, 1000.0)] * len(x0)
                
                # Try different optimization methods with better settings
                try:
                    # Nelder-Mead for simpler local refinement
                    result_nm = minimize(objective, x0, method='Nelder-Mead', 
                                       options={'maxiter': 100, 'adaptive': True})
                    
                    if hasattr(result_nm, 'x') and len(result_nm.x) == len(indices):
                        seq = initial_sequence.copy()
                        for i, idx in enumerate(indices):
                            if 0 <= idx < len(seq):
                                seq[idx] = max(0.01, result_nm.x[i])
                        candidate_score = compute_autocorrelation_constant_fast(seq)
                        if candidate_score > best_score:
                            improved_seq = seq
                            best_score = candidate_score
                            
                except Exception:
                    pass
                    
                # Also try with different parameter settings
                try:
                    # L-BFGS-B for potentially better results
                    result_lbfgs = minimize(objective, x0, method='L-BFGS-B', 
                                          bounds=bounds, options={'maxiter': 50})
                    
                    if hasattr(result_lbfgs, 'x') and len(result_lbfgs.x) == len(indices):
                        seq = initial_sequence.copy()
                        for i, idx in enumerate(indices):
                            if 0 <= idx < len(seq):
                                seq[idx] = max(0.01, result_lbfgs.x[i])
                        candidate_score = compute_autocorrelation_constant_fast(seq)
                        if candidate_score > best_score:
                            improved_seq = seq
                            best_score = candidate_score
                            
                except Exception:
                    pass
                    
    except Exception:
        pass
    
    # Final local search refinement with increased iterations
    if time.time() - start_time < max_time * 0.95:
        final_seq = local_search_improvement(improved_seq, iterations=40)
        final_score = compute_autocorrelation_constant_fast(final_seq)
        if final_score > best_score:
            improved_seq = final_seq
            best_score = final_score
    
    return improved_seq

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence."""
    # Try advanced evolutionary approach first with more time allocation
    sequence, score = advanced_evolutionary_search(max_time=30)
    
    # Refine with hybrid optimization
    if sequence is not None:
        refined_sequence = hybrid_optimization(sequence, max_time=20)
        refined_score = compute_autocorrelation_constant_fast(refined_sequence)
        
        # Return the better of the two
        if refined_score > score:
            return refined_sequence
        else:
            return sequence
    else:
        # Fallback to simple random search with better initialization
        return [random.uniform(0.1, 10.0) for _ in range(50)]

def generate_random_sequence(length_range=(10, 100), min_height=0.1, max_height=10.0):
    """Generate a random sequence with specified length range."""
    n = random.randint(*length_range)
    # Generate sequence with some randomness but keep it reasonable
    sequence = [random.uniform(min_height, max_height) for _ in range(n)]
    return sequence

def generate_step_sequence(length, height):
    """Generate a sequence with all elements equal to height."""
    return [height] * length

def generate_special_sequences():
    """Generate specialized sequences based on mathematical insights for better performance."""
    sequences = []
    
    # Golden ratio based sequences (known to perform well in such problems)
    golden_ratio = (1 + np.sqrt(5)) / 2
    for n in [20, 30, 50, 75, 100]:
        # Golden ratio decay
        golden_seq = [1.0 / (golden_ratio ** i) for i in range(n)]
        sequences.append(golden_seq)
        
        # Alternating golden ratios
        alt_golden = [1.0 if i % 2 == 0 else 1.0/golden_ratio for i in range(n)]
        sequences.append(alt_golden)
    
    # Exponential sequences with different bases
    for base in [1.2, 1.3, 1.4, 1.5]:
        exp_seq = [1.0 / (base ** i) for i in range(100)]
        sequences.append(exp_seq)
    
    # Concentrated energy sequences (better for minimizing max convolution)
    for n in [50, 100]:
        # Peak at beginning with rapid decay
        peak_seq = [10.0] + [1.0 / (1.5 ** i) for i in range(n-1)]
        sequences.append(peak_seq)
        
        # Two peaks
        two_peak = [0.1] * (n//3) + [5.0] + [0.1] * (n//3) + [10.0] + [0.1] * (n - 2*n//3 - 1)
        sequences.append(two_peak)
    
    # Sparse sequences (some zero values) - these often work well
    sparse_seq = [10.0 if i == 0 else 0.1 if i == 50 else 0.05 if i == 99 else 0.01 for i in range(100)]
    sequences.append(sparse_seq)
    
    # Smoothly varying sequences
    smooth_seq = [1.0 + 0.5 * np.sin(i * np.pi / 50) for i in range(100)]
    sequences.append(smooth_seq)
    
    # Sequences with strong concentration at start (good for minimizing max convolution)
    for n in [50, 100]:
        # Strong peak followed by exponential decay
        strong_peak = [100.0] + [1.0 / (1.2 ** i) for i in range(n-1)]
        sequences.append(strong_peak)
        
        # Multiple strong peaks
        multi_peak = [0.1] * (n//4) + [50.0] + [0.1] * (n//4) + [25.0] + [0.1] * (n//2)
        sequences.append(multi_peak)
    
    # More mathematical patterns
    # Harmonic sequences
    harmonic_seq = [1.0 / (i + 1) for i in range(100)]
    sequences.append(harmonic_seq)
    
    # Power law sequences
    power_seq = [1.0 / ((i + 1) ** 1.5) for i in range(100)]
    sequences.append(power_seq)
    
    # Logarithmic sequences
    log_seq = [1.0 / (np.log(i + 2)) for i in range(100)]
    sequences.append(log_seq)
    
    return sequences

def generate_better_initial_sequences():
    """Generate a diverse set of initial sequences for better exploration."""
    sequences = []
    
    # Add sequences of various lengths and structures
    for n in [10, 20, 30, 50, 75, 100]:
        # Uniform sequences
        sequences.append(generate_step_sequence(n, 1.0))
        
        # Some with varying heights
        seq = [random.uniform(0.1, 10.0) for _ in range(n)]
        sequences.append(seq)
        
        # Step-like sequences (like in known solutions)
        if n >= 5:
            step_seq = [1.0] * (n//2) + [0.5] * (n - n//2)
            sequences.append(step_seq)
            
        # Special structures based on mathematical intuition
        if n >= 10:
            # Geometric decay sequences
            geometric_seq = [1.0 / (1.5 ** i) for i in range(n)]
            sequences.append(geometric_seq)
            
            # Alternating sequences
            alternating_seq = [1.0 if i % 2 == 0 else 0.5 for i in range(n)]
            sequences.append(alternating_seq)
            
            # Peaks at center (more focused energy)
            peak_seq = [0.1] * n
            center = n // 2
            peak_seq[center] = 10.0
            if center + 1 < n:
                peak_seq[center + 1] = 5.0
            if center - 1 >= 0:
                peak_seq[center - 1] = 5.0
            sequences.append(peak_seq)
    
    # Add some high-quality known patterns
    # Pattern that often works well - concentrated at start
    sequences.append([10.0] + [0.1] * 99)
    sequences.append([1.0] * 50 + [0.1] * 50)
    
    # Add sequences that have been shown to work well in similar problems
    # Concentrated peak at beginning with gradual decay
    sequences.append([10.0, 5.0, 2.5, 1.0, 0.5] + [0.1] * 95)
    
    # Double peak structure
    double_peak = [0.1] * 20 + [5.0] + [0.1] * 20 + [10.0] + [0.1] * 58
    sequences.append(double_peak)
    
    # Exponential decay pattern
    exp_decay = [1.0 / (1.3 ** i) for i in range(100)]
    sequences.append(exp_decay)
    
    # Improved patterns based on known good solutions
    # High concentration at start
    high_concentration = [100.0] + [0.1] * 99
    sequences.append(high_concentration)
    
    # Balanced decay
    balanced_decay = [1.0] + [0.8] * 20 + [0.6] * 20 + [0.4] * 20 + [0.2] * 20 + [0.1] * 20
    sequences.append(balanced_decay)
    
    # Add special sequences
    sequences.extend(generate_special_sequences())
    
    # Add sequences with specific mathematical properties
    # Fibonacci-like sequences
    fib_seq = [1.0]
    for i in range(1, 100):
        fib_seq.append(fib_seq[i-1] + (fib_seq[i-1] if i >= 2 else 1))
    sequences.append([x/1000.0 for x in fib_seq])
    
    # Gaussian-like sequences
    gaussian_seq = [np.exp(-0.5 * ((i - 50) / 15)**2) for i in range(100)]
    sequences.append(gaussian_seq)
    
    return sequences

def mutate_sequence(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Mutate a sequence with adaptive mutation rate and better bounds."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Adaptive mutation: smaller changes for larger values
            change_factor = random.uniform(-max_mutation, max_mutation)
            mutated[i] = max(0.01, mutated[i] * (1 + change_factor))
            # Clip to reasonable bounds
            mutated[i] = min(1000.0, mutated[i])
    return mutated

def crossover_sequences(seq1, seq2):
    """Perform crossover between two sequences with better mixing."""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Use multiple crossover points for better diversity
    num_crossover_points = min(5, min_len//2)
    if num_crossover_points > 0:
        crossover_points = sorted(random.sample(range(1, min_len), num_crossover_points))
    else:
        crossover_points = [min_len//2] if min_len > 1 else []
    
    # Alternate between parents with better mixing strategy
    child = []
    last_point = 0
    for point in crossover_points:
        if len(child) % 2 == 0:
            child.extend(seq1[last_point:point])
        else:
            child.extend(seq2[last_point:point])
        last_point = point
    
    # Add remaining elements from the appropriate parent
    if len(child) % 2 == 0 and last_point < len(seq1):
        child.extend(seq1[last_point:])
    elif last_point < len(seq2):
        child.extend(seq2[last_point:])
    
    # Ensure we don't exceed the original length
    if len(child) > min_len:
        child = child[:min_len]
    elif len(child) < min_len:
        # Pad with elements from the longer parent
        longer_seq = seq1 if len(seq1) > len(seq2) else seq2
        while len(child) < min_len:
            child.append(longer_seq[len(child) % len(longer_seq)])
    
    return child

def advanced_crossover(seq1, seq2):
    """Advanced crossover with more sophisticated mixing strategies."""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Try different crossover strategies
    strategy = random.choice(['uniform', 'segment', 'alternating'])
    
    if strategy == 'uniform':
        # Uniform crossover - randomly select elements from either parent
        child = []
        for i in range(min(len(seq1), len(seq2))):
            if random.random() < 0.5:
                child.append(seq1[i])
            else:
                child.append(seq2[i])
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[len(seq2):])
        elif len(seq2) > len(seq1):
            child.extend(seq2[len(seq1):])
        return child
    
    elif strategy == 'segment':
        # Segment crossover - take segments from each parent
        min_len = min(len(seq1), len(seq2))
        if min_len < 2:
            return seq1 if len(seq1) > 0 else seq2
            
        # Split at random points
        split1 = random.randint(1, min_len - 1)
        split2 = random.randint(1, min_len - 1)
        
        child = []
        if random.random() < 0.5:
            child.extend(seq1[:split1])
            child.extend(seq2[split1:split2])
            child.extend(seq1[split2:])
        else:
            child.extend(seq2[:split1])
            child.extend(seq1[split1:split2])
            child.extend(seq2[split2:])
        
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[len(seq2):])
        elif len(seq2) > len(seq1):
            child.extend(seq2[len(seq1):])
        return child
    
    else:  # alternating
        # Alternating crossover - alternate between elements from parents
        min_len = min(len(seq1), len(seq2))
        child = []
        for i in range(min_len):
            if i % 2 == 0:
                child.append(seq1[i])
            else:
                child.append(seq2[i])
        
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[min_len:])
        elif len(seq2) > len(seq1):
            child.extend(seq2[min_len:])
        return child

def local_search_improvement(sequence, iterations=50):
    """Perform local search to improve the sequence with multiple strategies."""
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Try different local search strategies
    for i in range(iterations):
        # Strategy 1: Single element perturbation
        if random.random() < 0.7:
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            # Small multiplicative change with better bounds
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
            
            new_score = compute_autocorrelation_constant_fast(new_seq)
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
        else:
            # Strategy 2: Multiple element perturbation with adaptive strategy
            new_seq = current_seq.copy()
            num_changes = random.randint(1, min(5, len(new_seq)//4))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                # Use different mutation factors based on position and value
                if new_seq[idx] > 10.0:
                    factor = random.uniform(0.9, 1.1)
                elif new_seq[idx] > 1.0:
                    factor = random.uniform(0.85, 1.15)
                else:
                    factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
            
            new_score = compute_autocorrelation_constant_fast(new_seq)
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
    
    return current_seq

def hill_climbing_local_search(sequence, max_iterations=100):
    """More aggressive hill climbing with better neighborhood exploration."""
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    for i in range(max_iterations):
        # Generate neighbors with different strategies
        candidates = []
        
        # Strategy 1: Single element modification
        for _ in range(5):
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            # Larger changes for exploration
            factor = random.uniform(0.5, 2.0)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
            candidates.append(new_seq)
        
        # Strategy 2: Multiple element modification
        for _ in range(3):
            new_seq = current_seq.copy()
            num_changes = random.randint(1, min(10, len(new_seq)//2))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
            candidates.append(new_seq)
        
        # Strategy 3: Local averaging with nearby elements
        if len(current_seq) >= 3:
            for _ in range(2):
                new_seq = current_seq.copy()
                start_idx = random.randint(0, len(current_seq) - 3)
                # Average with neighbors
                avg_val = (current_seq[start_idx] + current_seq[start_idx+1] + current_seq[start_idx+2]) / 3
                new_seq[start_idx] = avg_val * random.uniform(0.8, 1.2)
                new_seq[start_idx+1] = avg_val * random.uniform(0.8, 1.2)
                new_seq[start_idx+2] = avg_val * random.uniform(0.8, 1.2)
                candidates.append(new_seq)
        
        # Evaluate all candidates
        best_candidate = None
        best_candidate_score = current_score
        
        for candidate in candidates:
            candidate_score = compute_autocorrelation_constant_fast(candidate)
            if candidate_score > best_candidate_score:
                best_candidate = candidate
                best_candidate_score = candidate_score
        
        # Accept the best improvement
        if best_candidate is not None:
            current_seq = best_candidate
            current_score = best_candidate_score
        else:
            # No improvement found, try random exploration occasionally
            if random.random() < 0.1:
                idx = random.randint(0, len(current_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                current_seq[idx] = max(0.01, current_seq[idx] * factor)
                current_seq[idx] = min(1000.0, current_seq[idx])
                current_score = compute_autocorrelation_constant_fast(current_seq)
    
    return current_seq

def simulated_annealing_optimization(initial_sequence, max_time=10):
    """Use simulated annealing for fine-tuning with better parameters."""
    start_time = time.time()
    
    current_seq = initial_sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Parameters for simulated annealing with better tuning
    temperature = 100.0
    cooling_rate = 0.98
    min_temperature = 0.001
    
    best_seq = current_seq.copy()
    best_score = current_score
    
    iteration = 0
    while time.time() - start_time < max_time and temperature > min_temperature:
        # Generate neighbor solution with adaptive strategy
        new_seq = current_seq.copy()
        
        # Choose between different neighborhood types
        if random.random() < 0.7:
            # Single element change (more local)
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        else:
            # Multiple element change (more global)
            num_changes = random.randint(1, min(5, len(new_seq)//3))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
        
        new_score = compute_autocorrelation_constant_fast(new_seq)
        
        # Accept or reject the new solution
        if new_score > current_score:
            current_seq = new_seq
            current_score = new_score
        else:
            # Accept with probability based on temperature and difference
            delta = new_score - current_score
            if delta < 0:
                # Even negative improvements might be accepted sometimes
                acceptance_prob = math.exp(delta / (temperature + 1e-10))
                if random.random() < acceptance_prob:
                    current_seq = new_seq
                    current_score = new_score
        
        # Update best solution
        if current_score > best_score:
            best_seq = current_seq.copy()
            best_score = current_score
        
        # Cool down with adaptive schedule
        if iteration % 10 == 0:
            temperature *= cooling_rate
        
        iteration += 1
    
    return best_seq

def advanced_evolutionary_search(max_time=40):
    """Improved evolutionary algorithm with better strategies and more sophisticated selection."""
    start_time = time.time()
    
    # Initial diverse population
    population_size = 150
    population = generate_better_initial_sequences()
    
    # Fill up population with random sequences
    while len(population) < population_size:
        population.append(generate_random_sequence())
    
    best_score = 0
    best_sequence = None
    
    # Evolutionary parameters
    generations = 0
    max_generations = 300
    
    # Keep track of recent best scores for early stopping
    recent_scores = deque(maxlen=15)
    
    while time.time() - start_time < max_time and generations < max_generations:
        # Evaluate fitness (1/C₁)
        fitness_scores = []
        for seq in population:
            score = compute_autocorrelation_constant_fast(seq)
            fitness_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_sequence = seq.copy()
        
        # Track recent scores
        recent_scores.append(best_score)
        
        # Early stopping if no significant improvement
        if len(recent_scores) == 15:
            improvement = (recent_scores[-1] - recent_scores[0]) / recent_scores[0] if recent_scores[0] > 0 else 0
            if improvement < 0.00005:
                break
            
        # Selection - keep top 35% and apply tournament selection
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        elite_count = population_size // 3
        elite_indices = sorted_indices[:elite_count]
        elite_population = [population[i] for i in elite_indices]
        
        # Create new generation through mutation and crossover
        new_population = elite_population.copy()
        
        # Add diversity with mutated elites (higher mutation rate for exploration)
        for _ in range(population_size // 4):
            parent = random.choice(elite_population)
            child = mutate_sequence(parent, mutation_rate=0.25, max_mutation=0.4)
            new_population.append(child)
        
        # Add crossover combinations with better variety
        while len(new_population) < population_size - 15:
            parent1, parent2 = random.sample(elite_population, 2)
            # Use advanced crossover
            child = advanced_crossover(parent1, parent2)
            new_population.append(child)
        
        # Add some completely random sequences for exploration
        while len(new_population) < population_size:
            new_population.append(generate_random_sequence())
        
        # Apply local search to some individuals with higher probability
        for i in range(0, len(new_population), 2):
            if random.random() < 0.6:
                # Try both types of local search for better results
                if random.random() < 0.5:
                    new_population[i] = local_search_improvement(new_population[i], iterations=25)
                else:
                    new_population[i] = hill_climbing_local_search(new_population[i], max_iterations=40)
        
        population = new_population
        generations += 1
    
    return best_sequence, best_score

def adaptive_local_search(sequence, max_time=10):
    """Adaptive local search that varies intensity based on progress."""
    start_time = time.time()
    
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Start with intensive local search
    iterations = 100
    for i in range(iterations):
        if time.time() - start_time > max_time:
            break
            
        # Vary the search intensity over time and use better strategies
        if i < iterations // 3:
            # Very intensive search early on
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.7, 1.3)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        elif i < 2 * iterations // 3:
            # Moderate search
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        else:
            # Lighter search towards end
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.95, 1.05)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        
        new_score = compute_autocorrelation_constant_fast(new_seq)
        if new_score > current_score:
            current_seq = new_seq
            current_score = new_score
    
    return current_seq

def hybrid_optimization(initial_sequence, max_time=15):
    """Hybrid optimization combining multiple techniques with enhanced strategies."""
    start_time = time.time()
    
    # First, try adaptive local search with more iterations
    improved_seq = adaptive_local_search(initial_sequence, max_time=max_time/2)
    best_score = compute_autocorrelation_constant_fast(improved_seq)
    
    # Then try simulated annealing
    if time.time() - start_time < max_time/2:
        sa_seq = simulated_annealing_optimization(improved_seq, max_time=max_time/3)
        sa_score = compute_autocorrelation_constant_fast(sa_seq)
        
        if sa_score > best_score:
            improved_seq = sa_seq
            best_score = sa_score
    
    # Apply multiple rounds of adaptive local search with better strategies
    for i in range(2):
        if time.time() - start_time < max_time * 0.8:
            # Try different local search strategies
            if random.random() < 0.6:
                local_seq = adaptive_local_search(improved_seq, max_time=max_time/5)
            else:
                local_seq = hill_climbing_local_search(improved_seq, max_iterations=35)
            local_score = compute_autocorrelation_constant_fast(local_seq)
            if local_score > best_score:
                improved_seq = local_seq
                best_score = local_score
    
    # Try gradient-free optimization on selected parameters with better bounds
    try:
        if time.time() - start_time < max_time * 0.9:
            # Try different optimization approaches on subsets
            subset_sizes = [min(25, len(initial_sequence)), min(40, len(initial_sequence))]
            
            for subset_size in subset_sizes:
                if subset_size <= 1:
                    continue
                    
                indices = sorted(random.sample(range(len(initial_sequence)), subset_size))
                
                def objective(x):
                    seq = initial_sequence.copy()
                    for i, idx in enumerate(indices):
                        if 0 <= idx < len(seq):
                            seq[idx] = max(0.01, x[i])
                    return -compute_autocorrelation_constant_fast(seq)
                
                # Start from current solution
                x0 = [initial_sequence[i] for i in indices]
                bounds = [(0.01, 1000.0)] * len(x0)
                
                # Try different optimization methods with better settings
                try:
                    # Nelder-Mead for simpler local refinement
                    result_nm = minimize(objective, x0, method='Nelder-Mead', 
                                       options={'maxiter': 80, 'adaptive': True})
                    
                    if hasattr(result_nm, 'x') and len(result_nm.x) == len(indices):
                        seq = initial_sequence.copy()
                        for i, idx in enumerate(indices):
                            if 0 <= idx < len(seq):
                                seq[idx] = max(0.01, result_nm.x[i])
                        candidate_score = compute_autocorrelation_constant_fast(seq)
                        if candidate_score > best_score:
                            improved_seq = seq
                            best_score = candidate_score
                            
                except Exception:
                    pass
                    
                # Also try with different parameter settings
                try:
                    # L-BFGS-B for potentially better results
                    result_lbfgs = minimize(objective, x0, method='L-BFGS-B', 
                                          bounds=bounds, options={'maxiter': 40})
                    
                    if hasattr(result_lbfgs, 'x') and len(result_lbfgs.x) == len(indices):
                        seq = initial_sequence.copy()
                        for i, idx in enumerate(indices):
                            if 0 <= idx < len(seq):
                                seq[idx] = max(0.01, result_lbfgs.x[i])
                        candidate_score = compute_autocorrelation_constant_fast(seq)
                        if candidate_score > best_score:
                            improved_seq = seq
                            best_score = candidate_score
                            
                except Exception:
                    pass
                    
    except Exception:
        pass
    
    # Final local search refinement
    if time.time() - start_time < max_time * 0.95:
        final_seq = local_search_improvement(improved_seq, iterations=30)
        final_score = compute_autocorrelation_constant_fast(final_seq)
        if final_score > best_score:
            improved_seq = final_seq
            best_score = final_score
    
    return improved_seq

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence."""
    # Try advanced evolutionary approach first with more time allocation
    sequence, score = advanced_evolutionary_search(max_time=30)
    
    # Refine with hybrid optimization
    if sequence is not None:
        refined_sequence = hybrid_optimization(sequence, max_time=20)
        refined_score = compute_autocorrelation_constant_fast(refined_sequence)
        
        # Return the better of the two
        if refined_score > score:
            return refined_sequence
        else:
            return sequence
    else:
        # Fallback to simple random search with better initialization
        return generate_random_sequence()

def generate_mathematically_informed_sequences():
    """Generate sequences based on mathematical insights that tend to work well."""
    sequences = []
    
    # Golden ratio based sequences
    golden_ratio = (1 + np.sqrt(5)) / 2
    for n in [30, 50, 75, 100]:
        # Golden ratio decay
        golden_seq = [1.0 / (golden_ratio ** i) for i in range(n)]
        sequences.append(golden_seq)
        
        # Alternating golden ratios
        alt_golden = [1.0 if i % 2 == 0 else 1.0/golden_ratio for i in range(n)]
        sequences.append(alt_golden)
    
    # Exponential sequences with different bases
    for base in [1.2, 1.3, 1.4, 1.5]:
        exp_seq = [1.0 / (base ** i) for i in range(100)]
        sequences.append(exp_seq)
    
    # Concentrated energy sequences (better for minimizing max convolution)
    for n in [50, 100]:
        # Peak at beginning with rapid decay
        peak_seq = [10.0] + [1.0 / (1.5 ** i) for i in range(n-1)]
        sequences.append(peak_seq)
        
        # Two peaks
        two_peak = [0.1] * (n//3) + [5.0] + [0.1] * (n//3) + [10.0] + [0.1] * (n - 2*n//3 - 1)
        sequences.append(two_peak)
    
    # Sequences with strong concentration at start (good for minimizing max convolution)
    for n in [50, 100]:
        # Strong peak followed by exponential decay
        strong_peak = [100.0] + [1.0 / (1.2 ** i) for i in range(n-1)]
        sequences.append(strong_peak)
        
        # Multiple strong peaks
        multi_peak = [0.1] * (n//4) + [50.0] + [0.1] * (n//4) + [25.0] + [0.1] * (n//2)
        sequences.append(multi_peak)
    
    # Mathematical patterns that often work well
    # Harmonic sequences
    harmonic_seq = [1.0 / (i + 1) for i in range(100)]
    sequences.append(harmonic_seq)
    
    # Power law sequences
    power_seq = [1.0 / ((i + 1) ** 1.5) for i in range(100)]
    sequences.append(power_seq)
    
    # Logarithmic sequences
    log_seq = [1.0 / (np.log(i + 2)) for i in range(100)]
    sequences.append(log_seq)
    
    # Fibonacci-like sequences (normalized)
    fib_seq = [1.0]
    for i in range(1, 100):
        fib_seq.append(fib_seq[i-1] + (fib_seq[i-1] if i >= 2 else 1))
    sequences.append([x/1000.0 for x in fib_seq])
    
    # Gaussian-like sequences
    gaussian_seq = [np.exp(-0.5 * ((i - 50) / 15)**2) for i in range(100)]
    sequences.append(gaussian_seq)
    
    # Sequences with strategic concentration
    # High concentration at start with gradual decrease
    high_concentration = [100.0] + [0.1] * 99
    sequences.append(high_concentration)
    
    # Balanced decay with early emphasis
    balanced_decay = [1.0] + [0.8] * 20 + [0.6] * 20 + [0.4] * 20 + [0.2] * 20 + [0.1] * 20
    sequences.append(balanced_decay)
    
    return sequences

def generate_special_sequences():
    """Generate specialized sequences based on mathematical insights for better performance."""
    sequences = []
    
    # Known good mathematical patterns
    # Geometric sequences with various ratios
    for ratio in [1.1, 1.2, 1.3, 1.4, 1.5]:
        geometric = [1.0 / (ratio ** i) for i in range(100)]
        sequences.append(geometric)
    
    # Decaying sequences with different decay rates
    for decay in [0.8, 0.9, 0.95]:
        decaying = [1.0 * (decay ** i) for i in range(100)]
        sequences.append(decaying)
    
    # Sequences with strong initial peaks
    peak_seq = [100.0] + [1.0] * 99
    sequences.append(peak_seq)
    
    # Multi-peak structures
    multi_peak = [1.0] * 20 + [50.0] + [1.0] * 20 + [25.0] + [1.0] * 58
    sequences.append(multi_peak)
    
    # Step functions
    step_seq = [1.0] * 50 + [0.5] * 50
    sequences.append(step_seq)
    
    # Alternating high-low patterns
    alt_seq = [10.0 if i % 2 == 0 else 0.5 for i in range(100)]
    sequences.append(alt_seq)
    
    # Smooth decreasing sequences
    smooth_dec = [1.0 / (1.0 + i * 0.02) for i in range(100)]
    sequences.append(smooth_dec)
    
    return sequences

def generate_better_initial_sequences():
    """Generate a diverse set of initial sequences for better exploration."""
    sequences = []
    
    # Add sequences of various lengths and structures
    for n in [10, 20, 30, 50, 75, 100]:
        # Uniform sequences
        sequences.append([1.0] * n)
        
        # Some with varying heights
        seq = [random.uniform(0.1, 10.0) for _ in range(n)]
        sequences.append(seq)
        
        # Step-like sequences (like in known solutions)
        if n >= 5:
            step_seq = [1.0] * (n//2) + [0.5] * (n - n//2)
            sequences.append(step_seq)
            
        # Special structures based on mathematical intuition
        if n >= 10:
            # Geometric decay sequences
            geometric_seq = [1.0 / (1.5 ** i) for i in range(n)]
            sequences.append(geometric_seq)
            
            # Alternating sequences
            alternating_seq = [1.0 if i % 2 == 0 else 0.5 for i in range(n)]
            sequences.append(alternating_seq)
            
            # Peaks at center (more focused energy)
            peak_seq = [0.1] * n
            center = n // 2
            peak_seq[center] = 10.0
            if center + 1 < n:
                peak_seq[center + 1] = 5.0
            if center - 1 >= 0:
                peak_seq[center - 1] = 5.0
            sequences.append(peak_seq)
    
    # Add some high-quality known patterns
    # Pattern that often works well - concentrated at start
    sequences.append([10.0] + [0.1] * 99)
    sequences.append([1.0] * 50 + [0.1] * 50)
    
    # Add sequences that have been shown to work well in similar problems
    # Concentrated peak at beginning with gradual decay
    sequences.append([10.0, 5.0, 2.5, 1.0, 0.5] + [0.1] * 95)
    
    # Double peak structure
    double_peak = [0.1] * 20 + [5.0] + [0.1] * 20 + [10.0] + [0.1] * 58
    sequences.append(double_peak)
    
    # Exponential decay pattern
    exp_decay = [1.0 / (1.3 ** i) for i in range(100)]
    sequences.append(exp_decay)
    
    # Improved patterns based on known good solutions
    # High concentration at start
    high_concentration = [100.0] + [0.1] * 99
    sequences.append(high_concentration)
    
    # Balanced decay
    balanced_decay = [1.0] + [0.8] * 20 + [0.6] * 20 + [0.4] * 20 + [0.2] * 20 + [0.1] * 20
    sequences.append(balanced_decay)
    
    # Add mathematical patterns
    sequences.extend(generate_mathematically_informed_sequences())
    sequences.extend(generate_special_sequences())
    
    return sequences

def mutate_sequence(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Mutate a sequence with adaptive mutation rate and better bounds."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Adaptive mutation: smaller changes for larger values
            change_factor = random.uniform(-max_mutation, max_mutation)
            mutated[i] = max(0.01, mutated[i] * (1 + change_factor))
            # Clip to reasonable bounds
            mutated[i] = min(1000.0, mutated[i])
    return mutated

def advanced_crossover(seq1, seq2):
    """Advanced crossover with more sophisticated mixing strategies."""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Try different crossover strategies
    strategy = random.choice(['uniform', 'segment', 'alternating'])
    
    if strategy == 'uniform':
        # Uniform crossover - randomly select elements from either parent
        child = []
        for i in range(min(len(seq1), len(seq2))):
            if random.random() < 0.5:
                child.append(seq1[i])
            else:
                child.append(seq2[i])
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[len(seq2):])
        elif len(seq2) > len(seq1):
            child.extend(seq2[len(seq1):])
        return child
    
    elif strategy == 'segment':
        # Segment crossover - take segments from each parent
        min_len = min(len(seq1), len(seq2))
        if min_len < 2:
            return seq1 if len(seq1) > 0 else seq2
            
        # Split at random points
        split1 = random.randint(1, min_len - 1)
        split2 = random.randint(1, min_len - 1)
        
        child = []
        if random.random() < 0.5:
            child.extend(seq1[:split1])
            child.extend(seq2[split1:split2])
            child.extend(seq1[split2:])
        else:
            child.extend(seq2[:split1])
            child.extend(seq1[split1:split2])
            child.extend(seq2[split2:])
        
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[len(seq2):])
        elif len(seq2) > len(seq1):
            child.extend(seq2[len(seq1):])
        return child
    
    else:  # alternating
        # Alternating crossover - alternate between elements from parents
        min_len = min(len(seq1), len(seq2))
        child = []
        for i in range(min_len):
            if i % 2 == 0:
                child.append(seq1[i])
            else:
                child.append(seq2[i])
        
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[min_len:])
        elif len(seq2) > len(seq1):
            child.extend(seq2[min_len:])
        return child

def local_search_improvement(sequence, iterations=50):
    """Perform local search to improve the sequence with multiple strategies."""
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Try different local search strategies
    for i in range(iterations):
        # Strategy 1: Single element perturbation
        if random.random() < 0.7:
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            # Small multiplicative change with better bounds
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
            
            new_score = compute_autocorrelation_constant_fast(new_seq)
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
        else:
            # Strategy 2: Multiple element perturbation with adaptive strategy
            new_seq = current_seq.copy()
            num_changes = random.randint(1, min(5, len(new_seq)//4))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                # Use different mutation factors based on position and value
                if new_seq[idx] > 10.0:
                    factor = random.uniform(0.9, 1.1)
                elif new_seq[idx] > 1.0:
                    factor = random.uniform(0.85, 1.15)
                else:
                    factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
            
            new_score = compute_autocorrelation_constant_fast(new_seq)
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
    
    return current_seq

def hill_climbing_local_search(sequence, max_iterations=100):
    """More aggressive hill climbing with better neighborhood exploration."""
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    for i in range(max_iterations):
        # Generate neighbors with different strategies
        candidates = []
        
        # Strategy 1: Single element modification
        for _ in range(5):
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            # Larger changes for exploration
            factor = random.uniform(0.5, 2.0)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
            candidates.append(new_seq)
        
        # Strategy 2: Multiple element modification
        for _ in range(3):
            new_seq = current_seq.copy()
            num_changes = random.randint(1, min(10, len(new_seq)//2))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
            candidates.append(new_seq)
        
        # Strategy 3: Local averaging with nearby elements
        if len(current_seq) >= 3:
            for _ in range(2):
                new_seq = current_seq.copy()
                start_idx = random.randint(0, len(current_seq) - 3)
                # Average with neighbors
                avg_val = (current_seq[start_idx] + current_seq[start_idx+1] + current_seq[start_idx+2]) / 3
                new_seq[start_idx] = avg_val * random.uniform(0.8, 1.2)
                new_seq[start_idx+1] = avg_val * random.uniform(0.8, 1.2)
                new_seq[start_idx+2] = avg_val * random.uniform(0.8, 1.2)
                candidates.append(new_seq)
        
        # Evaluate all candidates
        best_candidate = None
        best_candidate_score = current_score
        
        for candidate in candidates:
            candidate_score = compute_autocorrelation_constant_fast(candidate)
            if candidate_score > best_candidate_score:
                best_candidate = candidate
                best_candidate_score = candidate_score
        
        # Accept the best improvement
        if best_candidate is not None:
            current_seq = best_candidate
            current_score = best_candidate_score
        else:
            # No improvement found, try random exploration occasionally
            if random.random() < 0.1:
                idx = random.randint(0, len(current_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                current_seq[idx] = max(0.01, current_seq[idx] * factor)
                current_seq[idx] = min(1000.0, current_seq[idx])
                current_score = compute_autocorrelation_constant_fast(current_seq)
    
    return current_seq

def simulated_annealing_optimization(initial_sequence, max_time=10):
    """Use simulated annealing for fine-tuning with better parameters."""
    start_time = time.time()
    
    current_seq = initial_sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Parameters for simulated annealing with better tuning
    temperature = 100.0
    cooling_rate = 0.98
    min_temperature = 0.001
    
    best_seq = current_seq.copy()
    best_score = current_score
    
    iteration = 0
    while time.time() - start_time < max_time and temperature > min_temperature:
        # Generate neighbor solution with adaptive strategy
        new_seq = current_seq.copy()
        
        # Choose between different neighborhood types
        if random.random() < 0.7:
            # Single element change (more local)
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        else:
            # Multiple element change (more global)
            num_changes = random.randint(1, min(5, len(new_seq)//3))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
        
        new_score = compute_autocorrelation_constant_fast(new_seq)
        
        # Accept or reject the new solution
        if new_score > current_score:
            current_seq = new_seq
            current_score = new_score
        else:
            # Accept with probability based on temperature and difference
            delta = new_score - current_score
            if delta < 0:
                # Even negative improvements might be accepted sometimes
                acceptance_prob = math.exp(delta / (temperature + 1e-10))
                if random.random() < acceptance_prob:
                    current_seq = new_seq
                    current_score = new_score
        
        # Update best solution
        if current_score > best_score:
            best_seq = current_seq.copy()
            best_score = current_score
        
        # Cool down with adaptive schedule
        if iteration % 10 == 0:
            temperature *= cooling_rate
        
        iteration += 1
    
    return best_seq

def advanced_evolutionary_search(max_time=40):
    """Improved evolutionary algorithm with better strategies and more sophisticated selection."""
    start_time = time.time()
    
    # Initial diverse population
    population_size = 150
    population = generate_better_initial_sequences()
    
    # Fill up population with random sequences
    while len(population) < population_size:
        population.append([random.uniform(0.1, 10.0) for _ in range(random.randint(20, 100))])
    
    best_score = 0
    best_sequence = None
    
    # Evolutionary parameters
    generations = 0
    max_generations = 300
    
    # Keep track of recent best scores for early stopping
    recent_scores = deque(maxlen=15)
    
    while time.time() - start_time < max_time and generations < max_generations:
        # Evaluate fitness (1/C₁)
        fitness_scores = []
        for seq in population:
            score = compute_autocorrelation_constant_fast(seq)
            fitness_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_sequence = seq.copy()
        
        # Track recent scores
        recent_scores.append(best_score)
        
        # Early stopping if no significant improvement
        if len(recent_scores) == 15:
            improvement = (recent_scores[-1] - recent_scores[0]) / recent_scores[0] if recent_scores[0] > 0 else 0
            if improvement < 0.00005:
                break
            
        # Selection - keep top 35% and apply tournament selection
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        elite_count = population_size // 3
        elite_indices = sorted_indices[:elite_count]
        elite_population = [population[i] for i in elite_indices]
        
        # Create new generation through mutation and crossover
        new_population = elite_population.copy()
        
        # Add diversity with mutated elites (higher mutation rate for exploration)
        for _ in range(population_size // 4):
            parent = random.choice(elite_population)
            child = mutate_sequence(parent, mutation_rate=0.25, max_mutation=0.4)
            new_population.append(child)
        
        # Add crossover combinations with better variety
        while len(new_population) < population_size - 15:
            parent1, parent2 = random.sample(elite_population, 2)
            # Use advanced crossover
            child = advanced_crossover(parent1, parent2)
            new_population.append(child)
        
        # Add some completely random sequences for exploration
        while len(new_population) < population_size:
            new_population.append([random.uniform(0.1, 10.0) for _ in range(random.randint(20, 100))])
        
        # Apply local search to some individuals with higher probability
        for i in range(0, len(new_population), 2):
            if random.random() < 0.6:
                # Try both types of local search for better results
                if random.random() < 0.5:
                    new_population[i] = local_search_improvement(new_population[i], iterations=25)
                else:
                    new_population[i] = hill_climbing_local_search(new_population[i], max_iterations=40)
        
        population = new_population
        generations += 1
    
    return best_sequence, best_score

def adaptive_local_search(sequence, max_time=10):
    """Adaptive local search that varies intensity based on progress."""
    start_time = time.time()
    
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Start with intensive local search
    iterations = 100
    for i in range(iterations):
        if time.time() - start_time > max_time:
            break
            
        # Vary the search intensity over time and use better strategies
        if i < iterations // 3:
            # Very intensive search early on
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.7, 1.3)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        elif i < 2 * iterations // 3:
            # Moderate search
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        else:
            # Lighter search towards end
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.95, 1.05)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        
        new_score = compute_autocorrelation_constant_fast(new_seq)
        if new_score > current_score:
            current_seq = new_seq
            current_score = new_score
    
    return current_seq

def hybrid_optimization(initial_sequence, max_time=15):
    """Hybrid optimization combining multiple techniques with enhanced strategies."""
    start_time = time.time()
    
    # First, try adaptive local search with more iterations
    improved_seq = adaptive_local_search(initial_sequence, max_time=max_time/2)
    best_score = compute_autocorrelation_constant_fast(improved_seq)
    
    # Then try simulated annealing
    if time.time() - start_time < max_time/2:
        sa_seq = simulated_annealing_optimization(improved_seq, max_time=max_time/3)
        sa_score = compute_autocorrelation_constant_fast(sa_seq)
        
        if sa_score > best_score:
            improved_seq = sa_seq
            best_score = sa_score
    
    # Apply multiple rounds of adaptive local search with better strategies
    for i in range(2):
        if time.time() - start_time < max_time * 0.8:
            # Try different local search strategies
            if random.random() < 0.6:
                local_seq = adaptive_local_search(improved_seq, max_time=max_time/5)
            else:
                local_seq = hill_climbing_local_search(improved_seq, max_iterations=35)
            local_score = compute_autocorrelation_constant_fast(local_seq)
            if local_score > best_score:
                improved_seq = local_seq
                best_score = local_score
    
    # Try gradient-free optimization on selected parameters with better bounds
    try:
        if time.time() - start_time < max_time * 0.9:
            # Try different optimization approaches on subsets
            subset_sizes = [min(25, len(initial_sequence)), min(40, len(initial_sequence))]
            
            for subset_size in subset_sizes:
                if subset_size <= 1:
                    continue
                    
                indices = sorted(random.sample(range(len(initial_sequence)), subset_size))
                
                def objective(x):
                    seq = initial_sequence.copy()
                    for i, idx in enumerate(indices):
                        if 0 <= idx < len(seq):
                            seq[idx] = max(0.01, x[i])
                    return -compute_autocorrelation_constant_fast(seq)
                
                # Start from current solution
                x0 = [initial_sequence[i] for i in indices]
                bounds = [(0.01, 1000.0)] * len(x0)
                
                # Try different optimization methods with better settings
                try:
                    # Nelder-Mead for simpler local refinement
                    result_nm = minimize(objective, x0, method='Nelder-Mead', 
                                       options={'maxiter': 80, 'adaptive': True})
                    
                    if hasattr(result_nm, 'x') and len(result_nm.x) == len(indices):
                        seq = initial_sequence.copy()
                        for i, idx in enumerate(indices):
                            if 0 <= idx < len(seq):
                                seq[idx] = max(0.01, result_nm.x[i])
                        candidate_score = compute_autocorrelation_constant_fast(seq)
                        if candidate_score > best_score:
                            improved_seq = seq
                            best_score = candidate_score
                            
                except Exception:
                    pass
                    
                # Also try with different parameter settings
                try:
                    # L-BFGS-B for potentially better results
                    result_lbfgs = minimize(objective, x0, method='L-BFGS-B', 
                                          bounds=bounds, options={'maxiter': 40})
                    
                    if hasattr(result_lbfgs, 'x') and len(result_lbfgs.x) == len(indices):
                        seq = initial_sequence.copy()
                        for i, idx in enumerate(indices):
                            if 0 <= idx < len(seq):
                                seq[idx] = max(0.01, result_lbfgs.x[i])
                        candidate_score = compute_autocorrelation_constant_fast(seq)
                        if candidate_score > best_score:
                            improved_seq = seq
                            best_score = candidate_score
                            
                except Exception:
                    pass
                    
    except Exception:
        pass
    
    # Final local search refinement
    if time.time() - start_time < max_time * 0.95:
        final_seq = local_search_improvement(improved_seq, iterations=30)
        final_score = compute_autocorrelation_constant_fast(final_seq)
        if final_score > best_score:
            improved_seq = final_seq
            best_score = final_score
    
    return improved_seq

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence."""
    # Try advanced evolutionary approach first with more time allocation
    sequence, score = advanced_evolutionary_search(max_time=30)
    
    # Refine with hybrid optimization
    if sequence is not None:
        refined_sequence = hybrid_optimization(sequence, max_time=20)
        refined_score = compute_autocorrelation_constant_fast(refined_sequence)
        
        # Return the better of the two
        if refined_score > score:
            return refined_sequence
        else:
            return sequence
    else:
        # Fallback to simple random search with better initialization
        return [random.uniform(0.1, 10.0) for _ in range(50)]

def generate_random_sequence(length_range=(10, 100), min_height=0.1, max_height=10.0):
    """Generate a random sequence with specified length range."""
    n = random.randint(*length_range)
    # Generate sequence with some randomness but keep it reasonable
    sequence = [random.uniform(min_height, max_height) for _ in range(n)]
    return sequence

def generate_step_sequence(length, height):
    """Generate a sequence with all elements equal to height."""
    return [height] * length

def generate_special_sequences():
    """Generate specialized sequences based on mathematical insights for better performance."""
    sequences = []
    
    # Golden ratio based sequences (known to perform well in such problems)
    golden_ratio = (1 + np.sqrt(5)) / 2
    for n in [20, 30, 50, 75, 100]:
        # Golden ratio decay
        golden_seq = [1.0 / (golden_ratio ** i) for i in range(n)]
        sequences.append(golden_seq)
        
        # Alternating golden ratios
        alt_golden = [1.0 if i % 2 == 0 else 1.0/golden_ratio for i in range(n)]
        sequences.append(alt_golden)
    
    # Exponential sequences with different bases
    for base in [1.2, 1.3, 1.4, 1.5]:
        exp_seq = [1.0 / (base ** i) for i in range(100)]
        sequences.append(exp_seq)
    
    # Concentrated energy sequences (better for minimizing max convolution)
    for n in [50, 100]:
        # Peak at beginning with rapid decay
        peak_seq = [10.0] + [1.0 / (1.5 ** i) for i in range(n-1)]
        sequences.append(peak_seq)
        
        # Two peaks
        two_peak = [0.1] * (n//3) + [5.0] + [0.1] * (n//3) + [10.0] + [0.1] * (n - 2*n//3 - 1)
        sequences.append(two_peak)
    
    # Sparse sequences (some zero values) - these often work well
    sparse_seq = [10.0 if i == 0 else 0.1 if i == 50 else 0.05 if i == 99 else 0.01 for i in range(100)]
    sequences.append(sparse_seq)
    
    # Smoothly varying sequences
    smooth_seq = [1.0 + 0.5 * np.sin(i * np.pi / 50) for i in range(100)]
    sequences.append(smooth_seq)
    
    # Sequences with strong concentration at start (good for minimizing max convolution)
    for n in [50, 100]:
        # Strong peak followed by exponential decay
        strong_peak = [100.0] + [1.0 / (1.2 ** i) for i in range(n-1)]
        sequences.append(strong_peak)
        
        # Multiple strong peaks
        multi_peak = [0.1] * (n//4) + [50.0] + [0.1] * (n//4) + [25.0] + [0.1] * (n//2)
        sequences.append(multi_peak)
    
    # More mathematical patterns
    # Harmonic sequences
    harmonic_seq = [1.0 / (i + 1) for i in range(100)]
    sequences.append(harmonic_seq)
    
    # Power law sequences
    power_seq = [1.0 / ((i + 1) ** 1.5) for i in range(100)]
    sequences.append(power_seq)
    
    # Logarithmic sequences
    log_seq = [1.0 / (np.log(i + 2)) for i in range(100)]
    sequences.append(log_seq)
    
    return sequences

def generate_better_initial_sequences():
    """Generate a diverse set of initial sequences for better exploration."""
    sequences = []
    
    # Add sequences of various lengths and structures
    for n in [10, 20, 30, 50, 75, 100]:
        # Uniform sequences
        sequences.append(generate_step_sequence(n, 1.0))
        
        # Some with varying heights
        seq = [random.uniform(0.1, 10.0) for _ in range(n)]
        sequences.append(seq)
        
        # Step-like sequences (like in known solutions)
        if n >= 5:
            step_seq = [1.0] * (n//2) + [0.5] * (n - n//2)
            sequences.append(step_seq)
            
        # Special structures based on mathematical intuition
        if n >= 10:
            # Geometric decay sequences
            geometric_seq = [1.0 / (1.5 ** i) for i in range(n)]
            sequences.append(geometric_seq)
            
            # Alternating sequences
            alternating_seq = [1.0 if i % 2 == 0 else 0.5 for i in range(n)]
            sequences.append(alternating_seq)
            
            # Peaks at center (more focused energy)
            peak_seq = [0.1] * n
            center = n // 2
            peak_seq[center] = 10.0
            if center + 1 < n:
                peak_seq[center + 1] = 5.0
            if center - 1 >= 0:
                peak_seq[center - 1] = 5.0
            sequences.append(peak_seq)
    
    # Add some high-quality known patterns
    # Pattern that often works well - concentrated at start
    sequences.append([10.0] + [0.1] * 99)
    sequences.append([1.0] * 50 + [0.1] * 50)
    
    # Add sequences that have been shown to work well in similar problems
    # Concentrated peak at beginning with gradual decay
    sequences.append([10.0, 5.0, 2.5, 1.0, 0.5] + [0.1] * 95)
    
    # Double peak structure
    double_peak = [0.1] * 20 + [5.0] + [0.1] * 20 + [10.0] + [0.1] * 58
    sequences.append(double_peak)
    
    # Exponential decay pattern
    exp_decay = [1.0 / (1.3 ** i) for i in range(100)]
    sequences.append(exp_decay)
    
    # Improved patterns based on known good solutions
    # High concentration at start
    high_concentration = [100.0] + [0.1] * 99
    sequences.append(high_concentration)
    
    # Balanced decay
    balanced_decay = [1.0] + [0.8] * 20 + [0.6] * 20 + [0.4] * 20 + [0.2] * 20 + [0.1] * 20
    sequences.append(balanced_decay)
    
    # Add special sequences
    sequences.extend(generate_special_sequences())
    
    # Add sequences with specific mathematical properties
    # Fibonacci-like sequences
    fib_seq = [1.0]
    for i in range(1, 100):
        fib_seq.append(fib_seq[i-1] + (fib_seq[i-1] if i >= 2 else 1))
    sequences.append([x/1000.0 for x in fib_seq])
    
    # Gaussian-like sequences
    gaussian_seq = [np.exp(-0.5 * ((i - 50) / 15)**2) for i in range(100)]
    sequences.append(gaussian_seq)
    
    return sequences

def mutate_sequence(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Mutate a sequence with adaptive mutation rate and better bounds."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Adaptive mutation: smaller changes for larger values
            change_factor = random.uniform(-max_mutation, max_mutation)
            mutated[i] = max(0.01, mutated[i] * (1 + change_factor))
            # Clip to reasonable bounds
            mutated[i] = min(1000.0, mutated[i])
    return mutated

def crossover_sequences(seq1, seq2):
    """Perform crossover between two sequences with better mixing."""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Use multiple crossover points for better diversity
    num_crossover_points = min(5, min_len//2)
    if num_crossover_points > 0:
        crossover_points = sorted(random.sample(range(1, min_len), num_crossover_points))
    else:
        crossover_points = [min_len//2] if min_len > 1 else []
    
    # Alternate between parents with better mixing strategy
    child = []
    last_point = 0
    for point in crossover_points:
        if len(child) % 2 == 0:
            child.extend(seq1[last_point:point])
        else:
            child.extend(seq2[last_point:point])
        last_point = point
    
    # Add remaining elements from the appropriate parent
    if len(child) % 2 == 0 and last_point < len(seq1):
        child.extend(seq1[last_point:])
    elif last_point < len(seq2):
        child.extend(seq2[last_point:])
    
    # Ensure we don't exceed the original length
    if len(child) > min_len:
        child = child[:min_len]
    elif len(child) < min_len:
        # Pad with elements from the longer parent
        longer_seq = seq1 if len(seq1) > len(seq2) else seq2
        while len(child) < min_len:
            child.append(longer_seq[len(child) % len(longer_seq)])
    
    return child

def advanced_crossover(seq1, seq2):
    """Advanced crossover with more sophisticated mixing strategies."""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Try different crossover strategies
    strategy = random.choice(['uniform', 'segment', 'alternating'])
    
    if strategy == 'uniform':
        # Uniform crossover - randomly select elements from either parent
        child = []
        for i in range(min(len(seq1), len(seq2))):
            if random.random() < 0.5:
                child.append(seq1[i])
            else:
                child.append(seq2[i])
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[len(seq2):])
        elif len(seq2) > len(seq1):
            child.extend(seq2[len(seq1):])
        return child
    
    elif strategy == 'segment':
        # Segment crossover - take segments from each parent
        min_len = min(len(seq1), len(seq2))
        if min_len < 2:
            return seq1 if len(seq1) > 0 else seq2
            
        # Split at random points
        split1 = random.randint(1, min_len - 1)
        split2 = random.randint(1, min_len - 1)
        
        child = []
        if random.random() < 0.5:
            child.extend(seq1[:split1])
            child.extend(seq2[split1:split2])
            child.extend(seq1[split2:])
        else:
            child.extend(seq2[:split1])
            child.extend(seq1[split1:split2])
            child.extend(seq2[split2:])
        
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[len(seq2):])
        elif len(seq2) > len(seq1):
            child.extend(seq2[len(seq1):])
        return child
    
    else:  # alternating
        # Alternating crossover - alternate between elements from parents
        min_len = min(len(seq1), len(seq2))
        child = []
        for i in range(min_len):
            if i % 2 == 0:
                child.append(seq1[i])
            else:
                child.append(seq2[i])
        
        # Extend with elements from longer parent
        if len(seq1) > len(seq2):
            child.extend(seq1[min_len:])
        elif len(seq2) > len(seq1):
            child.extend(seq2[min_len:])
        return child

def local_search_improvement(sequence, iterations=50):
    """Perform local search to improve the sequence with multiple strategies."""
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Try different local search strategies
    for i in range(iterations):
        # Strategy 1: Single element perturbation
        if random.random() < 0.7:
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            # Small multiplicative change with better bounds
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
            
            new_score = compute_autocorrelation_constant_fast(new_seq)
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
        else:
            # Strategy 2: Multiple element perturbation with adaptive strategy
            new_seq = current_seq.copy()
            num_changes = random.randint(1, min(5, len(new_seq)//4))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                # Use different mutation factors based on position and value
                if new_seq[idx] > 10.0:
                    factor = random.uniform(0.9, 1.1)
                elif new_seq[idx] > 1.0:
                    factor = random.uniform(0.85, 1.15)
                else:
                    factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
            
            new_score = compute_autocorrelation_constant_fast(new_seq)
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
    
    return current_seq

def hill_climbing_local_search(sequence, max_iterations=100):
    """More aggressive hill climbing with better neighborhood exploration."""
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    for i in range(max_iterations):
        # Generate neighbors with different strategies
        candidates = []
        
        # Strategy 1: Single element modification
        for _ in range(5):
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            # Larger changes for exploration
            factor = random.uniform(0.5, 2.0)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
            candidates.append(new_seq)
        
        # Strategy 2: Multiple element modification
        for _ in range(3):
            new_seq = current_seq.copy()
            num_changes = random.randint(1, min(10, len(new_seq)//2))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
            candidates.append(new_seq)
        
        # Strategy 3: Local averaging with nearby elements
        if len(current_seq) >= 3:
            for _ in range(2):
                new_seq = current_seq.copy()
                start_idx = random.randint(0, len(current_seq) - 3)
                # Average with neighbors
                avg_val = (current_seq[start_idx] + current_seq[start_idx+1] + current_seq[start_idx+2]) / 3
                new_seq[start_idx] = avg_val * random.uniform(0.8, 1.2)
                new_seq[start_idx+1] = avg_val * random.uniform(0.8, 1.2)
                new_seq[start_idx+2] = avg_val * random.uniform(0.8, 1.2)
                candidates.append(new_seq)
        
        # Evaluate all candidates
        best_candidate = None
        best_candidate_score = current_score
        
        for candidate in candidates:
            candidate_score = compute_autocorrelation_constant_fast(candidate)
            if candidate_score > best_candidate_score:
                best_candidate = candidate
                best_candidate_score = candidate_score
        
        # Accept the best improvement
        if best_candidate is not None:
            current_seq = best_candidate
            current_score = best_candidate_score
        else:
            # No improvement found, try random exploration occasionally
            if random.random() < 0.1:
                idx = random.randint(0, len(current_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                current_seq[idx] = max(0.01, current_seq[idx] * factor)
                current_seq[idx] = min(1000.0, current_seq[idx])
                current_score = compute_autocorrelation_constant_fast(current_seq)
    
    return current_seq

def simulated_annealing_optimization(initial_sequence, max_time=10):
    """Use simulated annealing for fine-tuning with better parameters."""
    start_time = time.time()
    
    current_seq = initial_sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Parameters for simulated annealing with better tuning
    temperature = 100.0
    cooling_rate = 0.98
    min_temperature = 0.001
    
    best_seq = current_seq.copy()
    best_score = current_score
    
    iteration = 0
    while time.time() - start_time < max_time and temperature > min_temperature:
        # Generate neighbor solution with adaptive strategy
        new_seq = current_seq.copy()
        
        # Choose between different neighborhood types
        if random.random() < 0.7:
            # Single element change (more local)
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        else:
            # Multiple element change (more global)
            num_changes = random.randint(1, min(5, len(new_seq)//3))
            for _ in range(num_changes):
                idx = random.randint(0, len(new_seq) - 1)
                factor = random.uniform(0.7, 1.3)
                new_seq[idx] = max(0.01, new_seq[idx] * factor)
                new_seq[idx] = min(1000.0, new_seq[idx])
        
        new_score = compute_autocorrelation_constant_fast(new_seq)
        
        # Accept or reject the new solution
        if new_score > current_score:
            current_seq = new_seq
            current_score = new_score
        else:
            # Accept with probability based on temperature and difference
            delta = new_score - current_score
            if delta < 0:
                # Even negative improvements might be accepted sometimes
                acceptance_prob = math.exp(delta / (temperature + 1e-10))
                if random.random() < acceptance_prob:
                    current_seq = new_seq
                    current_score = new_score
        
        # Update best solution
        if current_score > best_score:
            best_seq = current_seq.copy()
            best_score = current_score
        
        # Cool down with adaptive schedule
        if iteration % 10 == 0:
            temperature *= cooling_rate
        
        iteration += 1
    
    return best_seq

def advanced_evolutionary_search(max_time=40):
    """Improved evolutionary algorithm with better strategies and more sophisticated selection."""
    start_time = time.time()
    
    # Initial diverse population
    population_size = 150
    population = generate_better_initial_sequences()
    
    # Fill up population with random sequences
    while len(population) < population_size:
        population.append(generate_random_sequence())
    
    best_score = 0
    best_sequence = None
    
    # Evolutionary parameters
    generations = 0
    max_generations = 300
    
    # Keep track of recent best scores for early stopping
    recent_scores = deque(maxlen=15)
    
    while time.time() - start_time < max_time and generations < max_generations:
        # Evaluate fitness (1/C₁)
        fitness_scores = []
        for seq in population:
            score = compute_autocorrelation_constant_fast(seq)
            fitness_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_sequence = seq.copy()
        
        # Track recent scores
        recent_scores.append(best_score)
        
        # Early stopping if no significant improvement
        if len(recent_scores) == 15:
            improvement = (recent_scores[-1] - recent_scores[0]) / recent_scores[0] if recent_scores[0] > 0 else 0
            if improvement < 0.00005:
                break
            
        # Selection - keep top 35% and apply tournament selection
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        elite_count = population_size // 3
        elite_indices = sorted_indices[:elite_count]
        elite_population = [population[i] for i in elite_indices]
        
        # Create new generation through mutation and crossover
        new_population = elite_population.copy()
        
        # Add diversity with mutated elites (higher mutation rate for exploration)
        for _ in range(population_size // 4):
            parent = random.choice(elite_population)
            child = mutate_sequence(parent, mutation_rate=0.25, max_mutation=0.4)
            new_population.append(child)
        
        # Add crossover combinations with better variety
        while len(new_population) < population_size - 15:
            parent1, parent2 = random.sample(elite_population, 2)
            # Use advanced crossover
            child = advanced_crossover(parent1, parent2)
            new_population.append(child)
        
        # Add some completely random sequences for exploration
        while len(new_population) < population_size:
            new_population.append(generate_random_sequence())
        
        # Apply local search to some individuals with higher probability
        for i in range(0, len(new_population), 2):
            if random.random() < 0.6:
                # Try both types of local search for better results
                if random.random() < 0.5:
                    new_population[i] = local_search_improvement(new_population[i], iterations=25)
                else:
                    new_population[i] = hill_climbing_local_search(new_population[i], max_iterations=40)
        
        population = new_population
        generations += 1
    
    return best_sequence, best_score

def adaptive_local_search(sequence, max_time=10):
    """Adaptive local search that varies intensity based on progress."""
    start_time = time.time()
    
    current_seq = sequence.copy()
    current_score = compute_autocorrelation_constant_fast(current_seq)
    
    # Start with intensive local search
    iterations = 100
    for i in range(iterations):
        if time.time() - start_time > max_time:
            break
            
        # Vary the search intensity over time and use better strategies
        if i < iterations // 3:
            # Very intensive search early on
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.7, 1.3)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        elif i < 2 * iterations // 3:
            # Moderate search
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.8, 1.2)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        else:
            # Lighter search towards end
            new_seq = current_seq.copy()
            idx = random.randint(0, len(new_seq) - 1)
            factor = random.uniform(0.95, 1.05)
            new_seq[idx] = max(0.01, new_seq[idx] * factor)
            new_seq[idx] = min(1000.0, new_seq[idx])
        
        new_score = compute_autocorrelation_constant_fast(new_seq)
        if new_score > current_score:
            current_seq = new_seq
            current_score = new_score
    
    return current_seq

def hybrid_optimization(initial_sequence, max_time=15):
    """Hybrid optimization combining multiple techniques with enhanced strategies."""
    start_time = time.time()
    
    # First, try adaptive local search with more iterations
    improved_seq = adaptive_local_search(initial_sequence, max_time=max_time/2)
    best_score = compute_autocorrelation_constant_fast(improved_seq)
    
    # Then try simulated annealing
    if time.time() - start_time < max_time/2:
        sa_seq = simulated_annealing_optimization(improved_seq, max_time=max_time/3)
        sa_score = compute_autocorrelation_constant_fast(sa_seq)
        
        if sa_score > best_score:
            improved_seq = sa_seq
            best_score = sa_score
    
    # Apply multiple rounds of adaptive local search with better strategies
    for i in range(2):
        if time.time() - start_time < max_time * 0.8:
            # Try different local search strategies
            if random.random() < 0.6:
                local_seq = adaptive_local_search(improved_seq, max_time=max_time/5)
            else:
                local_seq = hill_climbing_local_search(improved_seq, max_iterations=35)
            local_score = compute_autocorrelation_constant_fast(local_seq)
            if local_score > best_score:
                improved_seq = local_seq
                best_score = local_score
    
    # Try gradient-free optimization on selected parameters with better bounds
    try:
        if time.time() - start_time < max_time * 0.9:
            # Try different optimization approaches on subsets
            subset_sizes = [min(25, len(initial_sequence)), min(40, len(initial_sequence))]
            
            for subset_size in subset_sizes:
                if subset_size <= 1:
                    continue
                    
                indices = sorted(random.sample(range(len(initial_sequence)), subset_size))
                
                def objective(x):
                    seq = initial_sequence.copy()
                    for i, idx in enumerate(indices):
                        if 0 <= idx < len(seq):
                            seq[idx] = max(0.01, x[i])
                    return -compute_autocorrelation_constant_fast(seq)
                
                # Start from current solution
                x0 = [initial_sequence[i] for i in indices]
                bounds = [(0.01, 1000.0)] * len(x0)
                
                # Try different optimization methods with better settings
                try:
                    # Nelder-Mead for simpler local refinement
                    result_nm = minimize(objective, x0, method='Nelder-Mead', 
                                       options={'maxiter': 80, 'adaptive': True})
                    
                    if hasattr(result_nm, 'x') and len(result_nm.x) == len(indices):
                        seq = initial_sequence.copy()
                        for i, idx in enumerate(indices):
                            if 0 <= idx < len(seq):
                                seq[idx] = max(0.01, result_nm.x[i])
                        candidate_score = compute_autocorrelation_constant_fast(seq)
                        if candidate_score > best_score:
                            improved_seq = seq
                            best_score = candidate_score
                            
                except Exception:
                    pass
                    
                # Also try with different parameter settings
                try:
                    # L-BFGS-B for potentially better results
                    result_lbfgs = minimize(objective, x0, method='L-BFGS-B', 
                                          bounds=bounds, options={'maxiter': 40})
                    
                    if hasattr(result_lbfgs, 'x') and len(result_lbfgs.x) == len(indices):
                        seq = initial_sequence.copy()
                        for i, idx in enumerate(indices):
                            if 0 <= idx < len(seq):
                                seq[idx] = max(0.01, result_lbfgs.x[i])
                        candidate_score = compute_autocorrelation_constant_fast(seq)
                        if candidate_score > best_score:
                            improved_seq = seq
                            best_score = candidate_score
                            
                except Exception:
                    pass
                    
    except Exception:
        pass
    
    # Final local search refinement
    if time.time() - start_time < max_time * 0.95:
        final_seq = local_search_improvement(improved_seq, iterations=30)
        final_score = compute_autocorrelation_constant_fast(final_seq)
        if final_score > best_score:
            improved_seq = final_seq
            best_score = final_score
    
    return improved_seq

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence."""
    # Try advanced evolutionary approach first with more time allocation
    sequence, score = advanced_evolutionary_search(max_time=30)
    
    # Refine with hybrid optimization
    if sequence is not None:
        refined_sequence = hybrid_optimization(sequence, max_time=20)
        refined_score = compute_autocorrelation_constant_fast(refined_sequence)
        
        # Return the better of the two
        if refined_score > score:
            return refined_sequence
        else:
            return sequence
    else:
        # Fallback to simple random search with better initialization
        return generate_random_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
