# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy import signal
from typing import List, Tuple
import time
import math
from scipy.optimize import differential_evolution
import warnings
from numba import jit
from scipy.fft import fft, ifft
from scipy.signal import convolve

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@jit(nopython=True)
def fast_convolution_max(a):
    """Fast computation of maximum convolution value using Numba"""
    n = len(a)
    max_val = 0.0
    
    # Compute convolution manually for better control
    # For a sequence of length n, convolution has length 2n-1
    conv_length = 2 * n - 1
    conv = np.zeros(conv_length)
    
    # Compute convolution using nested loop (can be optimized further)
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < conv_length:
                conv[idx] += a[i] * a[j]
    
    # Find maximum
    for i in range(conv_length):
        if conv[i] > max_val:
            max_val = conv[i]
    
    return max_val

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    Returns (C₁, 1/C₁) where C₁ = 2n * max(convolution) / (sum(sequence))²
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Convert to numpy array
    a = np.array(sequence, dtype=np.float64)
    
    # Use FFT for efficiency when possible, fallback to manual for small arrays
    if len(a) > 100:
        # Compute convolution using FFT for efficiency
        # Use scipy.signal.convolve for better numerical stability
        conv = convolve(a, a[::-1], mode='full')
        max_conv = np.max(conv)
    else:
        # For small arrays, use manual computation for accuracy
        max_conv = fast_convolution_max(a)
    
    # Sum of elements
    sum_a = np.sum(a)
    
    # Avoid division by zero
    if sum_a < 1e-10:
        return float('inf'), 0.0
    
    n = len(a)
    
    # Compute C₁
    C1 = (2 * n * max_conv) / (sum_a ** 2)
    
    # Return both C₁ and its reciprocal
    return C1, 1.0 / C1

def compute_autocorrelation_constant_optimized(sequence: List[float]) -> Tuple[float, float]:
    """
    Optimized version using explicit convolution calculation that's more stable
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Convert to numpy array
    a = np.array(sequence, dtype=np.float64)
    
    # For very small sequences, use direct computation
    if len(a) <= 50:
        # Direct convolution computation for small sequences
        n = len(a)
        conv = np.zeros(2 * n - 1)
        for i in range(n):
            for j in range(n):
                conv[i + j] += a[i] * a[j]
        max_conv = np.max(conv)
    else:
        # Use FFT for larger sequences with scipy.signal.convolve for better stability
        conv = convolve(a, a[::-1], mode='full')
        max_conv = np.max(conv)
    
    # Sum of elements
    sum_a = np.sum(a)
    
    # Avoid division by zero
    if sum_a < 1e-10:
        return float('inf'), 0.0
    
    n = len(a)
    
    # Compute C₁
    C1 = (2 * n * max_conv) / (sum_a ** 2)
    
    # Return both C₁ and its reciprocal
    return C1, 1.0 / C1

def evaluate_sequence(sequence: List[float]) -> float:
    """
    Evaluate a sequence by returning 1/C₁ (higher is better).
    """
    try:
        # Filter out invalid sequences
        if len(sequence) == 0:
            return 0.0
        if sum(sequence) < 0.01:
            return 0.0
            
        _, inv_c1 = compute_autocorrelation_constant(sequence)
        return inv_c1 if not math.isnan(inv_c1) and not math.isinf(inv_c1) else 0.0
    except Exception:
        return 0.0

def generate_random_sequence(n_steps: int) -> List[float]:
    """
    Generate a random step function with specified number of steps.
    """
    # Generate random heights between 0.01 and 1000
    heights = [random.uniform(0.01, 1000.0) for _ in range(n_steps)]
    return heights

def generate_pattern_sequence(n_steps: int, pattern_type: str = "uniform") -> List[float]:
    """
    Generate sequences with specific patterns that might perform well.
    """
    if pattern_type == "uniform":
        return [1.0] * n_steps
    elif pattern_type == "exponential":
        # Exponentially decaying pattern with better decay rates
        return [1.0 * (0.97 ** i) for i in range(n_steps)]
    elif pattern_type == "step":
        # Step-like pattern with high first element
        return [10.0] + [1.0] * (n_steps - 1)
    elif pattern_type == "two_level":
        # Two-level pattern
        half = n_steps // 2
        return [1.0] * half + [0.5] * (n_steps - half)
    elif pattern_type == "linear_decrease":
        # Linearly decreasing pattern
        return [1.0 - i/(n_steps-1) for i in range(n_steps)]
    elif pattern_type == "gaussian":
        # Gaussian-like pattern (centered)
        center = n_steps // 2
        std = n_steps / 6.0
        return [math.exp(-((i - center)**2) / (2 * std**2)) for i in range(n_steps)]
    elif pattern_type == "inverse_linear":
        # Inverse linear pattern
        return [1.0 / (1.0 + i) for i in range(n_steps)]
    elif pattern_type == "logarithmic":
        # Logarithmic pattern
        return [math.log(i + 2) for i in range(n_steps)]
    elif pattern_type == "quadratic":
        # Quadratic pattern
        return [(i + 1) ** 0.5 for i in range(n_steps)]
    elif pattern_type == "sinusoidal":
        # Sinusoidal pattern
        return [0.5 + 0.5 * math.sin(2 * math.pi * i / n_steps) for i in range(n_steps)]
    elif pattern_type == "power_decay":
        # Power law decay pattern
        return [1.0 / ((i + 1) ** 1.5) for i in range(n_steps)]
    elif pattern_type == "modified_gaussian":
        # Modified gaussian with sharper peak
        center = n_steps // 2
        std = n_steps / 8.0
        return [math.exp(-((i - center)**2) / (2 * std**2)) for i in range(n_steps)]
    elif pattern_type == "bimodal":
        # Bimodal pattern - two peaks
        mid = n_steps // 2
        return [math.exp(-((i - mid/2)**2) / 100) + 0.5 * math.exp(-((i - 3*mid/2)**2) / 100) for i in range(n_steps)]
    elif pattern_type == "hyperbolic_tangent":
        # Hyperbolic tangent pattern (smooth, bounded)
        center = n_steps // 2
        return [math.tanh((i - center) / (n_steps/6)) for i in range(n_steps)]
    elif pattern_type == "golden_ratio":
        # Golden ratio pattern: phi^i where phi = (1+sqrt(5))/2
        phi = (1 + 5**0.5) / 2
        return [phi**i for i in range(n_steps)]
    elif pattern_type == "fibonacci":
        # Fibonacci-like sequence normalized
        fib_seq = [1.0, 1.0]
        for i in range(2, n_steps):
            fib_seq.append(fib_seq[i-1] + fib_seq[i-2])
        max_val = max(fib_seq)
        if max_val > 0:
            return [x/max_val for x in fib_seq]
        else:
            return [1.0] * n_steps
    elif pattern_type == "sine_wave":
        # Sine wave with increasing frequency
        return [math.sin(2 * math.pi * i / n_steps * (1 + i/n_steps)) for i in range(n_steps)]
    elif pattern_type == "polynomial":
        # Polynomial decay pattern
        return [1.0 / (1.0 + i**1.2) for i in range(n_steps)]
    elif pattern_type == "double_exponential":
        # Double exponential decay
        return [1.0 * (0.95 ** i) + 0.5 * (0.9 ** i) for i in range(n_steps)]
    elif pattern_type == "cubic_decay":
        # Cubic decay pattern
        return [1.0 / ((i + 1) ** 3) for i in range(n_steps)]
    elif pattern_type == "hyperbolic":
        # Hyperbolic pattern
        return [1.0 / math.sqrt(1 + i**2) for i in range(n_steps)]
    elif pattern_type == "asymmetric":
        # Asymmetric pattern - high at beginning, low at end
        return [1.0 + 0.5 * math.exp(-i/10) for i in range(n_steps)]
    elif pattern_type == "spike":
        # Spike pattern with single high value
        spike_pos = n_steps // 2
        result = [0.1] * n_steps
        result[spike_pos] = 10.0
        return result
    elif pattern_type == "bell":
        # Bell-shaped pattern with peak in center
        center = n_steps // 2
        return [math.exp(-((i - center)**2) / (2 * (n_steps/10)**2)) for i in range(n_steps)]
    elif pattern_type == "inverse_quadratic":
        # Inverse quadratic pattern
        return [1.0 / (1.0 + (i**2)/100) for i in range(n_steps)]
    elif pattern_type == "rational":
        # Rational decay pattern
        return [1.0 / (1.0 + i**1.8) for i in range(n_steps)]
    elif pattern_type == "double_peak":
        # Double peak pattern
        peak1 = n_steps // 3
        peak2 = 2 * n_steps // 3
        return [math.exp(-((i - peak1)**2) / 50) + 0.5 * math.exp(-((i - peak2)**2) / 50) for i in range(n_steps)]
    elif pattern_type == "smooth_step":
        # Smooth step pattern
        return [0.5 * (1 + math.tanh((i - n_steps/2) / (n_steps/10))) for i in range(n_steps)]
    elif pattern_type == "harmonic":
        # Harmonic-like pattern
        return [1.0 / (i + 1) for i in range(n_steps)]
    elif pattern_type == "square_root":
        # Square root decay
        return [1.0 / math.sqrt(i + 1) for i in range(n_steps)]
    elif pattern_type == "tanh_decay":
        # Tanh decay pattern
        return [math.tanh(i / (n_steps/2)) for i in range(n_steps)]
    elif pattern_type == "cosine":
        # Cosine pattern
        return [0.5 * (1 + math.cos(2 * math.pi * i / n_steps)) for i in range(n_steps)]
    elif pattern_type == "log_exp":
        # Log-exponential pattern
        return [math.log(1 + math.exp(i/10)) for i in range(n_steps)]
    elif pattern_type == "gaussian_peak":
        # Gaussian with peak at start
        return [math.exp(-((i - 0)**2) / (2 * (n_steps/10)**2)) for i in range(n_steps)]
    elif pattern_type == "double_bell":
        # Double bell pattern for better balance
        center1 = n_steps // 3
        center2 = 2 * n_steps // 3
        std = n_steps / 8.0
        return [math.exp(-((i - center1)**2) / (2 * std**2)) + 0.7 * math.exp(-((i - center2)**2) / (2 * std**2)) for i in range(n_steps)]
    elif pattern_type == "modified_power":
        # Modified power decay with different exponents
        return [1.0 / ((i + 1) ** 1.3) for i in range(n_steps)]
    elif pattern_type == "weighted_exponential":
        # Weighted exponential decay
        weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        if n_steps < 10:
            return [1.0 * (0.95 ** i) for i in range(n_steps)]
        else:
            return [weights[i % len(weights)] * (0.95 ** i) for i in range(n_steps)]
    else:
        return [1.0] * n_steps

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1, 
                   mutation_strength: float = 0.5) -> List[float]:
    """
    Mutate a sequence by randomly changing some heights.
    """
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply multiplicative mutation
            change_factor = random.uniform(1-mutation_strength, 1+mutation_strength)
            mutated[i] = max(0.01, mutated[i] * change_factor)
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """
    Perform crossover between two sequences.
    """
    if len(seq1) != len(seq2):
        # If lengths differ, use the shorter one
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    # Uniform crossover with some probability of inheritance from either parent
    child = []
    for i in range(len(seq1)):
        if random.random() < 0.5:
            child.append(seq1[i])
        else:
            child.append(seq2[i])
    
    return child

def improved_evolutionary_search(max_time_seconds: float = 60.0) -> List[float]:
    """
    Improved evolutionary algorithm with better strategies and convergence.
    """
    start_time = time.time()
    
    # Initial population with diverse patterns
    population_size = 300
    population = []
    
    # Generate initial population with different strategies
    for i in range(population_size):
        if i < 80:  # Random sequences
            n_steps = random.randint(30, 600)
            individual = generate_random_sequence(n_steps)
        elif i < 180:  # Pattern-based sequences
            n_steps = random.randint(30, 600)
            pattern_types = ["uniform", "exponential", "step", "two_level", "gaussian", 
                           "inverse_linear", "logarithmic", "quadratic", "sinusoidal",
                           "power_decay", "modified_gaussian", "bimodal", "hyperbolic_tangent",
                           "golden_ratio", "fibonacci", "sine_wave", "polynomial", 
                           "double_exponential", "cubic_decay", "hyperbolic", "asymmetric",
                           "spike", "bell", "inverse_quadratic", "rational", "double_peak",
                           "smooth_step", "harmonic", "square_root", "tanh_decay", "cosine",
                           "log_exp", "gaussian_peak", "double_bell", "modified_power",
                           "weighted_exponential"]
            pattern = random.choice(pattern_types)
            individual = generate_pattern_sequence(n_steps, pattern)
        else:  # Mix of both
            n_steps = random.randint(30, 600)
            if random.random() < 0.5:
                individual = generate_random_sequence(n_steps)
            else:
                pattern_types = ["uniform", "exponential", "step", "two_level", "gaussian", 
                               "inverse_linear", "logarithmic", "quadratic", "sinusoidal",
                               "power_decay", "modified_gaussian", "bimodal", "hyperbolic_tangent",
                               "golden_ratio", "fibonacci", "sine_wave", "polynomial", 
                               "double_exponential", "cubic_decay", "hyperbolic", "asymmetric",
                               "spike", "bell", "inverse_quadratic", "rational", "double_peak",
                               "smooth_step", "harmonic", "square_root", "tanh_decay", "cosine",
                               "log_exp", "gaussian_peak", "double_bell", "modified_power",
                               "weighted_exponential"]
                pattern = random.choice(pattern_types)
                individual = generate_pattern_sequence(n_steps, pattern)
        
        population.append(individual)
    
    best_individual = None
    best_score = 0.0
    
    generation = 0
    while time.time() - start_time < max_time_seconds:
        generation += 1
        
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            score = evaluate_sequence(individual)
            fitness_scores.append(score)
            
            if score > best_score and score > 0.0:
                best_score = score
                best_individual = individual.copy()
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_individuals = [population[i] for i in sorted_indices[:population_size//4]]
        
        # Create new population through selection, crossover, and mutation
        new_population = []
        
        # Keep top individuals (elitism)
        new_population.extend(top_individuals)
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection with size 3
            tournament_size = 3
            parent1 = random.choice(top_individuals)
            parent2 = random.choice(top_individuals)
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Adaptive mutation with better parameters
            mutation_rate = max(0.05, 0.3 - generation * 0.001)  # Decreasing rate
            mutation_strength = max(0.1, 0.7 - generation * 0.0015)  # Decreasing strength
            child = mutate_sequence(child, mutation_rate, mutation_strength)
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Occasionally add some diversity
        if generation % 3 == 0:
            for i in range(min(30, population_size // 8)):
                if random.random() < 0.7:  # Most of the time use patterns
                    n_steps = random.randint(30, 600)
                    pattern_types = ["uniform", "exponential", "step", "two_level", "gaussian", 
                                   "inverse_linear", "logarithmic", "quadratic", "sinusoidal",
                                   "power_decay", "modified_gaussian", "bimodal", "hyperbolic_tangent",
                                   "golden_ratio", "fibonacci", "sine_wave", "polynomial", 
                                   "double_exponential", "cubic_decay", "hyperbolic", "asymmetric",
                                   "spike", "bell", "inverse_quadratic", "rational", "double_peak",
                                   "smooth_step", "harmonic", "square_root", "tanh_decay", "cosine",
                                   "log_exp", "gaussian_peak", "double_bell", "modified_power",
                                   "weighted_exponential"]
                    pattern = random.choice(pattern_types)
                    population[random.randint(0, len(population)-1)] = generate_pattern_sequence(n_steps, pattern)
                else:  # Occasionally add random
                    n_steps = random.randint(30, 600)
                    population[random.randint(0, len(population)-1)] = generate_random_sequence(n_steps)
    
    return best_individual if best_individual is not None else []

def advanced_grid_search_approach() -> List[float]:
    """
    Advanced grid search focusing on promising mathematical patterns.
    """
    best_sequence = []
    best_inv_c1 = 0.0
    
    # Focus on lengths that tend to produce better results based on research
    length_candidates = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 
                        110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300,
                        320, 340, 360, 380, 400, 450, 500, 550, 600]
    
    # Enhanced pattern types that often perform well plus new ones
    pattern_types = [
        "uniform", "exponential", "step", "two_level", 
        "gaussian", "inverse_linear", "logarithmic", "quadratic",
        "power_decay", "modified_gaussian", "bimodal", "hyperbolic_tangent",
        "golden_ratio", "fibonacci", "sine_wave", "polynomial", "double_exponential",
        "cubic_decay", "hyperbolic", "asymmetric", "spike", "bell",
        "inverse_quadratic", "rational", "double_peak", "smooth_step", "harmonic",
        "square_root", "tanh_decay", "cosine", "log_exp", "gaussian_peak",
        "double_bell", "modified_power", "weighted_exponential"
    ]
    
    # Try combinations of patterns and parameters
    for n_steps in length_candidates:
        # Try multiple pattern variations
        for pattern_type in pattern_types:
            # Generate base pattern
            base_pattern = generate_pattern_sequence(n_steps, pattern_type)
            
            # Try different scaling factors
            scales = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
            for scale in scales:
                scaled_pattern = [h * scale for h in base_pattern]
                
                # Ensure we have the right number of steps
                if len(scaled_pattern) < n_steps:
                    scaled_pattern.extend([1.0] * (n_steps - len(scaled_pattern)))
                elif len(scaled_pattern) > n_steps:
                    scaled_pattern = scaled_pattern[:n_steps]
                
                # Evaluate
                inv_c1 = evaluate_sequence(scaled_pattern)
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = scaled_pattern.copy()
    
    # Also try some mathematical sequences that are known to work well
    # Fibonacci-like sequences with better normalization
    fib_lengths = [25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155]
    for n in fib_lengths:
        # Generate fibonacci-like sequence (starting with 1, 1)
        fib_seq = [1.0, 1.0]
        for i in range(2, n):
            fib_seq.append(fib_seq[i-1] + fib_seq[i-2])
        
        # Normalize by the maximum value to keep reasonable magnitudes
        max_val = max(fib_seq)
        if max_val > 0:
            fib_seq = [x/max_val for x in fib_seq]
        
        # Evaluate
        inv_c1 = evaluate_sequence(fib_seq)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = fib_seq.copy()
    
    # Try golden ratio patterns with various lengths
    golden_ratios = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
    for n in golden_ratios:
        # Golden ratio pattern: phi^i where phi = (1+sqrt(5))/2
        phi = (1 + 5**0.5) / 2
        golden_pattern = [phi**i for i in range(n)]
        # Normalize
        max_val = max(golden_pattern)
        if max_val > 0:
            golden_pattern = [x/max_val for x in golden_pattern]
        
        # Evaluate
        inv_c1 = evaluate_sequence(golden_pattern)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = golden_pattern.copy()
    
    # Try more specialized mathematical patterns
    # Hyperbolic tangent pattern (smooth, bounded)
    hyperbolic_lengths = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    for n in hyperbolic_lengths:
        # Hyperbolic tangent pattern
        # This creates a smooth, symmetric pattern centered around middle
        center = n // 2
        hyperbolic_pattern = [math.tanh((i - center) / (n/6)) for i in range(n)]
        # Normalize to [0.01, 1000] range
        min_val, max_val = min(hyperbolic_pattern), max(hyperbolic_pattern)
        if max_val > min_val:
            hyperbolic_pattern = [(x - min_val) / (max_val - min_val) * 999.99 + 0.01 for x in hyperbolic_pattern]
        else:
            hyperbolic_pattern = [1.0] * n
        
        # Evaluate
        inv_c1 = evaluate_sequence(hyperbolic_pattern)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = hyperbolic_pattern.copy()
    
    # Try sine wave patterns with varying frequencies
    sine_lengths = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    for n in sine_lengths:
        sine_pattern = [math.sin(2 * math.pi * i / n * (1 + i/n)) for i in range(n)]
        # Normalize to [0.01, 1000] range
        min_val, max_val = min(sine_pattern), max(sine_pattern)
        if max_val > min_val:
            sine_pattern = [(x - min_val) / (max_val - min_val) * 999.99 + 0.01 for x in sine_pattern]
        else:
            sine_pattern = [1.0] * n
        
        # Evaluate
        inv_c1 = evaluate_sequence(sine_pattern)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sine_pattern.copy()
    
    # Try additional promising patterns
    # Specific exponential decay with different rates
    exp_rates = [0.95, 0.96, 0.97, 0.98, 0.99]
    for rate in exp_rates:
        for n in [80, 100, 120, 140, 160, 180, 200]:
            exp_pattern = [1.0 * (rate ** i) for i in range(n)]
            # Normalize to reasonable range
            max_val = max(exp_pattern)
            if max_val > 0:
                exp_pattern = [x/max_val * 1000 for x in exp_pattern]
            
            inv_c1 = evaluate_sequence(exp_pattern)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = exp_pattern.copy()
    
    # Try specialized patterns that work well in practice
    # Bell-shaped patterns with specific parameters
    bell_lengths = [100, 120, 140, 160, 180, 200]
    for n in bell_lengths:
        # Bell shape with adjustable width
        center = n // 2
        std_dev = n / 8.0
        bell_pattern = [math.exp(-((i - center)**2) / (2 * std_dev**2)) for i in range(n)]
        # Normalize
        max_val = max(bell_pattern)
        if max_val > 0:
            bell_pattern = [x/max_val * 1000 for x in bell_pattern]
        
        inv_c1 = evaluate_sequence(bell_pattern)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = bell_pattern.copy()
    
    # Try double exponential decay patterns
    double_exp_lengths = [100, 150, 200, 250, 300]
    for n in double_exp_lengths:
        double_exp_pattern = [1.0 * (0.95 ** i) + 0.5 * (0.9 ** i) for i in range(n)]
        # Normalize
        max_val = max(double_exp_pattern)
        if max_val > 0:
            double_exp_pattern = [x/max_val * 1000 for x in double_exp_pattern]
        
        inv_c1 = evaluate_sequence(double_exp_pattern)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = double_exp_pattern.copy()
    
    return best_sequence

def local_optimization_refinement(initial_sequence: List[float], max_time_seconds: float = 10.0) -> List[float]:
    """
    Refine a sequence using local optimization techniques.
    """
    start_time = time.time()
    current_sequence = initial_sequence.copy()
    current_score = evaluate_sequence(current_sequence)
    
    # Try adaptive hill climbing approach
    iteration = 0
    step_size = 0.1
    max_step_size = 0.5
    
    while time.time() - start_time < max_time_seconds and iteration < 2000:
        iteration += 1
        
        # Make small random perturbations
        candidate = current_sequence.copy()
        # Modify about 15% of elements
        num_modifications = max(1, len(candidate) // 7)
        indices_to_modify = random.sample(range(len(candidate)), num_modifications)
        
        for i in indices_to_modify:
            # Adaptive step size that decreases over time
            current_step = min(step_size * (1.0 - iteration/2000.0) + 0.05, max_step_size)
            # Small change
            change_factor = random.uniform(1-current_step, 1+current_step)
            candidate[i] = max(0.01, candidate[i] * change_factor)
        
        # Evaluate candidate
        candidate_score = evaluate_sequence(candidate)
        if candidate_score > current_score:
            current_sequence = candidate
            current_score = candidate_score
            # Reset step size when improvement is made
            step_size = 0.1
        else:
            # Occasionally accept worse solutions to escape local optima
            if random.random() < 0.01:
                current_sequence = candidate
                current_score = candidate_score
    
    return current_sequence

def hybrid_search_approach(max_time_seconds: float = 60.0) -> List[float]:
    """
    Hybrid approach combining multiple strategies.
    """
    start_time = time.time()
    best_sequence = []
    best_inv_c1 = 0.0
    
    # Strategy 1: Improved evolutionary search (most promising)
    try:
        evol_seq = improved_evolutionary_search(max_time_seconds=max_time_seconds * 0.4)
        evol_score = evaluate_sequence(evol_seq)
        if evol_score > best_inv_c1:
            best_inv_c1 = evol_score
            best_sequence = evol_seq.copy()
    except Exception as e:
        pass
    
    # Strategy 2: Advanced grid search (explores known good patterns)
    try:
        grid_seq = advanced_grid_search_approach()
        grid_score = evaluate_sequence(grid_seq)
        if grid_score > best_inv_c1:
            best_inv_c1 = grid_score
            best_sequence = grid_seq.copy()
    except Exception as e:
        pass
    
    # Strategy 3: Local optimization around promising points
    if best_sequence and len(best_sequence) > 0:
        try:
            # Try local refinement
            refined_seq = local_optimization_refinement(best_sequence, max_time_seconds * 0.2)
            refined_score = evaluate_sequence(refined_seq)
            if refined_score > best_inv_c1:
                best_inv_c1 = refined_score
                best_sequence = refined_seq.copy()
        except Exception as e:
            pass
    
    # Strategy 4: Final exploration of promising regions
    if best_sequence and len(best_sequence) > 0 and time.time() - start_time < max_time_seconds * 0.7:
        try:
            # Add some randomness to explore nearby solutions
            for _ in range(100):
                if time.time() - start_time > max_time_seconds:
                    break
                    
                # Slightly perturb the best sequence
                mutated = best_sequence.copy()
                for i in range(len(mutated)):
                    if random.random() < 0.15:  # 15% chance to modify
                        change_factor = random.uniform(0.85, 1.15)
                        mutated[i] = max(0.01, mutated[i] * change_factor)
                
                score = evaluate_sequence(mutated)
                if score > best_inv_c1:
                    best_inv_c1 = score
                    best_sequence = mutated.copy()
        except Exception as e:
            pass
    
    # Strategy 5: Additional focused search on promising lengths
    if time.time() - start_time < max_time_seconds * 0.8:
        try:
            # Try specific promising lengths and patterns
            promising_lengths = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
            for n in promising_lengths:
                # Try exponential decay with different decay rates
                for decay_rate in [0.95, 0.96, 0.97, 0.98, 0.99]:
                    exp_pattern = [1.0 * (decay_rate ** i) for i in range(n)]
                    # Normalize to reasonable range
                    max_val = max(exp_pattern)
                    if max_val > 0:
                        exp_pattern = [x/max_val * 1000 for x in exp_pattern]
                    
                    score = evaluate_sequence(exp_pattern)
                    if score > best_inv_c1:
                        best_inv_c1 = score
                        best_sequence = exp_pattern.copy()
        except Exception as e:
            pass
    
    # Strategy 6: Specialized mathematical sequences
    if time.time() - start_time < max_time_seconds * 0.9:
        try:
            # Try specific mathematical sequences that work well
            # Double exponential decay with optimized parameters
            for n in [120, 150, 180, 200, 250, 300]:
                # Use a combination of two different exponential decays
                double_exp_pattern = [1.0 * (0.95 ** i) + 0.3 * (0.9 ** i) for i in range(n)]
                # Normalize to keep reasonable values
                max_val = max(double_exp_pattern)
                if max_val > 0:
                    double_exp_pattern = [x/max_val * 1000 for x in double_exp_pattern]
                
                score = evaluate_sequence(double_exp_pattern)
                if score > best_inv_c1:
                    best_inv_c1 = score
                    best_sequence = double_exp_pattern.copy()
        except Exception as e:
            pass
    
    # Strategy 7: Explore more specialized mathematical patterns
    if time.time() - start_time < max_time_seconds * 0.95:
        try:
            # Try bell-shaped patterns with different parameters
            for n in [100, 120, 140, 160, 180, 200]:
                # Bell shape with adjustable width
                center = n // 2
                std_dev = n / 6.0
                bell_pattern = [math.exp(-((i - center)**2) / (2 * std_dev**2)) for i in range(n)]
                # Normalize to keep reasonable values
                max_val = max(bell_pattern)
                if max_val > 0:
                    bell_pattern = [x/max_val * 1000 for x in bell_pattern]
                
                score = evaluate_sequence(bell_pattern)
                if score > best_inv_c1:
                    best_inv_c1 = score
                    best_sequence = bell_pattern.copy()
        except Exception as e:
            pass
    
    # Strategy 8: Test carefully crafted sequences
    if time.time() - start_time < max_time_seconds * 0.98:
        try:
            # Try specific patterns that have been found to work well in related problems
            # A sequence designed to minimize the maximum convolution value
            test_patterns = [
                # Modified exponential decay
                [1.0 * (0.96 ** i) for i in range(100)],
                # Combination of two decays
                [1.0 * (0.95 ** i) + 0.5 * (0.9 ** i) for i in range(120)],
                # Smooth bell-like pattern with sharp peak
                [math.exp(-((i - 50)**2) / 100) for i in range(100)],
                # Logarithmic decay with slight increase at end
                [math.log(i + 2) + 0.01*i for i in range(100)],
                # Double bell pattern
                [math.exp(-((i - 33)**2) / 100) + 0.7 * math.exp(-((i - 66)**2) / 100) for i in range(100)]
            ]
            
            for pattern in test_patterns:
                # Normalize to reasonable range
                max_val = max(pattern)
                if max_val > 0:
                    pattern = [x/max_val * 1000 for x in pattern]
                
                score = evaluate_sequence(pattern)
                if score > best_inv_c1:
                    best_inv_c1 = score
                    best_sequence = pattern.copy()
        except Exception as e:
            pass
    
    # Strategy 9: Optimization of best found sequence using gradient-free methods
    if time.time() - start_time < max_time_seconds * 0.99:
        try:
            # Try a more sophisticated local search with better neighborhood exploration
            if best_sequence and len(best_sequence) > 0:
                # Try to improve by exploring better variants of the current best
                current_best = best_sequence.copy()
                current_score = evaluate_sequence(current_best)
                
                # Try different patterns with similar structure but adjusted parameters
                for _ in range(50):
                    if time.time() - start_time > max_time_seconds:
                        break
                        
                    # Try variation of best sequence
                    variant = current_best.copy()
                    # Adjust a few elements with more controlled changes
                    num_changes = max(1, len(variant) // 10)
                    indices = random.sample(range(len(variant)), num_changes)
                    for i in indices:
                        # Smaller, more precise mutations
                        factor = random.uniform(0.9, 1.1)
                        variant[i] = max(0.01, variant[i] * factor)
                    
                    score = evaluate_sequence(variant)
                    if score > current_score:
                        current_best = variant
                        current_score = score
                
                if current_score > best_inv_c1:
                    best_inv_c1 = current_score
                    best_sequence = current_best
        except Exception as e:
            pass
    
    return best_sequence

def search_for_best_sequence() -> List[float]:
    """
    Main search function using hybrid optimization strategies.
    """
    return hybrid_search_approach(max_time_seconds=60.0)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
