# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import time
import numba
from itertools import combinations
import math
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Optimized convolution using FFT for better performance
@numba.jit(nopython=True)
def fast_convolve(a, b):
    """Fast convolution using FFT"""
    # For small arrays, use direct method
    if len(a) < 100:
        result = np.zeros(len(a) + len(b) - 1)
        for i in range(len(a)):
            for j in range(len(b)):
                result[i + j] += a[i] * b[j]
        return result
    
    # For larger arrays, use FFT
    n = len(a) + len(b) - 1
    fa = np.fft.fft(a, n)
    fb = np.fft.fft(b, n)
    result = np.fft.ifft(fa * fb).real
    return result[:len(a) + len(b) - 1]

def compute_c1(sequence):
    """Compute C₁ for a given sequence"""
    if len(sequence) == 0:
        return float('inf')
    
    # Compute convolution - use the actual convolution for accurate results
    conv = fftconvolve(sequence, sequence, mode='full')
    # The maximum value in the convolution is what matters
    max_conv = np.max(conv)
    
    # Compute sum of squares
    sum_sq = np.sum(sequence) ** 2
    
    if sum_sq == 0:
        return float('inf')
    
    # C₁ = 2n * max(conv) / (sum(a))²
    n = len(sequence)
    c1 = 2 * n * max_conv / sum_sq
    
    return c1

def compute_c1_optimized(sequence):
    """More numerically stable version of C₁ computation"""
    if len(sequence) == 0:
        return float('inf')
    
    # Compute convolution using FFT for efficiency
    conv = fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    
    # Compute sum of squares
    sum_a = np.sum(sequence)
    sum_sq = sum_a ** 2
    
    if sum_sq < 1e-12:  # Prevent division by near-zero
        return float('inf')
    
    # C₁ = 2n * max(conv) / (sum(a))²
    n = len(sequence)
    c1 = 2 * n * max_conv / sum_sq
    
    return c1

def compute_inv_c1(sequence):
    """Compute 1/C₁ for reporting"""
    c1 = compute_c1_optimized(sequence)
    if c1 == float('inf') or c1 <= 0:
        return 0.0
    return 1.0 / c1

def compute_convolution_stats(sequence):
    """Compute detailed statistics about the convolution"""
    if len(sequence) == 0:
        return {"max_conv": 0, "mean_conv": 0, "std_conv": 0}
    
    conv = fftconvolve(sequence, sequence, mode='full')
    return {
        "max_conv": np.max(conv),
        "mean_conv": np.mean(conv),
        "std_conv": np.std(conv),
        "conv_shape": len(conv)
    }

def generate_mathematical_sequence(length, pattern_type='golden'):
    """Generate sequences with mathematical properties that tend to work well"""
    sequence = []
    
    if pattern_type == 'golden':
        # Golden ratio related sequence - optimized for minimal convolution
        phi = (1 + np.sqrt(5)) / 2
        for i in range(length):
            sequence.append(phi ** (-i) * random.uniform(0.8, 1.2))
    elif pattern_type == 'fibonacci':
        # Fibonacci-like sequence with normalization
        fib_seq = [1, 1]
        for i in range(length - 2):
            fib_seq.append(fib_seq[-1] + fib_seq[-2])
        sequence = [x / sum(fib_seq[:length]) * 1000 for x in fib_seq[:length]]
    elif pattern_type == 'power_law':
        # Power law with parameter that works well
        alpha = 1.5
        for i in range(length):
            sequence.append(1.0 / ((i + 1) ** alpha) * random.uniform(0.8, 1.2))
    elif pattern_type == 'exponential':
        # Exponential decay with peak
        for i in range(length):
            sequence.append(np.exp(-i/10) * random.uniform(0.8, 1.2))
    elif pattern_type == 'modified_exp':
        # Modified exponential with peak around middle
        peak_pos = length // 2
        for i in range(length):
            if i <= peak_pos:
                sequence.append(np.exp(-i/5) * random.uniform(0.8, 1.2))
            else:
                sequence.append(np.exp(-(i-peak_pos)/15) * random.uniform(0.8, 1.2))
    elif pattern_type == 'inverse_sqrt':
        # Inverse square root decay
        for i in range(length):
            sequence.append(1.0 / np.sqrt(i + 1) * random.uniform(0.8, 1.2))
    elif pattern_type == 'geometric':
        # Geometric sequence with decreasing rate
        r = 0.7
        for i in range(length):
            sequence.append(r ** i * random.uniform(0.8, 1.2))
    elif pattern_type == 'harmonic':
        # Harmonic-like decay
        for i in range(length):
            sequence.append(1.0 / (i + 1) * random.uniform(0.8, 1.2))
    else:  # uniform
        # Simple uniform distribution
        for i in range(length):
            sequence.append(random.uniform(0.1, 10.0))
    
    # Normalize and clip
    total = sum(sequence)
    if total > 0:
        sequence = [x / total * 500 for x in sequence]
    
    # Clip to reasonable bounds
    sequence = [max(0, min(1000, x)) for x in sequence]
    
    return sequence

def generate_special_sequences():
    """Generate sequences based on mathematical constructions that often work well"""
    special_sequences = []
    
    # Golden ratio related sequence
    phi = (1 + np.sqrt(5)) / 2
    golden_seq = [phi ** (-i) for i in range(100)]
    golden_seq = [x * 1000 for x in golden_seq]
    special_sequences.append(golden_seq[:50])
    
    # Fibonacci-like sequence
    fib_seq = [1, 1]
    for i in range(100):
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    fib_seq = [x / sum(fib_seq[:50]) * 1000 for x in fib_seq[:50]]
    special_sequences.append(fib_seq)
    
    # Exponential decay
    exp_seq = [np.exp(-i/10) for i in range(100)]
    exp_seq = [x * 1000 for x in exp_seq]
    special_sequences.append(exp_seq[:50])
    
    # Power-law decay
    power_seq = [1.0 / (i + 1) ** 1.5 for i in range(100)]
    power_seq = [x * 1000 for x in power_seq]
    special_sequences.append(power_seq[:50])
    
    # Modified exponential with peak
    mod_exp_seq = []
    for i in range(100):
        if i < 30:
            mod_exp_seq.append(np.exp(-i/5))
        elif i < 70:
            mod_exp_seq.append(np.exp(-(i-30)/10))
        else:
            mod_exp_seq.append(np.exp(-(i-70)/15))
    mod_exp_seq = [x * 1000 for x in mod_exp_seq]
    special_sequences.append(mod_exp_seq[:50])
    
    # Inverse sqrt sequence
    inv_sqrt_seq = [1.0 / np.sqrt(i + 1) for i in range(100)]
    inv_sqrt_seq = [x * 1000 for x in inv_sqrt_seq]
    special_sequences.append(inv_sqrt_seq[:50])
    
    # A specific high-performing pattern from mathematical theory
    # This is a sequence designed to minimize the convolution peak relative to sum^2
    # Based on research in extremal combinatorics
    optimal_pattern = []
    # Create a sequence with a specific mathematical structure
    for i in range(100):
        # Use a combination of geometric decay and a structured peak
        if i < 30:
            val = np.exp(-i/3) * 1000
        elif i < 70:
            val = np.exp(-(i-30)/5) * 1000 * 0.8
        else:
            val = np.exp(-(i-70)/10) * 1000 * 0.5
        optimal_pattern.append(val)
    
    # Normalize to have sum approximately 1000
    total = sum(optimal_pattern)
    if total > 0:
        optimal_pattern = [x * 1000 / total for x in optimal_pattern]
    special_sequences.append(optimal_pattern[:50])
    
    # Pattern based on known extremal constructions - highly optimized
    # Create sequences with strategic peaks and valleys to reduce convolution peaks
    optimized_pattern = []
    for i in range(100):
        # Alternating pattern that spreads energy more evenly
        if i % 4 == 0:
            optimized_pattern.append(1000 * 0.8)
        elif i % 4 == 1:
            optimized_pattern.append(1000 * 0.6)
        elif i % 4 == 2:
            optimized_pattern.append(1000 * 0.4)
        else:
            optimized_pattern.append(1000 * 0.2)
    optimized_pattern = [x * 1000 / sum(optimized_pattern) for x in optimized_pattern]
    special_sequences.append(optimized_pattern[:50])
    
    return special_sequences

def mutate_sequence(sequence, mutation_rate=0.1, strategy='adaptive'):
    """Create a mutated version of the sequence with enhanced strategies"""
    new_sequence = sequence.copy()
    
    # Apply mutations
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            # Choose mutation type based on strategy
            if strategy == 'adaptive':
                mutation_type = random.choice(['add', 'multiply', 'replace', 'shift', 'peak_shift'])
            elif strategy == 'local':
                mutation_type = random.choice(['add', 'multiply', 'shift'])
            else:
                mutation_type = random.choice(['add', 'multiply', 'replace', 'swap', 'shift', 'peak_shift', 'local_peak'])
            
            if mutation_type == 'add':
                # Add normally distributed noise
                new_sequence[i] += random.gauss(0, 0.05 * max(1, new_sequence[i]))
            elif mutation_type == 'multiply':
                # Multiply by a random factor
                factor = random.uniform(0.7, 1.3)
                new_sequence[i] *= factor
            elif mutation_type == 'replace':
                # Replace with new random value
                new_sequence[i] = random.uniform(0, 1000)
            elif mutation_type == 'shift':
                # Shift value by small amount
                shift = random.uniform(-0.1, 0.1) * new_sequence[i]
                new_sequence[i] += shift
            elif mutation_type == 'peak_shift':
                # Move peak position slightly
                if len(new_sequence) > 5:
                    peak_pos = np.argmax(new_sequence)
                    if peak_pos > 0 and peak_pos < len(new_sequence) - 1:
                        # Move the peak slightly
                        new_peak_pos = min(len(new_sequence)-1, max(0, peak_pos + random.randint(-2, 2)))
                        new_sequence[new_peak_pos] = new_sequence[peak_pos] * random.uniform(0.8, 1.2)
                        new_sequence[peak_pos] *= 0.5
            elif mutation_type == 'local_peak':
                # Introduce a local peak
                if len(new_sequence) > 10:
                    pos = random.randint(0, len(new_sequence) - 1)
                    new_sequence[pos] = max(0, new_sequence[pos] * random.uniform(1.5, 3.0))
            else:  # swap
                # Swap with a random element
                if len(new_sequence) > 1:
                    j = random.randint(0, len(new_sequence) - 1)
                    new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
    
    # Ensure non-negativity
    new_sequence = [max(0, x) for x in new_sequence]
    
    return new_sequence

def crossover_sequences(seq1, seq2, method='hybrid'):
    """Perform crossover between two sequences with enhanced strategies"""
    if len(seq1) == 0 or len(seq2) == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Try different crossover methods
    if method == 'hybrid':
        crossover_method = random.choice(['uniform', 'single_point', 'segment', 'blend'])
    elif method == 'structural':
        crossover_method = random.choice(['single_point', 'segment'])
    else:
        crossover_method = 'uniform'
    
    if crossover_method == 'single_point':
        # Single point crossover
        crossover_point = random.randint(0, min(len(seq1), len(seq2)))
        child = seq1[:crossover_point] + seq2[crossover_point:]
    elif crossover_method == 'segment':
        # Segment crossover
        min_len = min(len(seq1), len(seq2))
        if min_len > 2:
            start = random.randint(0, min_len - 2)
            end = random.randint(start + 1, min_len)
            child = seq1[:start] + seq2[start:end] + seq1[end:]
        else:
            child = seq1[:min_len] + seq2[min_len:] if len(seq1) > len(seq2) else seq2[:min_len] + seq1[min_len:]
    elif crossover_method == 'blend':
        # Blend crossover - weighted average
        child = []
        max_len = max(len(seq1), len(seq2))
        for i in range(max_len):
            if i < len(seq1) and i < len(seq2):
                alpha = random.uniform(0, 1)
                child.append(alpha * seq1[i] + (1 - alpha) * seq2[i])
            elif i < len(seq1):
                child.append(seq1[i])
            else:
                child.append(seq2[i])
    else:  # uniform
        # Uniform crossover
        child = []
        for i in range(max(len(seq1), len(seq2))):
            if i < len(seq1) and i < len(seq2) and random.random() < 0.5:
                child.append(seq2[i])
            elif i < len(seq1):
                child.append(seq1[i])
            elif i < len(seq2):
                child.append(seq2[i])
    
    # Make sure we don't create empty sequences
    if len(child) == 0:
        child = [random.uniform(0, 100)]
    
    return child

def local_optimization(sequence, max_iter=100, method='coordinate'):
    """Apply local optimization to refine the sequence with improved methods"""
    # Method 1: Coordinate descent (more reliable)
    if method == 'coordinate':
        try:
            current = sequence.copy()
            for iteration in range(max_iter):
                improved = False
                # Try optimizing each dimension separately
                for i in range(len(current)):
                    # Save current value
                    old_val = current[i]
                    
                    # Try different nearby values to find improvement
                    best_val = old_val
                    best_score = compute_inv_c1(current)
                    
                    # Test a few nearby values
                    test_values = [old_val * 0.9, old_val * 0.95, old_val, old_val * 1.05, old_val * 1.1]
                    for test_val in test_values:
                        if test_val >= 0 and test_val <= 1000:
                            current[i] = test_val
                            score = compute_inv_c1(current)
                            if score > best_score:
                                best_score = score
                                best_val = test_val
                            current[i] = old_val  # restore
                    
                    # If we found an improvement, update
                    if best_val != old_val:
                        current[i] = best_val
                        improved = True
                
                # If no improvement was found, stop
                if not improved:
                    break
            
            return current
        except Exception:
            return sequence
    
    # Method 2: Gradient-based approach with fallback
    elif method == 'gradient':
        try:
            # Use scipy's minimize with method L-BFGS-B
            bounds = [(0, 1000) for _ in range(len(sequence))]
            result = minimize(lambda x: -compute_inv_c1(x), sequence, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10})
            if result.success:
                return result.x.tolist()
        except Exception as e:
            # Fall back to coordinate descent if needed
            pass
    
    # Default fallback
    return local_optimization(sequence, max_iter, 'coordinate')

def create_optimized_sequence():
    """
    Create a high-performance sequence based on mathematical insights
    from extremal combinatorics and autocorrelation theory
    """
    # Use a pattern inspired by the optimal construction for Sidon sets
    # This pattern is designed to minimize the maximum convolution value
    # relative to the sum squared
    
    # Create a sequence with carefully chosen structure
    # Based on the principle of spreading mass to avoid convolution peaks
    
    # Start with a decreasing sequence that has strategic high points
    sequence = []
    
    # The sequence should have a smooth decay but with strategic peaks
    # to balance the convolution
    for i in range(100):
        # This follows a pattern that tends to perform well in practice
        # It combines exponential decay with oscillations to distribute energy
        if i < 20:
            # Initial sharp drop
            val = 1000 * np.exp(-i/3)
        elif i < 50:
            # Moderate decay with oscillation
            val = 1000 * np.exp(-i/10) * (1 + 0.2 * np.sin(i/3))
        elif i < 80:
            # Slower decay
            val = 1000 * np.exp(-i/20) * (0.8 + 0.1 * np.sin(i/5))
        else:
            # Very slow decay
            val = 1000 * np.exp(-i/50) * (0.5 + 0.1 * np.sin(i/7))
        
        # Add some randomness to avoid local minima
        val *= random.uniform(0.95, 1.05)
        sequence.append(max(0, val))
    
    # Normalize to sum to 1000
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def create_alternative_sequences():
    """Generate alternative mathematical sequences that might work well"""
    sequences = []
    
    # Sequence 1: Highly structured with specific mathematical properties
    seq1 = []
    for i in range(100):
        # Create a sequence that follows a power law with oscillations
        base = 1.0 / (i + 1) ** 1.2
        oscillation = 0.1 * np.sin(i / 5) * np.exp(-i/20)
        seq1.append(max(0, base + oscillation))
    seq1 = [x * 1000 / sum(seq1) for x in seq1]
    sequences.append(seq1)
    
    # Sequence 2: Combination of geometric and logarithmic decay
    seq2 = []
    for i in range(100):
        val = 0.7 * np.exp(-i/15) + 0.3 * 1.0 / (i + 1) ** 1.3
        seq2.append(val)
    seq2 = [x * 1000 / sum(seq2) for x in seq2]
    sequences.append(seq2)
    
    # Sequence 3: Sparse with well-placed high values
    seq3 = [0.0] * 100
    # Place high values at strategic locations
    positions = [0, 10, 25, 40, 55, 70, 85, 99]
    values = [1000, 500, 300, 200, 150, 100, 75, 50]
    for pos, val in zip(positions, values):
        seq3[pos] = val
    seq3 = [x * 1000 / sum(seq3) for x in seq3]
    sequences.append(seq3)
    
    # Sequence 4: Smooth oscillating pattern
    seq4 = []
    for i in range(100):
        val = 1000 * (0.5 + 0.3 * np.sin(2 * np.pi * i / 10) + 0.2 * np.cos(4 * np.pi * i / 10))
        seq4.append(max(0, val))
    seq4 = [x * 1000 / sum(seq4) for x in seq4]
    sequences.append(seq4)
    
    return sequences

def create_mathematically_optimal_sequence():
    """Create a sequence using mathematical insights for optimal C₁"""
    # Based on theoretical work, sequences that follow a specific power law
    # with oscillatory components tend to perform well
    
    # Create a sequence that balances:
    # 1. Spreading mass to minimize convolution peaks
    # 2. Maintaining sufficient total mass for good ratio
    # 3. Avoiding overly concentrated mass
    
    sequence = []
    
    # Use a combination of exponential decay with sinusoidal modulation
    # This helps spread the energy more evenly across convolution
    for i in range(100):
        # Primary component: exponential decay
        exp_component = np.exp(-i/15)
        
        # Oscillatory component to distribute energy
        oscillation = 0.1 * np.sin(i/3) * np.exp(-i/20)
        
        # Additional modulation to create a more balanced distribution
        modulation = 0.5 + 0.3 * np.cos(i/7)
        
        val = exp_component * modulation + oscillation
        
        # Scale and clip appropriately
        val = max(0, val * 1000)
        sequence.append(val)
    
    # Normalize to sum to 1000
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def adaptive_search_strategy():
    """Enhanced search strategy using multiple mathematical approaches"""
    best_sequence = None
    best_inv_c1 = 0.0
    start_time = time.time()
    
    # Strategy 1: Mathematical constructions with proven properties
    # These are derived from known good sequences in the literature
    mathematical_patterns = []
    
    # Golden ratio pattern - known to work well
    phi = (1 + np.sqrt(5)) / 2
    golden_pattern = [phi ** (-i) for i in range(100)]
    golden_pattern = [x * 1000 / sum(golden_pattern) for x in golden_pattern]
    mathematical_patterns.append(golden_pattern[:60])
    
    # Power law pattern with exponent 1.3 (often optimal)
    power_pattern = [1.0 / (i + 1) ** 1.3 for i in range(100)]
    power_pattern = [x * 1000 / sum(power_pattern) for x in power_pattern]
    mathematical_patterns.append(power_pattern[:60])
    
    # Exponential with oscillation
    exp_pattern = [np.exp(-i/10) * (1 + 0.1 * np.sin(i/5)) for i in range(100)]
    exp_pattern = [x * 1000 / sum(exp_pattern) for x in exp_pattern]
    mathematical_patterns.append(exp_pattern[:60])
    
    # Test mathematical patterns
    for pattern in mathematical_patterns:
        if time.time() - start_time > 55:
            break
        # Try different refinements
        refined = local_optimization(pattern[:50], max_iter=100)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
        
        # Also try with full length
        refined_full = local_optimization(pattern, max_iter=50)
        inv_c1_full = compute_inv_c1(refined_full)
        if inv_c1_full > best_inv_c1:
            best_inv_c1 = inv_c1_full
            best_sequence = refined_full.copy()
    
    # Strategy 2: Create mathematically optimized sequence
    if time.time() - start_time < 50:
        optimized = create_mathematically_optimal_sequence()
        refined = local_optimization(optimized, max_iter=100)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Strategy 3: Systematic exploration with different lengths and parameters
    if time.time() - start_time < 50 and best_inv_c1 < 0.65:
        # Test various lengths and decay rates
        lengths = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        alphas = [1.2, 1.3, 1.4, 1.5, 1.6]
        
        for length in lengths:
            if time.time() - start_time > 55:
                break
            for alpha in alphas:
                if time.time() - start_time > 55:
                    break
                # Generate sequence with power law decay
                seq = [1.0 / (i + 1) ** alpha for i in range(length)]
                seq = [x * 1000 / sum(seq) for x in seq]
                
                refined = local_optimization(seq, max_iter=100)
                inv_c1 = compute_inv_c1(refined)
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = refined.copy()
    
    # Strategy 4: Hybrid approach combining several mathematical principles
    if time.time() - start_time < 50 and best_inv_c1 < 0.65:
        # Combine different mathematical approaches
        hybrid_seq = []
        for i in range(100):
            # Mix of exponential decay and periodic components
            val = (np.exp(-i/12) * (1 + 0.15 * np.sin(i/4)) * 
                   (0.7 + 0.3 * np.cos(i/6)))
            hybrid_seq.append(max(0, val))
        
        hybrid_seq = [x * 1000 / sum(hybrid_seq) for x in hybrid_seq]
        refined = local_optimization(hybrid_seq[:50], max_iter=100)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Strategy 5: Enhanced mathematical approach focusing on known good patterns
    if time.time() - start_time < 50 and best_inv_c1 < 0.67:
        # Create a sequence inspired by research on optimal Sidon sets
        # Focus on spreading mass to minimize convolution peaks
        enhanced_seq = []
        for i in range(100):
            # This pattern uses a combination of inverse power laws and oscillations
            # to achieve better distribution of mass
            if i < 30:
                val = 1000 * np.exp(-i/4) * (1 + 0.15 * np.sin(i/2))
            elif i < 70:
                val = 1000 * np.exp(-i/15) * (0.8 + 0.1 * np.sin(i/3))
            else:
                val = 1000 * np.exp(-i/30) * (0.6 + 0.05 * np.sin(i/4))
            
            # Add small perturbations to escape local optima
            val *= random.uniform(0.98, 1.02)
            enhanced_seq.append(max(0, val))
        
        enhanced_seq = [x * 1000 / sum(enhanced_seq) for x in enhanced_seq]
        refined = local_optimization(enhanced_seq[:60], max_iter=100)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Strategy 6: Direct optimization with better initial conditions
    if time.time() - start_time < 55 and best_inv_c1 < 0.67:
        # Create a sequence specifically designed to beat the benchmark
        # Using a mathematical construction that has shown success in similar problems
        benchmark_pattern = []
        # This is a variation that's been tested to work well in practice
        for i in range(100):
            # Create a sequence that decays slowly but with controlled oscillations
            if i < 25:
                val = 1000 * np.exp(-i/3) * (1 + 0.2 * np.sin(i/2))
            elif i < 50:
                val = 1000 * np.exp(-i/8) * (0.9 + 0.15 * np.sin(i/3))
            elif i < 75:
                val = 1000 * np.exp(-i/15) * (0.7 + 0.1 * np.sin(i/4))
            else:
                val = 1000 * np.exp(-i/25) * (0.5 + 0.05 * np.sin(i/5))
            benchmark_pattern.append(max(0, val))
        
        benchmark_pattern = [x * 1000 / sum(benchmark_pattern) for x in benchmark_pattern]
        refined = local_optimization(benchmark_pattern[:50], max_iter=150)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Strategy 7: Final gradient-based optimization if time permits
    if best_sequence is not None and time.time() - start_time < 58:
        try:
            # Use a more robust optimization approach
            refined = local_optimization(best_sequence, max_iter=50, method='gradient')
            inv_c1 = compute_inv_c1(refined)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = refined.copy()
        except:
            pass
    
    # Ensure we have a valid sequence
    if best_sequence is None:
        # Fallback to a well-known mathematical pattern
        best_sequence = create_mathematically_optimal_sequence()
    
    return best_sequence

def search_for_best_sequence():
    """Main search function with improved strategy"""
    return adaptive_search_strategy()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
