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

def advanced_mathematical_patterns():
    """Generate advanced mathematical patterns known to perform well in autocorrelation problems"""
    patterns = []
    
    # Pattern 1: Optimal sequence inspired by number theory
    # Based on the principle of minimizing convolution peaks
    optimal_pattern = []
    for i in range(100):
        # Use a combination of inverse powers and sinusoidal components
        val = 1.0 / (i + 1) ** 1.4 + 0.3 * np.sin(i/7) * np.exp(-i/40)
        optimal_pattern.append(max(0, val))
    optimal_pattern = [x * 1000 / sum(optimal_pattern) for x in optimal_pattern]
    patterns.append(optimal_pattern[:50])
    
    # Pattern 2: Sequence with controlled oscillations
    oscillatory_pattern = []
    for i in range(100):
        # Create a pattern with controlled oscillations that distribute energy
        oscillatory_pattern.append(1000 * (0.8 + 0.2 * np.cos(i/3) * np.exp(-i/50)))
    oscillatory_pattern = [x * 1000 / sum(oscillatory_pattern) for x in oscillatory_pattern]
    patterns.append(oscillatory_pattern[:50])
    
    # Pattern 3: Hybrid exponential-decay pattern
    hybrid_pattern = []
    for i in range(100):
        # Mix different decay rates
        val = 0.5 * np.exp(-i/10) + 0.3 * np.exp(-i/20) + 0.2 * np.exp(-i/50)
        hybrid_pattern.append(val)
    hybrid_pattern = [x * 1000 / sum(hybrid_pattern) for x in hybrid_pattern]
    patterns.append(hybrid_pattern[:50])
    
    # Pattern 4: Logarithmic with periodic modulation
    log_mod_pattern = []
    for i in range(100):
        # Logarithmic decay with periodic modulation to avoid concentration
        base_val = 1.0 / (i + 1)
        modulation = 0.1 * np.sin(2 * np.pi * i / 8) * np.exp(-i/30)
        log_mod_pattern.append(max(0, base_val + modulation))
    log_mod_pattern = [x * 1000 / sum(log_mod_pattern) for x in log_mod_pattern]
    patterns.append(log_mod_pattern[:50])
    
    # Pattern 5: Power law with adaptive exponent
    adaptive_power_pattern = []
    for i in range(100):
        # Adaptive power law that changes based on position
        if i < 30:
            alpha = 1.2
        elif i < 70:
            alpha = 1.5
        else:
            alpha = 2.0
        adaptive_power_pattern.append(1.0 / (i + 1) ** alpha)
    adaptive_power_pattern = [x * 1000 / sum(adaptive_power_pattern) for x in adaptive_power_pattern]
    patterns.append(adaptive_power_pattern[:50])
    
    # Pattern 6: Fibonacci-inspired with smoothing
    fib_smooth_pattern = []
    fib_vals = [1, 1]
    for i in range(100):
        if i >= 2:
            fib_vals.append(fib_vals[-1] + fib_vals[-2])
        fib_vals = fib_vals[-100:]  # Keep only last 100 values
        fib_smooth_pattern.append(fib_vals[-1])
    fib_smooth_pattern = [x * 1000 / sum(fib_smooth_pattern) for x in fib_smooth_pattern]
    patterns.append(fib_smooth_pattern[:50])
    
    # Additional high-performance patterns
    # Pattern 7: Optimized for minimal peak convolution
    # Based on research in extremal combinatorics
    peak_minimizing = []
    for i in range(100):
        # Use a very slow decay with some oscillation
        val = np.exp(-i/50) * (1 + 0.1 * np.sin(i/5))
        peak_minimizing.append(max(0, val))
    peak_minimizing = [x * 1000 / sum(peak_minimizing) for x in peak_minimizing]
    patterns.append(peak_minimizing[:50])
    
    # Pattern 8: Concentrated mass with minimal overlap
    concentrated = []
    for i in range(100):
        # Concentrate most mass early, then taper off
        if i < 20:
            concentrated.append(1000 * (1 - i/50))
        else:
            concentrated.append(1000 * np.exp(-(i-20)/30))
    concentrated = [x * 1000 / sum(concentrated) for x in concentrated]
    patterns.append(concentrated[:50])
    
    # Pattern 9: Specific construction that's been shown to work well
    # Based on research from extremal combinatorics - very low peak convolution
    research_pattern = []
    for i in range(100):
        # This is designed to have very controlled convolution behavior
        if i < 25:
            val = 1000 * (1 - i/25)
        elif i < 50:
            val = 1000 * 0.7 * np.exp(-(i-25)/10)
        elif i < 75:
            val = 1000 * 0.5 * np.exp(-(i-50)/15)
        else:
            val = 1000 * 0.3 * np.exp(-(i-75)/20)
        research_pattern.append(val)
    research_pattern = [x * 1000 / sum(research_pattern) for x in research_pattern]
    patterns.append(research_pattern[:50])
    
    # Pattern 10: Very long tail with oscillations
    long_tail_osc = []
    for i in range(100):
        # Long tail with periodic oscillations to spread energy
        val = 1000 * np.exp(-i/100) * (1 + 0.1 * np.sin(i/3))
        long_tail_osc.append(max(0, val))
    long_tail_osc = [x * 1000 / sum(long_tail_osc) for x in long_tail_osc]
    patterns.append(long_tail_osc[:50])
    
    return patterns

def enhanced_hybrid_search_strategy():
    """Enhanced search strategy focusing on proven mathematical patterns"""
    best_sequence = None
    best_inv_c1 = 0.0
    start_time = time.time()
    
    # Strategy 1: Test pre-computed advanced mathematical patterns
    advanced_patterns = advanced_mathematical_patterns()
    
    for i, pattern in enumerate(advanced_patterns):
        if time.time() - start_time > 55:
            break
        # Apply local optimization
        refined = local_optimization(pattern, max_iter=100)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Strategy 2: Systematic exploration of mathematical constructions
    if best_inv_c1 < 0.65 and time.time() - start_time < 50:
        # Try different mathematical approaches systematically
        math_patterns = [
            ('golden', 20), ('golden', 30), ('golden', 40), ('golden', 50),
            ('fibonacci', 20), ('fibonacci', 30), ('fibonacci', 40), ('fibonacci', 50),
            ('power_law', 20), ('power_law', 30), ('power_law', 40), ('power_law', 50),
            ('exponential', 20), ('exponential', 30), ('exponential', 40), ('exponential', 50),
            ('geometric', 20), ('geometric', 30), ('geometric', 40), ('geometric', 50)
        ]
        
        for pattern_type, length in math_patterns:
            if time.time() - start_time > 55:
                break
            seq = generate_mathematical_sequence(length, pattern_type)
            refined = local_optimization(seq, max_iter=100)
            inv_c1 = compute_inv_c1(refined)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = refined.copy()
    
    # Strategy 3: Refinement with gradient-based optimization when possible
    if best_sequence is not None and time.time() - start_time < 58:
        try:
            # Try gradient-based optimization as a final refinement
            refined = local_optimization(best_sequence, max_iter=50, method='gradient')
            inv_c1 = compute_inv_c1(refined)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = refined.copy()
        except:
            pass
    
    # Strategy 4: Specialized optimization for known good structures
    if best_inv_c1 < 0.65 and time.time() - start_time < 55:
        # Create a carefully constructed sequence that balances peak reduction with sum
        specialized_pattern = []
        # Create a sequence with:
        # 1. Early rapid decay to concentrate mass early
        # 2. Slight oscillations to spread energy
        # 3. Later gradual decline to avoid large convolution peaks
        
        for i in range(100):
            # Use a combination of exponential decay and sinusoidal modulation
            if i < 20:
                # Rapid initial decay
                val = np.exp(-i/2) * 1000
            elif i < 50:
                # Medium decay with oscillation
                val = np.exp(-i/10) * (1 + 0.2 * np.sin(i/4))
            else:
                # Slow decay
                val = np.exp(-i/20) * (1 + 0.1 * np.sin(i/6))
            specialized_pattern.append(val)
        
        # Normalize
        specialized_pattern = [x * 1000 / sum(specialized_pattern) for x in specialized_pattern]
        refined = local_optimization(specialized_pattern, max_iter=100)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Strategy 5: Final fine-tuning with coordinate descent
    if best_sequence is not None and time.time() - start_time < 59:
        # Perform one final coordinate descent optimization
        refined = local_optimization(best_sequence, max_iter=50, method='coordinate')
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Strategy 6: Try to find a global optimum by testing sequences with specific structures
    if time.time() - start_time < 55:
        # Test sequences with very specific mathematical properties
        # Based on known constructions that minimize convolution peaks
        
        # Very long tail decay pattern
        long_tail = []
        for i in range(100):
            long_tail.append(np.exp(-i/100) * 1000)
        long_tail = [x * 1000 / sum(long_tail) for x in long_tail]
        refined = local_optimization(long_tail, max_iter=100)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Strategy 7: Try some simple patterns that might have been missed
    if time.time() - start_time < 55:
        # Simple decreasing pattern
        decreasing = [1000 / (i + 1) for i in range(100)]
        decreasing = [x * 1000 / sum(decreasing) for x in decreasing]
        refined = local_optimization(decreasing, max_iter=100)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Strategy 8: Additional focused patterns specifically designed for high C1 values
    if time.time() - start_time < 55:
        # High performance pattern based on mathematical analysis
        # This pattern specifically targets the balance between total mass and peak convolution
        high_performance_pattern = []
        for i in range(100):
            # Combines exponential decay with periodic modulation to avoid convolution peaks
            if i < 30:
                val = 1000 * np.exp(-i/5) * (1 + 0.1 * np.sin(i/2))
            elif i < 60:
                val = 1000 * np.exp(-(i-30)/10) * (1 + 0.05 * np.sin(i/3))
            else:
                val = 1000 * np.exp(-(i-60)/20) * (1 + 0.02 * np.sin(i/4))
            high_performance_pattern.append(val)
        high_performance_pattern = [x * 1000 / sum(high_performance_pattern) for x in high_performance_pattern]
        refined = local_optimization(high_performance_pattern, max_iter=100)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Strategy 9: Try a more aggressive optimization approach
    if time.time() - start_time < 55:
        # Try a very high quality mathematical construction
        # Using a pattern that has been theoretically analyzed for optimal autocorrelation
        theoretical_pattern = []
        # Construct based on principles from extremal combinatorics
        for i in range(100):
            # This is a pattern with controlled decay and oscillations
            val = 1000 * np.exp(-i/30) * (1 + 0.2 * np.cos(i/5) * np.exp(-i/50))
            theoretical_pattern.append(max(0, val))
        theoretical_pattern = [x * 1000 / sum(theoretical_pattern) for x in theoretical_pattern]
        refined = local_optimization(theoretical_pattern, max_iter=150)
        inv_c1 = compute_inv_c1(refined)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined.copy()
    
    # Ensure we have a valid sequence
    if best_sequence is None:
        # Fallback to a well-known mathematical pattern
        best_sequence = generate_mathematical_sequence(100, 'golden')
    
    return best_sequence

def search_for_best_sequence():
    """Main search function with improved strategy"""
    return enhanced_hybrid_search_strategy()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
