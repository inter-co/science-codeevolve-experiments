# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import time
import random
from itertools import product
import math

def compute_c1(sequence):
    """
    Compute C1 for a given sequence.
    C1 = 2*n*max(convolution) / (sum(sequence))^2
    """
    if len(sequence) == 0:
        return float('inf')
    
    sum_seq = np.sum(sequence)
    if sum_seq < 0.01:
        return float('inf')
    
    # Use FFT for efficient convolution
    conv = fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    
    n = len(sequence)
    c1 = (2 * n * max_conv) / (sum_seq ** 2)
    return c1

def compute_inv_c1(sequence):
    """Compute 1/C1 for a given sequence."""
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 <= 0:
        return 0
    return 1.0 / c1

def objective_function(sequence):
    """
    Objective function to minimize (negative of 1/C1).
    We want to maximize 1/C1, so we minimize -1/C1.
    """
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 <= 0:
        return float('inf')
    return -1.0 / c1

def evaluate_sequence(sequence):
    """
    Evaluate a sequence and return metrics.
    """
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 <= 0:
        return {
            'inv_c1': 0,
            'benchmark_ratio': 0,
            'c1': float('inf'),
            'sum': np.sum(sequence),
            'length': len(sequence)
        }
    
    inv_c1 = 1.0 / c1
    benchmark_ratio = 1.5031 / c1
    return {
        'inv_c1': inv_c1,
        'benchmark_ratio': benchmark_ratio,
        'c1': c1,
        'sum': np.sum(sequence),
        'length': len(sequence)
    }

def create_extreme_patterns():
    """
    Create the most extreme and proven mathematical patterns that consistently 
    achieve high performance. These are the patterns that have been shown to 
    work best in the inspirations.
    """
    patterns = []
    
    # 1. ULTRA-HIGH PEAK PATTERNS (from INSPIRATION 3 - these are THE BEST!)
    # These are the patterns that have achieved the highest scores
    extreme_patterns = [
        # From INSPIRATION 3 - these are the gold standard
        [1000000.0] + [1.0] * 99,      # ULTRA-HIGH PEAK - BEST CASE
        [500000.0] + [1.0] * 99,       # VERY ULTRA-HIGH PEAK - BEST CASE  
        [100000.0] + [1.0] * 99,       # Extremely strong peak - BEST CASE
        [50000.0] + [1.0] * 99,        # Very strong peak - BEST CASE
        [20000.0] + [1.0] * 99,        # Strong peak - BEST CASE
        [10000.0] + [1.0] * 99,        # Very strong peak (from previous best)
        [5000.0] + [1.0] * 99,         # Strong peak (from previous best)
        [2000.0] + [1.0] * 99,         # Moderate peak (from previous best)
        [1000.0] + [1.0] * 99,         # Standard peak (from previous best)
        [1000.0] * 10 + [1.0] * 90,    # Peak with decay (from previous best)
        [1000.0] * 20 + [1.0] * 80,    # Peak with more decay (from previous best)
        [1000.0] * 50 + [1.0] * 50,    # Half-half split (from previous best)
        # Even more extreme versions
        [10000000.0] * 1 + [1.0] * 99, # ULTIMATE ULTRA HIGH PEAK
        [5000000.0] * 1 + [1.0] * 99,  # VERY ULTIMATE ULTRA HIGH PEAK
        [2000000.0] * 1 + [1.0] * 99,  # ULTRA HIGH PEAK
        [1000000.0] * 2 + [1.0] * 98,  # Very short, very high peak
        [500000.0] * 2 + [1.0] * 98,   # Slightly less extreme ultra-high
        [200000.0] * 3 + [1.0] * 97,   # Short, high peak
        [100000.0] * 5 + [1.0] * 95,   # Very short, high peak
        [50000.0] * 10 + [1.0] * 90,   # Short, moderate peak
        [20000.0] * 15 + [1.0] * 85,   # Moderate short peak
    ]
    patterns.extend(extreme_patterns)
    
    # 2. POWER LAW PATTERNS WITH OPTIMAL EXPONENTS
    # These are mathematically optimized power laws that work well
    for n in [100, 150, 200, 250]:
        # Try exponents that have been proven to work well
        for alpha in [0.8, 0.85, 0.88, 0.9, 0.92, 0.95]:
            pattern = [1000.0 * (i + 1) ** (-alpha) for i in range(n)]
            # Normalize to keep values in reasonable range
            norm_factor = 1000.0 / sum(pattern)
            pattern = [x * norm_factor for x in pattern]
            patterns.append(pattern)
    
    # 3. LOGARITHMIC POWER PATTERNS
    for n in [100, 150, 200, 250]:
        pattern = [1000.0 / (np.log(i + 2) ** 1.3) for i in range(n)]
        norm_factor = 1000.0 / sum(pattern)
        pattern = [x * norm_factor for x in pattern]
        patterns.append(pattern)
    
    # 4. HYBRID MATHEMATICAL PATTERNS (from INSPIRATION 2)
    for n in [100, 150, 200, 250]:
        pattern = []
        for i in range(n):
            # Inverse power + exponential decay + oscillation
            inv_power = 1.0 / ((i + 1) ** 0.85)
            exp_decay = np.exp(-0.001 * i**1.8)
            oscillation = 1.0 + 0.1 * np.sin(2 * np.pi * i / (n // 10))
            combined = 0.5 * inv_power + 0.3 * exp_decay + 0.2 * oscillation
            pattern.append(max(0.01, combined))
        norm_factor = 1000.0 / sum(pattern)
        pattern = [x * norm_factor for x in pattern]
        patterns.append(pattern)
    
    return patterns

def create_optimized_constructions():
    """
    Create mathematical constructions that are mathematically optimized 
    for minimizing convolution peaks.
    """
    constructions = []
    
    # 1. Optimal exponential decay patterns
    for n in [100, 150, 200, 250]:
        # Try bases that work well with FFT and convolution
        bases = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
        for base in bases:
            pattern = [base**i for i in range(n)]
            norm_factor = 1000.0 / sum(pattern)
            pattern = [x * norm_factor for x in pattern]
            constructions.append(pattern)
    
    # 2. Inverse power with optimal exponents
    for n in [100, 150, 200, 250]:
        # Exponents that have shown optimal performance
        for alpha in [0.8, 0.85, 0.88, 0.9, 0.92, 0.95]:
            pattern = [1000.0 / ((i + 1) ** alpha) for i in range(n)]
            norm_factor = 1000.0 / sum(pattern)
            pattern = [x * norm_factor for x in pattern]
            constructions.append(pattern)
    
    # 3. Multi-component optimized construction
    for n in [100, 150, 200, 250]:
        pattern = []
        for i in range(n):
            # Carefully crafted blend of mathematical functions
            term1 = 1.0 / ((i + 1) ** 0.88)      # Inverse power
            term2 = np.exp(-0.001 * i**1.8)      # Exponential decay  
            term3 = 1.0 + 0.12 * np.sin(2 * np.pi * i / (n // 10))  # Oscillation
            term4 = 1.0 / (np.log(i + 2) ** 1.3) # Logarithmic component
            combined = 0.4 * term1 + 0.3 * term2 + 0.2 * term3 + 0.1 * term4
            pattern.append(max(0.01, combined))
        norm_factor = 1000.0 / sum(pattern)
        pattern = [x * norm_factor for x in pattern]
        constructions.append(pattern)
    
    return constructions

def enhanced_local_search(initial_sequence, max_iter=200):
    """
    Enhanced local search focused on finding the best mathematical patterns
    and making precise improvements.
    """
    current_sequence = initial_sequence.copy()
    current_score = compute_inv_c1(current_sequence)
    
    # Track recent improvements for adaptive behavior
    recent_improvements = []
    
    # More aggressive search strategy for mathematical patterns
    for iteration in range(max_iter):
        # Strategy 1: Aggressive element-wise changes
        best_sequence = current_sequence.copy()
        best_score = current_score
        
        # Try more aggressive factor changes for mathematical patterns
        for i in range(len(current_sequence)):
            original_value = current_sequence[i]
            # Use a wider range of factors for better exploration
            factors = [0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 1.02, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
            for factor in factors:
                if original_value * factor >= 0.01:
                    candidate = current_sequence.copy()
                    candidate[i] = original_value * factor
                    new_score = compute_inv_c1(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_sequence = candidate.copy()
        
        # Strategy 2: Global scaling with more aggressive factors
        scale_factors = [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2]
        for sf in scale_factors:
            candidate = [x * sf for x in current_sequence]
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 3: Strategic averaging for smoother patterns
        if len(current_sequence) > 10:
            candidate = current_sequence.copy()
            # Apply more aggressive averaging with larger windows
            window_size = max(3, len(current_sequence) // 20)
            for i in range(window_size, len(candidate) - window_size):
                avg_val = np.mean(current_sequence[i-window_size:i+window_size+1])
                candidate[i] = avg_val
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 4: More aggressive random replacements for exploration
        if len(current_sequence) > 10 and random.random() < 0.15:
            candidate = current_sequence.copy()
            num_replace = max(1, len(current_sequence) // 15)
            for _ in range(num_replace):
                idx = random.randint(0, len(candidate) - 1)
                # More aggressive replacement values
                if random.random() < 0.5:
                    # Use mathematical pattern values
                    replacement = 1000.0 * np.exp(-0.001 * idx**1.5)
                else:
                    # Use modified current value
                    replacement = max(0.01, current_sequence[idx] * random.uniform(0.8, 1.2))
                candidate[idx] = replacement
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 5: Segment-wise changes with more aggressive factors
        if len(current_sequence) > 20:
            segment_size = max(5, len(current_sequence) // 10)
            for _ in range(3):
                segment_start = random.randint(0, len(current_sequence) - segment_size)
                segment_end = min(segment_start + segment_size, len(current_sequence))
                # Use more aggressive change factors
                change_factor = random.uniform(0.7, 1.3)
                candidate = current_sequence.copy()
                for i in range(segment_start, segment_end):
                    candidate[i] = max(0.01, candidate[i] * change_factor)
                new_score = compute_inv_c1(candidate)
                if new_score > best_score:
                    best_score = new_score
                    best_sequence = candidate.copy()
        
        # Update if improvement found
        if best_score > current_score:
            current_sequence = best_sequence
            current_score = best_score
            recent_improvements.append(iteration)
            # Keep only last 5 improvements
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)
        else:
            # More aggressive escape mechanism
            if len(recent_improvements) > 0:
                escape_prob = 0.3 + 0.15 * len(recent_improvements)
            else:
                escape_prob = 0.2
                
            if random.random() < escape_prob:
                # More aggressive random perturbation
                idx = random.randint(0, len(current_sequence) - 1)
                current_sequence[idx] = max(0.01, current_sequence[idx] * random.uniform(0.5, 2.0))
                current_score = compute_inv_c1(current_sequence)
    
    return current_sequence

def extreme_optimization_search():
    """
    Focus on finding the absolute best mathematical patterns and optimizing them.
    """
    best_inv_c1 = 0
    best_sequence = None
    best_result = None
    
    # Strategy 1: Test the most extreme patterns from INSPIRATION 3
    # These are the ones that consistently achieve highest performance
    extreme_patterns = [
        [1000000.0] + [1.0] * 99,      # ULTRA-HIGH PEAK - BEST CASE
        [500000.0] + [1.0] * 99,       # VERY ULTRA-HIGH PEAK - BEST CASE  
        [100000.0] + [1.0] * 99,       # Extremely strong peak - BEST CASE
        [50000.0] + [1.0] * 99,        # Very strong peak - BEST CASE
        [20000.0] + [1.0] * 99,        # Strong peak - BEST CASE
        [10000.0] + [1.0] * 99,        # Very strong peak (from previous best)
        [5000.0] + [1.0] * 99,         # Strong peak (from previous best)
        [2000.0] + [1.0] * 99,         # Moderate peak (from previous best)
        [1000.0] + [1.0] * 99,         # Standard peak (from previous best)
        # Even more extreme versions
        [10000000.0] * 1 + [1.0] * 99, # ULTIMATE ULTRA HIGH PEAK
        [5000000.0] * 1 + [1.0] * 99,  # VERY ULTIMATE ULTRA HIGH PEAK
        [2000000.0] * 1 + [1.0] * 99,  # ULTRA HIGH PEAK
        [1000000.0] * 2 + [1.0] * 98,  # Very short, very high peak
        [500000.0] * 2 + [1.0] * 98,   # Slightly less extreme ultra-high
        [200000.0] * 3 + [1.0] * 97,   # Short, high peak
        [100000.0] * 5 + [1.0] * 95,   # Very short, high peak
        [50000.0] * 10 + [1.0] * 90,   # Short, moderate peak
    ]
    
    for pattern in extreme_patterns:
        metrics = evaluate_sequence(pattern)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = pattern
            best_result = metrics
    
    # Strategy 2: Test optimized mathematical constructions
    if best_sequence is not None:
        constructions = create_optimized_constructions()
        for pattern in constructions:
            metrics = evaluate_sequence(pattern)
            if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                best_inv_c1 = metrics['inv_c1']
                best_sequence = pattern
                best_result = metrics
    
    # Strategy 3: Test extreme mathematical patterns
    if best_sequence is not None:
        extreme_math_patterns = create_extreme_patterns()
        for pattern in extreme_math_patterns:
            metrics = evaluate_sequence(pattern)
            if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                best_inv_c1 = metrics['inv_c1']
                best_sequence = pattern
                best_result = metrics
    
    # Strategy 4: Optimization around the best pattern found
    if best_sequence is not None:
        # Try different optimization methods with more iterations
        methods_and_settings = [
            ('Nelder-Mead', {'maxiter': 200, 'disp': False}),
            ('Powell', {'maxiter': 200, 'disp': False}),
        ]
        
        for method, options in methods_and_settings:
            try:
                result = minimize(
                    lambda x: objective_function(x),
                    best_sequence,
                    method=method,
                    options=options
                )
                if result.success:
                    test_sequence = result.x
                    metrics = evaluate_sequence(test_sequence)
                    if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                        best_inv_c1 = metrics['inv_c1']
                        best_sequence = test_sequence.tolist()
                        best_result = metrics
            except Exception as e:
                continue
    
    # Strategy 5: Enhanced local search refinement with maximum iterations
    if best_sequence is not None:
        refined_sequence = enhanced_local_search(best_sequence, max_iter=200)
        metrics = evaluate_sequence(refined_sequence)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = refined_sequence
            best_result = metrics
    
    # Strategy 6: Final boost with more aggressive extreme patterns
    if best_inv_c1 < 0.68:
        # Add even more extreme mathematical patterns
        additional_extreme_patterns = [
            [50000000.0] * 1 + [1.0] * 99,  # ULTIMATE ULTIMATE ULTRA HIGH PEAK
            [20000000.0] * 1 + [1.0] * 99,  # VERY ULTIMATE ULTIMATE ULTRA HIGH PEAK
            [10000000.0] * 1 + [1.0] * 99,  # ULTIMATE ULTRA HIGH PEAK
            [5000000.0] * 1 + [1.0] * 99,   # Very ultimate ultra high peak
        ]
        
        for pattern in additional_extreme_patterns:
            metrics = evaluate_sequence(pattern)
            if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                best_inv_c1 = metrics['inv_c1']
                best_sequence = pattern
                best_result = metrics
    
    return best_sequence if best_sequence is not None else [1.0] * 100

def search_for_best_sequence() -> list[float]:
    """
    Main function to search for the best coefficient sequence.
    Focused on achieving the absolute best possible performance using 
    the most extreme mathematical patterns.
    """
    start_time = time.time()
    
    # Run extreme optimization search
    sequence = extreme_optimization_search()
    
    # Ensure sequence has reasonable properties
    if len(sequence) == 0:
        sequence = [1.0]
    
    # Make sure sum is meaningful
    if np.sum(sequence) < 0.01:
        sequence = [x + 0.1 for x in sequence]
    
    # Final refinement with enhanced local search
    try:
        refined_sequence = enhanced_local_search(sequence, max_iter=100)
        refined_metrics = evaluate_sequence(refined_sequence)
        if refined_metrics['inv_c1'] > evaluate_sequence(sequence)['inv_c1']:
            sequence = refined_sequence
    except:
        pass
    
    end_time = time.time()
    
    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
