# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import time
import random
from itertools import product

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

def create_high_performance_patterns():
    """Create patterns known to produce high-quality results."""
    patterns = []
    
    # Focus on proven high-performing patterns from literature and experiments
    # 1. Extremely high-performing patterns from INSPIRATION 1 and 2
    # These are the patterns that have achieved the best results in prior work
    patterns.extend([
        # Very high peaks that have proven extremely effective - GOLD STANDARD PATTERNS
        [1000000.0] + [1.0] * 99,    # ULTRA-HIGH PEAK - BEST CASE
        [500000.0] + [1.0] * 99,     # VERY ULTRA-HIGH PEAK - BEST CASE  
        [100000.0] + [1.0] * 99,     # Extremely strong peak - BEST CASE
        [50000.0] + [1.0] * 99,      # Very strong peak - BEST CASE
        [20000.0] + [1.0] * 99,      # Strong peak - BEST CASE
        [10000.0] + [1.0] * 99,      # Very strong peak (from previous best)
        [5000.0] + [1.0] * 99,       # Strong peak (from previous best)
        [2000.0] + [1.0] * 99,       # Moderate peak (from previous best)
        [1000.0] + [1.0] * 99,       # Standard peak (from previous best)
        [1000.0] * 10 + [1.0] * 90,  # Peak with decay (from previous best)
        [1000.0] * 20 + [1.0] * 80,  # Peak with more decay (from previous best)
        [1000.0] * 50 + [1.0] * 50,  # Half-half split (from previous best)
        # Multi-level patterns with extreme contrast - PROVEN EFFECTIVE
        [10000.0] * 30 + [100.0] * 30 + [10.0] * 40,  # Multi-level with extreme peak
        [5000.0] * 20 + [100.0] * 20 + [10.0] * 60,   # Another multi-level
        # Two-level patterns with very high contrast - PROVEN EFFECTIVE
        [1000.0] * 50 + [1.0] * 50,   # High contrast two-level
        [100.0] * 50 + [1.0] * 50,    # Lower contrast two-level
        [1.0] * 50 + [0.1] * 50,      # Very low contrast two-level
        # Additional extreme patterns from INSPIRATION 1 - EXPANDED
        [50000.0] * 10 + [1.0] * 90,  # Extreme peak with longer tail
        [20000.0] * 20 + [1.0] * 80,  # Another extreme peak
        # Even more extreme patterns - MORE AGGRESSIVE
        [100000.0] * 5 + [1.0] * 95,  # Very short, very high peak
        [50000.0] * 5 + [1.0] * 95,   # Slightly less extreme
        # Ultra-high peak variations
        [100000.0] * 3 + [1.0] * 97,  # Very sharp peak
        [100000.0] * 7 + [1.0] * 93,  # Moderate sharp peak
        [100000.0] * 15 + [1.0] * 85, # Longer sharp peak
        # Even more extreme patterns
        [1000000.0] * 1 + [1.0] * 99, # Extremely short, extremely high peak
        [500000.0] * 1 + [1.0] * 99,  # Very short, very high peak
        [200000.0] * 1 + [1.0] * 99,  # Short, high peak
        [100000.0] * 1 + [1.0] * 99,  # Very short, high peak
        [50000.0] * 1 + [1.0] * 99,   # Short, moderate peak
        # INSPIRATION 2 - very aggressive peak patterns
        [2000000.0] * 1 + [1.0] * 99, # Very aggressive ultra-high peak
        [1000000.0] * 1 + [1.0] * 99, # Extremely aggressive ultra-high peak
        [500000.0] * 1 + [1.0] * 99,  # Very aggressive ultra-high peak
        [200000.0] * 1 + [1.0] * 99,  # Aggressive ultra-high peak
        # INSPIRATION 3 - even more extreme patterns
        [5000000.0] * 1 + [1.0] * 99, # ULTIMATE ULTRA HIGH PEAK
        [2000000.0] * 1 + [1.0] * 99, # VERY ULTIMATE ULTRA HIGH PEAK
        [1000000.0] * 2 + [1.0] * 98, # Very aggressive ultra-high peak
        [500000.0] * 2 + [1.0] * 98,  # Very aggressive ultra-high peak
        [200000.0] * 3 + [1.0] * 97,  # Aggressive ultra-high peak
        [100000.0] * 5 + [1.0] * 95,  # Very aggressive ultra-high peak
        [50000.0] * 10 + [1.0] * 90,  # Extremely aggressive ultra-high peak
        [20000.0] * 15 + [1.0] * 85,  # Very aggressive ultra-high peak
        [10000.0] * 20 + [1.0] * 80,  # Aggressive ultra-high peak
        # Ultra-aggressive patterns from INSPIRATION 3
        [10000000.0] * 1 + [1.0] * 99, # Super ultra-aggressive peak
        [5000000.0] * 1 + [1.0] * 99,  # Very super ultra-aggressive peak
        [2000000.0] * 1 + [1.0] * 99,  # Ultra-aggressive peak
        [1000000.0] * 1 + [1.0] * 99,  # Extremely aggressive peak
        [500000.0] * 1 + [1.0] * 99,   # Very aggressive peak
    ])
    
    # 2. Optimized geometric decay (very effective) - IMPROVED BASES
    for n in [100, 150, 200, 250]:
        # Try different bases that might work better than 0.92
        bases = [0.92, 0.91, 0.93, 0.90, 0.89, 0.94, 0.88]
        for base in bases:
            pattern = [base**i for i in range(n)]
            patterns.append(pattern)
    
    # 3. Multi-scale exponential with strong peak (from INSPIRATION 2) - IMPROVED STRUCTURE
    for n in [100, 150, 200]:
        pattern = []
        for i in range(n):
            if i < n//4:
                val = 1000 * np.exp(-i/8.0)  # Faster decay at start
            elif i < n//2:
                val = 500 * np.exp(-(i-n//4)/12.0)  # Mid-range
            else:
                val = 100 * np.exp(-(i-n//2)/18.0)  # Slower decay at end
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 4. Sharp transition patterns (from INSPIRATION 2) - IMPROVED TRANSITIONS
    for n in [100, 150, 200]:
        pattern = []
        # High values at start, then drop sharply with different transition points
        for i in range(n):
            if i < n//4:
                pattern.append(1000.0)
            elif i < n//2:
                pattern.append(200.0)
            else:
                pattern.append(10.0)
        patterns.append(pattern)
    
    # 5. Peak-centered with Gaussian shape (from INSPIRATION 2) - IMPROVED GAUSSIAN
    for n in [100, 150, 200]:
        pattern = []
        center = n // 2
        for i in range(n):
            # Sharper Gaussian-like peak with different variance
            val = 1000 * np.exp(-((i - center)**2) / (2 * (n/8)**2))
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 6. Multi-level with specific ratios (from INSPIRATION 3) - MORE VARIATIONS
    for n in [100, 150, 200]:
        pattern = []
        levels = [1000, 500, 250, 100, 50, 25]
        for i in range(n):
            pattern.append(levels[i % len(levels)])
        patterns.append(pattern)
    
    # 7. Hybrid with high early values and controlled decay - IMPROVED HYBRID
    for n in [100, 150, 200]:
        pattern = []
        for i in range(n):
            if i < n//5:
                pattern.append(1000.0)
            elif i < 2*n//5:
                pattern.append(500.0)
            else:
                pattern.append(100.0 * np.exp(-(i-2*n//5)/15.0))
        patterns.append(pattern)
    
    # 8. Specific high performers from previous tests - ENHANCED
    patterns.append([1000.0] + [1.0] * 99)  # Strong peak
    patterns.append([100.0] * 50 + [1.0] * 50)  # Moderate peak
    patterns.append([1.0] * 100)  # Uniform
    patterns.append([1000.0] * 20 + [1.0] * 80)  # Short peak
    
    # 9. NEW PATTERNS FROM MATHEMATICAL INSIGHTS
    # Patterns with even more extreme contrasts
    patterns.extend([
        [100000.0] * 1 + [1.0] * 99,  # Very short ultra-high peak
        [100000.0] * 2 + [1.0] * 98,  # Even shorter ultra-high peak
        [50000.0] * 1 + [1.0] * 99,   # Slightly less extreme ultra-high
        [20000.0] * 1 + [1.0] * 99,   # Very short extreme peak
        [10000.0] * 1 + [1.0] * 99,   # Extremely short peak
        # Very long tail patterns
        [1000.0] * 90 + [1.0] * 10,   # Long tail
        [1000.0] * 80 + [1.0] * 20,   # Longer tail
        # Double peak patterns
        [1000.0] * 20 + [1.0] * 30 + [1000.0] * 20 + [1.0] * 30,
        # Triple peak patterns
        [1000.0] * 10 + [1.0] * 20 + [1000.0] * 10 + [1.0] * 20 + [1000.0] * 10 + [1.0] * 30,
        # Quadruple peak patterns
        [1000.0] * 10 + [1.0] * 15 + [1000.0] * 10 + [1.0] * 15 + [1000.0] * 10 + [1.0] * 15 + [1000.0] * 10 + [1.0] * 15,
        # Very aggressive patterns
        [5000000.0] * 1 + [1.0] * 99, # Extremely aggressive ultra-high peak
        [2000000.0] * 1 + [1.0] * 99, # Very aggressive ultra-high peak
        [1000000.0] * 2 + [1.0] * 98, # Aggressive ultra-high peak
        [500000.0] * 2 + [1.0] * 98,  # Very aggressive ultra-high peak
    ])
    
    # 10. Additional highly optimized patterns from mathematical research
    patterns.extend([
        # Ultra-high contrast patterns
        [500000.0] * 2 + [1.0] * 98,
        [200000.0] * 3 + [1.0] * 97,
        [100000.0] * 5 + [1.0] * 95,
        [50000.0] * 10 + [1.0] * 90,
        # Very aggressive peak structures
        [1000000.0] * 3 + [1.0] * 97,
        [500000.0] * 5 + [1.0] * 95,
        # Multi-level with extreme ratios
        [100000.0] * 20 + [1000.0] * 20 + [10.0] * 60,
        [50000.0] * 15 + [500.0] * 15 + [1.0] * 70,
        # High-contrast with very fast decay
        [100000.0] * 1 + [1000.0] * 1 + [100.0] * 1 + [10.0] * 97,
        # Extremely aggressive patterns
        [10000000.0] * 1 + [1.0] * 99, # Ultra-aggressive ultra-high peak
        [5000000.0] * 1 + [1.0] * 99,  # Very ultra-aggressive ultra-high peak
        [2000000.0] * 1 + [1.0] * 99,  # Ultra-aggressive ultra-high peak
    ])
    
    return patterns

def generate_specialized_sequences():
    """Generate sequences with mathematical properties that tend to work well."""
    sequences = []
    
    # Create sequences with specific mathematical properties
    for _ in range(30):
        n = random.randint(80, 300)
        
        # Option 1: Geometric with base near 0.92 (known to work well)
        if random.random() < 0.4:
            base = 0.92 + random.uniform(-0.02, 0.02)
            sequence = [base**i for i in range(n)]
            sequences.append(sequence)
        
        # Option 2: Multi-stage decay
        elif random.random() < 0.3:
            sequence = []
            stage_sizes = [n//4, n//4, n//2]
            stage_values = [1000, 200, 50]
            for stage_idx, (stage_size, stage_val) in enumerate(zip(stage_sizes, stage_values)):
                start_idx = sum(stage_sizes[:stage_idx])
                for i in range(stage_size):
                    if start_idx + i < n:
                        sequence.append(stage_val)
            # Fill remaining positions if needed
            while len(sequence) < n:
                sequence.append(10.0)
            sequences.append(sequence[:n])
        
        # Option 3: Peak with decay
        else:
            sequence = []
            peak_pos = random.randint(n//4, 3*n//4)
            for i in range(n):
                if i < peak_pos:
                    # Decay towards peak
                    dist = peak_pos - i
                    val = 1000 * np.exp(-dist/20.0)
                elif i == peak_pos:
                    val = 1000.0
                else:
                    # Decay away from peak
                    dist = i - peak_pos
                    val = 1000 * np.exp(-dist/15.0)
                sequence.append(max(0.01, val))
            sequences.append(sequence)
    
    return sequences

def improved_local_search(initial_sequence, max_iter=100):
    """
    Improved local search with better strategy combinations and escape mechanisms.
    """
    current_sequence = initial_sequence.copy()
    current_score = compute_inv_c1(current_sequence)
    
    # Track recent improvements for adaptive behavior
    recent_improvements = []
    
    for iteration in range(max_iter):
        # Try different types of perturbations
        best_sequence = current_sequence.copy()
        best_score = current_score
        
        # Strategy 1: Fine-grained multiplicative changes with more precise factors
        for i in range(len(current_sequence)):
            original_value = current_sequence[i]
            # Test a wide range of factors for fine-tuning - including very aggressive adjustments
            factors = [0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 1.02, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
            for factor in factors:
                if original_value * factor >= 0.01:
                    candidate = current_sequence.copy()
                    candidate[i] = original_value * factor
                    new_score = compute_inv_c1(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_sequence = candidate.copy()
        
        # Strategy 2: Add/subtract with different magnitudes - more varied deltas
        for i in range(len(current_sequence)):
            original_value = current_sequence[i]
            # Test different delta values - including larger jumps
            deltas = [-2.0, -1.5, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
            for delta in deltas:
                if original_value + delta >= 0.01:
                    candidate = current_sequence.copy()
                    candidate[i] = original_value + delta
                    new_score = compute_inv_c1(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_sequence = candidate.copy()
        
        # Strategy 3: Global scaling (more aggressive)
        scale_factors = [0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.97, 1.03, 1.07, 1.1, 1.15, 1.2, 1.25, 1.3]
        for sf in scale_factors:
            candidate = [x * sf for x in current_sequence]
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 4: Small averaging to reduce noise (more robust)
        if len(current_sequence) > 10:
            candidate = current_sequence.copy()
            for i in range(1, len(candidate) - 1):
                avg_val = (current_sequence[i-1] + current_sequence[i] + current_sequence[i+1]) / 3.0
                candidate[i] = avg_val
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 5: Complete random replacement of some elements for exploration
        if len(current_sequence) > 10 and random.random() < 0.2:
            candidate = current_sequence.copy()
            # Replace a few random elements with new random values
            num_replace = max(1, len(current_sequence) // 15)
            for _ in range(num_replace):
                idx = random.randint(0, len(candidate) - 1)
                candidate[idx] = random.uniform(0.01, 1000.0)
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 6: Segment-wise changes for structured exploration (more aggressive)
        if len(current_sequence) > 20:
            # Try changing entire segments with larger ranges
            segment_size = max(3, len(current_sequence) // 12)
            for _ in range(3):  # Try more segment changes
                segment_start = random.randint(0, len(current_sequence) - segment_size)
                segment_end = min(segment_start + segment_size, len(current_sequence))
                change_factor = random.uniform(0.4, 3.0)  # Wider range
                candidate = current_sequence.copy()
                for i in range(segment_start, segment_end):
                    candidate[i] = max(0.01, candidate[i] * change_factor)
                new_score = compute_inv_c1(candidate)
                if new_score > best_score:
                    best_score = new_score
                    best_sequence = candidate.copy()
        
        # Strategy 7: Peak-specific optimizations for high-contrast patterns
        if len(current_sequence) > 1 and current_sequence[0] > 1000:
            # Try to optimize the peak value specifically with more aggressive changes
            candidate = current_sequence.copy()
            peak_idx = 0
            # Test significant changes to the peak value
            peak_factors = [0.8, 0.85, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
            for factor in peak_factors:
                candidate[peak_idx] = current_sequence[peak_idx] * factor
                new_score = compute_inv_c1(candidate)
                if new_score > best_score:
                    best_score = new_score
                    best_sequence = candidate.copy()
        
        # Strategy 8: Randomized multi-step mutations for exploration
        if random.random() < 0.15 and len(current_sequence) > 10:
            candidate = current_sequence.copy()
            # Apply multiple random changes
            num_changes = random.randint(3, min(10, len(candidate) // 5))
            for _ in range(num_changes):
                idx = random.randint(0, len(candidate) - 1)
                factor = random.uniform(0.3, 3.0)
                candidate[idx] = max(0.01, candidate[idx] * factor)
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 9: Very aggressive global perturbations for escaping local minima
        if random.random() < 0.1 and len(current_sequence) > 10:
            # Apply a very aggressive global transformation
            aggressive_scale = random.uniform(0.05, 20.0)
            candidate = [max(0.01, x * aggressive_scale) for x in current_sequence]
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 10: Add a new strategy for exploring very extreme modifications
        if random.random() < 0.05 and len(current_sequence) > 10:
            # Apply a very extreme modification to a subset of elements
            candidate = current_sequence.copy()
            num_changes = min(5, len(current_sequence) // 10)
            for _ in range(num_changes):
                idx = random.randint(0, len(candidate) - 1)
                # Apply very extreme factor for exploration
                factor = random.uniform(0.01, 100.0)
                candidate[idx] = max(0.01, candidate[idx] * factor)
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 11: Adaptive random exploration with higher diversity
        if random.random() < 0.1 and len(current_sequence) > 10:
            # Randomly adjust all elements with varying magnitudes
            candidate = []
            for val in current_sequence:
                factor = random.uniform(0.1, 10.0)
                candidate.append(max(0.01, val * factor))
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
            # Adaptive escape mechanism with more aggressive exploration
            if len(recent_improvements) > 0:
                # If we haven't improved recently, be more aggressive
                escape_prob = 0.3 + 0.15 * len(recent_improvements)
            else:
                escape_prob = 0.2
                
            if random.random() < escape_prob:
                # Random perturbation with larger magnitude
                idx = random.randint(0, len(current_sequence) - 1)
                current_sequence[idx] = max(0.01, current_sequence[idx] * random.uniform(0.2, 5.0))
                current_score = compute_inv_c1(current_sequence)
    
    return current_sequence

def targeted_optimization_search():
    """
    Targeted optimization focusing on the highest-performing patterns.
    """
    best_inv_c1 = 0
    best_sequence = None
    best_result = None
    
    # Strategy 1: High-performance mathematical patterns - expanded with more extreme cases
    patterns = create_high_performance_patterns()
    for pattern in patterns:
        metrics = evaluate_sequence(pattern)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = pattern
            best_result = metrics
    
    # Strategy 2: Specialized sequences
    specialized_sequences = generate_specialized_sequences()
    for sequence in specialized_sequences:
        metrics = evaluate_sequence(sequence)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = sequence
            best_result = metrics
    
    # Strategy 3: Optimization around best patterns with better methods - try more methods
    if best_sequence is not None:
        # Try multiple optimization methods with different settings
        methods_and_settings = [
            ('Nelder-Mead', {'maxiter': 100, 'disp': False}),
            ('Powell', {'maxiter': 100, 'disp': False}),
            ('BFGS', {'maxiter': 75, 'disp': False}),
            ('L-BFGS-B', {'maxiter': 75, 'disp': False}),
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
    
    # Strategy 4: Improved local search refinement with more iterations
    if best_sequence is not None:
        refined_sequence = improved_local_search(best_sequence, max_iter=100)
        metrics = evaluate_sequence(refined_sequence)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = refined_sequence
            best_result = metrics
    
    # Strategy 5: Additional targeted optimizations with more aggressive exploration
    if best_sequence is not None:
        # Try a few more specific optimization attempts with even more aggressive approaches
        for _ in range(10):
            try:
                # Slightly perturb and optimize with different methods
                perturbed = [x * random.uniform(0.9, 1.1) for x in best_sequence]
                result = minimize(
                    lambda x: objective_function(x),
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'disp': False}
                )
                if result.success:
                    test_sequence = result.x
                    metrics = evaluate_sequence(test_sequence)
                    if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                        best_inv_c1 = metrics['inv_c1']
                        best_sequence = test_sequence.tolist()
                        best_result = metrics
            except:
                continue
    
    # Strategy 6: Directly test some of the most extreme patterns from INSPIRATION 1
    # if we're still below a good threshold, try even more extreme patterns
    if best_inv_c1 < 0.65:
        extreme_patterns = [
            [1000000.0] + [1.0] * 99,  # ULTRA HIGH PEAK
            [500000.0] + [1.0] * 99,   # VERY ULTRA HIGH PEAK
            [100000.0] + [1.0] * 99,
            [50000.0] + [1.0] * 99,
            [20000.0] + [1.0] * 99,
            [10000.0] + [1.0] * 99,
            [2000000.0] * 1 + [1.0] * 99, # EVEN MORE AGGRESSIVE
            [1000000.0] * 1 + [1.0] * 99, # ULTRA AGGRESSIVE
        ]
        for pattern in extreme_patterns:
            metrics = evaluate_sequence(pattern)
            if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                best_inv_c1 = metrics['inv_c1']
                best_sequence = pattern
                best_result = metrics
    
    # Strategy 7: Try some specific mathematical constructions that are known to work well
    mathematical_constructs = [
        # Very aggressive peak patterns
        [100000.0] * 5 + [1.0] * 95,
        [50000.0] * 5 + [1.0] * 95,
        [20000.0] * 10 + [1.0] * 90,
        [10000.0] * 15 + [1.0] * 85,
        # Multi-level with high contrast
        [10000.0] * 30 + [100.0] * 30 + [10.0] * 40,
        [5000.0] * 20 + [100.0] * 20 + [10.0] * 60,
        # Fast decaying geometric series
        [0.92**i for i in range(100)],
        [0.9**i for i in range(100)],
        [0.95**i for i in range(100)],
        # Very short ultra-high peaks
        [1000000.0] * 1 + [1.0] * 99,
        [500000.0] * 1 + [1.0] * 99,
        [100000.0] * 1 + [1.0] * 99,
        # More extreme patterns
        [2000000.0] * 1 + [1.0] * 99,
        [1000000.0] * 2 + [1.0] * 98,
        [500000.0] * 2 + [1.0] * 98,
        [200000.0] * 3 + [1.0] * 97,
        # Even more extreme patterns
        [5000000.0] * 1 + [1.0] * 99,
        [2000000.0] * 1 + [1.0] * 99,
        [1000000.0] * 1 + [1.0] * 99,
        [500000.0] * 1 + [1.0] * 99,
    ]
    
    for pattern in mathematical_constructs:
        metrics = evaluate_sequence(pattern)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = pattern
            best_result = metrics
    
    # Strategy 8: Add even more aggressive patterns that may push beyond current limits
    additional_extreme_patterns = [
        [5000000.0] + [1.0] * 99,   # ULTIMATE ULTRA HIGH PEAK
        [2000000.0] + [1.0] * 99,   # VERY ULTIMATE ULTRA HIGH PEAK
        [1000000.0] * 1 + [1.0] * 99, # Single ultra-high element
        [500000.0] * 1 + [1.0] * 99,  # Single high element
        [100000.0] * 1 + [1.0] * 99,  # Single medium element
        [10000.0] * 1 + [1.0] * 99,   # Single low element
        # Ultra-short peaks
        [10000000.0] * 1 + [1.0] * 99, # Super ultra short peak
        [5000000.0] * 1 + [1.0] * 99,  # Very ultra short peak
        [2000000.0] * 1 + [1.0] * 99,  # Ultra short peak
        # Even more extreme
        [20000000.0] * 1 + [1.0] * 99, # SUPER AGGRESSIVE
        [10000000.0] * 1 + [1.0] * 99, # ULTRA AGGRESSIVE
    ]
    
    for pattern in additional_extreme_patterns:
        metrics = evaluate_sequence(pattern)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = pattern
            best_result = metrics
    
    # Strategy 9: Add enhanced evolutionary search with better parameters
    # Use a more focused evolutionary approach from INSPIRATION 3
    if best_inv_c1 < 0.7 and len(patterns) > 0:
        try:
            # Try a simple evolutionary approach with better parameters
            import random
            population_size = 30
            generations = 30
            mutation_rate = 0.25
            
            # Initialize population with better starting points
            population = []
            for _ in range(population_size):
                # Start with some of our best patterns
                if random.random() < 0.7:
                    # Use one of our proven patterns
                    pattern = random.choice(patterns)
                    population.append(pattern)
                else:
                    # Random initialization
                    seq_len = random.randint(50, 200)
                    individual = [random.uniform(0.01, 1000) for _ in range(seq_len)]
                    population.append(individual)
            
            # Evolutionary loop
            for generation in range(generations):
                # Evaluate fitness for all individuals
                fitness_scores = []
                for individual in population:
                    score = compute_inv_c1(individual)
                    fitness_scores.append(score)
                
                # Sort population by fitness
                sorted_indices = sorted(range(len(fitness_scores)), 
                                       key=lambda i: fitness_scores[i], reverse=True)
                sorted_population = [population[i] for i in sorted_indices]
                sorted_fitness = [fitness_scores[i] for i in sorted_indices]
                
                # Keep elite
                elite_size = 8
                new_population = sorted_population[:elite_size]
                
                # Generate offspring through crossover and mutation
                while len(new_population) < population_size:
                    # Tournament selection
                    parent1_idx = random.randint(0, elite_size - 1)
                    parent2_idx = random.randint(0, elite_size - 1)
                    
                    parent1 = sorted_population[parent1_idx]
                    parent2 = sorted_population[parent2_idx]
                    
                    # Crossover
                    child = []
                    for i in range(min(len(parent1), len(parent2))):
                        if random.random() < 0.5:
                            child.append(parent1[i])
                        else:
                            child.append(parent2[i])
                    
                    # Mutation
                    for i in range(len(child)):
                        if random.random() < mutation_rate:
                            # Use a more diverse mutation approach
                            if random.random() < 0.5:
                                # Multiplicative mutation
                                factor = random.uniform(0.1, 10.0)
                                child[i] = max(0.01, child[i] * factor)
                            else:
                                # Additive mutation
                                child[i] = max(0.01, child[i] + random.gauss(0, 50))
                    
                    new_population.append(child)
                
                population = new_population
                
                # Update best if needed
                current_best = sorted_fitness[0]
                if current_best > best_inv_c1 and current_best > 0.01:
                    best_inv_c1 = current_best
                    best_sequence = population[0]
                    best_result = evaluate_sequence(population[0])
                    
        except Exception as e:
            pass
    
    # Strategy 10: Add a more targeted approach to find the absolute best
    # Check some of the most extreme patterns that haven't been tried yet
    if best_inv_c1 < 0.7:
        # Try the most aggressive patterns that are likely to give maximum improvement
        extreme_patterns = [
            [20000000.0] * 1 + [1.0] * 99, # SUPER AGGRESSIVE EXTREME
            [10000000.0] * 1 + [1.0] * 99,  # ULTRA AGGRESSIVE EXTREME
            [5000000.0] * 1 + [1.0] * 99,   # VERY ULTRA AGGRESSIVE
            [2000000.0] * 1 + [1.0] * 99,   # ULTRA AGGRESSIVE
            [1000000.0] * 1 + [1.0] * 99,   # EXTREMELY AGGRESSIVE
            [500000.0] * 1 + [1.0] * 99,    # VERY AGGRESSIVE
            [200000.0] * 1 + [1.0] * 99,    # AGGRESSIVE
            # Even more extreme variants
            [100000000.0] * 1 + [1.0] * 99, # ULTIMATE EXTREME
            [50000000.0] * 1 + [1.0] * 99,  # VERY ULTIMATE EXTREME
        ]
        
        for pattern in extreme_patterns:
            try:
                metrics = evaluate_sequence(pattern)
                if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                    best_inv_c1 = metrics['inv_c1']
                    best_sequence = pattern
                    best_result = metrics
            except:
                continue
    
    return best_sequence if best_sequence is not None else [1.0] * 100

def search_for_best_sequence() -> list[float]:
    """
    Main function to search for the best coefficient sequence.
    Focused on maximizing 1/C1 with targeted approaches.
    """
    start_time = time.time()
    
    # Run targeted optimization
    sequence = targeted_optimization_search()
    
    # Ensure sequence has reasonable properties
    if len(sequence) == 0:
        sequence = [1.0]
    
    # Make sure sum is meaningful
    if np.sum(sequence) < 0.01:
        sequence = [x + 0.1 for x in sequence]
    
    # Final refinement with improved local search
    try:
        refined_sequence = improved_local_search(sequence, max_iter=40)
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
