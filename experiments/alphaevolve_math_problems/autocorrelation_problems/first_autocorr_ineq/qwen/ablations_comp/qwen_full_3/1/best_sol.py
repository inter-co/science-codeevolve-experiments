# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.signal import fftconvolve
import time
import random

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

def create_highly_optimized_patterns():
    """Create the most promising mathematical patterns based on proven approaches."""
    patterns = []
    
    # 1. Extreme peak patterns from INSPIRATION 1 (these are the gold standard)
    # Adding even more extreme variations
    extreme_patterns = [
        [100000.0] + [1.0] * 99,     # Absolute best - extremely strong peak
        [50000.0] + [1.0] * 99,      # Very strong peak
        [20000.0] + [1.0] * 99,      # Strong peak
        [10000.0] + [1.0] * 99,      # Very strong peak
        [5000.0] + [1.0] * 99,       # Strong peak
        [2000.0] + [1.0] * 99,       # Moderate peak
        [1000.0] + [1.0] * 99,       # Standard peak
        [10000.0] * 10 + [1.0] * 90, # Short peak
        [10000.0] * 20 + [1.0] * 80, # Longer peak
        [10000.0] * 50 + [1.0] * 50, # Half-half split
        # Multi-level patterns with extreme contrast
        [10000.0] * 30 + [100.0] * 30 + [10.0] * 40,  # Multi-level
        [5000.0] * 20 + [100.0] * 20 + [10.0] * 60,   # Another multi-level
        # Two-level with high contrast
        [10000.0] * 50 + [0.1] * 50,  # Very high contrast
        [10000.0] * 25 + [0.1] * 75,  # Different split
        # Even more extreme patterns
        [200000.0] + [1.0] * 99,     # Ultra-high peak
        [100000.0] * 5 + [1.0] * 95, # Very short ultra-high peak
        [50000.0] * 5 + [1.0] * 95,  # Slightly less extreme ultra-high
    ]
    patterns.extend(extreme_patterns)
    
    # 2. Proven geometric patterns with precise bases
    bases_to_try = [0.88, 0.90, 0.92, 0.94, 0.95]
    for base in bases_to_try:
        for n in [100, 150, 200, 250, 300]:
            pattern = [base**i for i in range(n)]
            patterns.append(pattern)
    
    # 3. Multi-scale exponential with specific decay rates
    for n in [100, 150, 200]:
        pattern = []
        for i in range(n):
            if i < n//4:
                val = 1000 * np.exp(-i/8.0)
            elif i < n//2:
                val = 500 * np.exp(-(i-n//4)/12.0)
            else:
                val = 100 * np.exp(-(i-n//2)/18.0)
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 4. Sharp transition patterns with controlled ratios
    for n in [100, 150, 200]:
        pattern = []
        for i in range(n):
            if i < n//3:
                pattern.append(1000.0)
            elif i < 2*n//3:
                pattern.append(100.0)
            else:
                pattern.append(10.0)
        patterns.append(pattern)
    
    # 5. Peak-centered with Gaussian shape
    for n in [100, 150, 200]:
        pattern = []
        center = n // 2
        for i in range(n):
            val = 1000 * np.exp(-((i - center)**2) / (2 * (n/5)**2))
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 6. Multi-level patterns with proven ratios
    for n in [100, 150, 200]:
        pattern = []
        levels = [1000, 500, 200, 100, 50]
        for i in range(n):
            pattern.append(levels[i % len(levels)])
        patterns.append(pattern)
    
    # 7. Hybrid with specific decay profiles
    for n in [100, 150, 200]:
        pattern = []
        for i in range(n):
            if i < n//4:
                pattern.append(1000.0)
            elif i < n//2:
                pattern.append(500.0)
            else:
                pattern.append(100.0 * np.exp(-(i-n//2)/15.0))
        patterns.append(pattern)
    
    # 8. Specific high performers from literature
    patterns.append([1000.0] + [1.0] * 99)  # Strong peak
    patterns.append([100.0] * 50 + [1.0] * 50)  # Moderate peak
    patterns.append([1.0] * 100)  # Uniform
    patterns.append([1000.0] * 20 + [1.0] * 80)  # Short peak
    patterns.append([1000.0] * 30 + [50.0] * 70)  # Balanced peak
    
    # 9. Specialized patterns with very high concentration
    for n in [100, 150]:
        pattern = []
        peak_start = n//2 - 5
        peak_end = n//2 + 5
        for i in range(n):
            if peak_start <= i <= peak_end:
                pattern.append(1000.0)
            elif i < peak_start:
                pattern.append(1000.0 * np.exp(-(peak_start - i)/10.0))
            else:
                pattern.append(1000.0 * np.exp(-(i - peak_end)/15.0))
        patterns.append(pattern)
    
    # 10. Power law distributions with specific exponents
    for n in [100, 150, 200]:
        for alpha in [0.8, 1.0, 1.2, 1.5]:
            pattern = [1.0 / (i+1)**alpha for i in range(n)]
            total = sum(pattern)
            if total > 0:
                pattern = [x * 1000 / total for x in pattern]
            patterns.append(pattern)
    
    # 11. Patterns from INSPIRATION 2 that were particularly effective
    # Very high contrast patterns
    patterns.extend([
        [100000.0] * 10 + [0.01] * 90,   # Very high contrast
        [50000.0] * 15 + [0.01] * 85,    # Medium high contrast
        [10000.0] * 20 + [0.01] * 80,    # Lower contrast
        [1000.0] * 25 + [0.01] * 75,     # Even lower contrast
        # Multi-stage patterns with extreme contrasts
        [10000.0] * 30 + [100.0] * 30 + [1.0] * 40,  # Multi-level extreme
        [5000.0] * 25 + [50.0] * 25 + [0.1] * 50,    # Another multi-level
    ])
    
    return patterns

def generate_specialized_sequences():
    """Generate sequences with mathematical properties that tend to work well."""
    sequences = []
    
    # Create sequences with specific mathematical properties
    for _ in range(50):  # Increased from 30 to get more candidates
        n = random.randint(80, 300)
        
        # Option 1: Geometric with base near 0.92 (known to work well)
        if random.random() < 0.35:
            base = 0.92 + random.uniform(-0.02, 0.02)
            sequence = [base**i for i in range(n)]
            sequences.append(sequence)
        
        # Option 2: Multi-stage decay
        elif random.random() < 0.35:
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

def enhanced_local_search(initial_sequence, max_iter=100):
    """
    Enhanced local search with more sophisticated perturbation strategies
    and better escape mechanisms.
    """
    current_sequence = initial_sequence.copy()
    current_score = compute_inv_c1(current_sequence)
    
    # Track recent improvements for adaptive behavior
    recent_improvements = []
    
    # Track visited states to prevent cycling
    visited_states = set()
    visited_states.add(tuple(current_sequence))
    
    # Add even more aggressive perturbation strategies
    aggressive_factors = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    
    for iteration in range(max_iter):
        # Try different types of perturbations
        best_sequence = current_sequence.copy()
        best_score = current_score
        
        # Strategy 1: Fine-grained multiplicative changes (even more comprehensive)
        for i in range(len(current_sequence)):
            original_value = current_sequence[i]
            # Test a wide range of factors for fine-tuning - including more extreme values
            factors = aggressive_factors
            for factor in factors:
                if original_value * factor >= 0.01:
                    candidate = current_sequence.copy()
                    candidate[i] = original_value * factor
                    new_score = compute_inv_c1(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_sequence = candidate.copy()
        
        # Strategy 2: Add/subtract with different magnitudes (larger deltas)
        for i in range(len(current_sequence)):
            original_value = current_sequence[i]
            # Test different delta values with larger ranges
            deltas = [-2.0, -1.5, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
            for delta in deltas:
                if original_value + delta >= 0.01:
                    candidate = current_sequence.copy()
                    candidate[i] = original_value + delta
                    new_score = compute_inv_c1(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_sequence = candidate.copy()
        
        # Strategy 3: Global scaling with more aggressive factors
        scale_factors = [0.8, 0.85, 0.9, 0.93, 0.97, 1.03, 1.07, 1.1, 1.15, 1.2]
        for sf in scale_factors:
            candidate = [x * sf for x in current_sequence]
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 4: Local averaging to reduce noise
        if len(current_sequence) > 10:
            candidate = current_sequence.copy()
            for i in range(1, len(candidate) - 1):
                avg_val = (current_sequence[i-1] + current_sequence[i] + current_sequence[i+1]) / 3.0
                candidate[i] = avg_val
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 5: Window-based averaging for better smoothing
        if len(current_sequence) > 20:
            window_size = min(5, len(current_sequence) // 10)
            if window_size > 1:
                candidate = current_sequence.copy()
                for i in range(window_size, len(candidate) - window_size):
                    window_sum = sum(current_sequence[i-window_size:i+window_size+1])
                    avg_val = window_sum / (2 * window_size + 1)
                    candidate[i] = avg_val
                new_score = compute_inv_c1(candidate)
                if new_score > best_score:
                    best_score = new_score
                    best_sequence = candidate.copy()
        
        # Strategy 6: Strategic segment changes for high-contrast patterns
        if len(current_sequence) > 10 and current_sequence[0] > 1000:
            # Try to optimize the peak specifically for high-peak patterns
            segment_size = max(3, len(current_sequence) // 20)
            for _ in range(3):  # Try 3 segment changes
                segment_start = random.randint(0, len(current_sequence) - segment_size)
                segment_end = min(segment_start + segment_size, len(current_sequence))
                change_factor = random.choice(aggressive_factors)
                candidate = current_sequence.copy()
                for i in range(segment_start, segment_end):
                    candidate[i] = max(0.01, candidate[i] * change_factor)
                new_score = compute_inv_c1(candidate)
                if new_score > best_score:
                    best_score = new_score
                    best_sequence = candidate.copy()
        
        # Strategy 7: Random replacement of elements for global exploration
        if random.random() < 0.1 and len(current_sequence) > 10:
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
        
        # Update if improvement found
        if best_score > current_score:
            current_sequence = best_sequence
            current_score = best_score
            recent_improvements.append(iteration)
            # Keep only last 5 improvements
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)
            visited_states.add(tuple(current_sequence))
        else:
            # Adaptive escape mechanism with better probability calculation
            if len(recent_improvements) > 0:
                # If we haven't improved recently, be more aggressive
                escape_prob = 0.25 + 0.15 * len(recent_improvements)
            else:
                escape_prob = 0.2
                
            if random.random() < escape_prob:
                # Random perturbation with larger magnitude
                idx = random.randint(0, len(current_sequence) - 1)
                current_sequence[idx] = max(0.01, current_sequence[idx] * random.uniform(0.7, 1.3))
                current_score = compute_inv_c1(current_sequence)
                visited_states.add(tuple(current_sequence))
            else:
                # Random mutation with lower probability to maintain diversity
                if random.random() < 0.05 and len(visited_states) < 20:
                    # If we haven't explored many states recently, diversify
                    idx = random.randint(0, len(current_sequence) - 1)
                    current_sequence[idx] = max(0.01, current_sequence[idx] * random.uniform(0.85, 1.15))
                    current_score = compute_inv_c1(current_sequence)
                    visited_states.add(tuple(current_sequence))
    
    return current_sequence

def enhanced_optimization_search():
    """
    Enhanced optimization combining multiple strategies from the inspirations.
    """
    best_inv_c1 = 0
    best_sequence = None
    best_result = None
    
    # Strategy 1: Highly optimized mathematical patterns (including extreme peaks)
    patterns = create_highly_optimized_patterns()
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
    
    # Strategy 3: Optimization around best patterns with better methods
    if best_sequence is not None:
        # Try multiple optimization methods with different settings
        methods_and_settings = [
            ('Nelder-Mead', {'maxiter': 75, 'disp': False}),
            ('Powell', {'maxiter': 75, 'disp': False}),
            ('L-BFGS-B', {'maxiter': 50, 'disp': False}),
        ]
        
        for method, options in methods_and_settings:
            try:
                result = minimize(
                    lambda x: -compute_inv_c1(x),  # Minimize negative of 1/C1
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
    
    # Strategy 4: Enhanced local search refinement with more iterations
    if best_sequence is not None:
        refined_sequence = enhanced_local_search(best_sequence, max_iter=80)
        metrics = evaluate_sequence(refined_sequence)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = refined_sequence
            best_result = metrics
    
    # Strategy 5: Additional targeted optimizations with bounds
    if best_sequence is not None:
        # Try with bounds for better numerical stability
        bounds = [(0.01, 1000.0)] * len(best_sequence)
        try:
            def objective(x):
                return -compute_inv_c1(x)
            
            # Try differential evolution for global exploration
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=25,
                popsize=20,
                disp=False
            )
            if de_result.success:
                test_sequence = de_result.x
                metrics = evaluate_sequence(test_sequence)
                if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                    best_inv_c1 = metrics['inv_c1']
                    best_sequence = test_sequence.tolist()
                    best_result = metrics
        except:
            pass
        
        # Try a few more specific optimization attempts
        for _ in range(5):
            try:
                # Slightly perturb and optimize with bounds
                perturbed = [x * random.uniform(0.95, 1.05) for x in best_sequence]
                result = minimize(
                    lambda x: -compute_inv_c1(x),
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'disp': False}
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
    
    # Strategy 6: Final boost with extreme patterns if needed
    if best_sequence is not None and best_inv_c1 < 0.65:
        # Try the absolute best extreme patterns directly
        extreme_patterns = [
            [100000.0] + [1.0] * 99,
            [50000.0] + [1.0] * 99,
            [20000.0] + [1.0] * 99,
        ]
        for pattern in extreme_patterns:
            metrics = evaluate_sequence(pattern)
            if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                best_inv_c1 = metrics['inv_c1']
                best_sequence = pattern
                best_result = metrics
    
    # Strategy 7: Add even more extreme patterns that may have been missed
    if best_inv_c1 < 0.67:  # If we're still below a very good threshold
        ultra_extreme_patterns = [
            [200000.0] + [1.0] * 99,    # Ultra-high peak
            [100000.0] * 5 + [1.0] * 95, # Very short ultra-high peak
            [50000.0] * 5 + [1.0] * 95,  # Slightly less extreme ultra-high
            [10000.0] * 5 + [1.0] * 95,  # Short but strong peak
        ]
        for pattern in ultra_extreme_patterns:
            metrics = evaluate_sequence(pattern)
            if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                best_inv_c1 = metrics['inv_c1']
                best_sequence = pattern
                best_result = metrics
    
    return best_sequence if best_sequence is not None else [1.0] * 100

def search_for_best_sequence() -> list[float]:
    """
    Main function to search for the best coefficient sequence.
    Incorporates insights from all inspirations for better performance.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    start_time = time.time()
    
    # Run enhanced optimization
    sequence = enhanced_optimization_search()
    
    # Ensure sequence has reasonable properties
    if len(sequence) == 0:
        sequence = [1.0]
    
    # Make sure sum is meaningful
    if np.sum(sequence) < 0.01:
        sequence = [x + 0.1 for x in sequence]
    
    # Final refinement with enhanced local search
    try:
        refined_sequence = enhanced_local_search(sequence, max_iter=50)
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
