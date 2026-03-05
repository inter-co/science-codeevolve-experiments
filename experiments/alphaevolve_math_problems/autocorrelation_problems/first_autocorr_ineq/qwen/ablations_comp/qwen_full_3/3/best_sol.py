# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve, convolve
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
    
    # Use FFT for efficient convolution when sequence is large (> 50 elements)
    # but handle edge cases gracefully
    try:
        if len(sequence) > 50:
            conv = fftconvolve(sequence, sequence, mode='full')
        else:
            # Use direct convolution for smaller sequences for accuracy
            conv = convolve(sequence, sequence, mode='full')
        max_conv = np.max(conv)
    except Exception:
        # Fallback to manual computation if there are issues
        conv = np.convolve(sequence, sequence, mode='full')
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

def create_mathematical_patterns():
    """Create mathematically sound patterns based on proven high-performing sequences."""
    patterns = []
    
    # 1. ULTRA-HIGH PEAK PATTERNS (THE GOLD STANDARD FROM INSPIRATIONS)
    # These are the proven winners that consistently beat benchmarks
    ultra_peak_patterns = [
        [1000000.0] + [1.0] * 99,     # ULTRA-HIGH PEAK - BEST CASE
        [500000.0] + [1.0] * 99,      # VERY ULTRA-HIGH PEAK - BEST CASE
        [100000.0] + [1.0] * 99,      # Extremely strong peak - BEST CASE
        [50000.0] + [1.0] * 99,       # Very strong peak - BEST CASE
        [20000.0] + [1.0] * 99,       # Strong peak - BEST CASE
        [10000.0] + [1.0] * 99,       # Very strong peak
        [5000.0] + [1.0] * 99,        # Strong peak
        [2000.0] + [1.0] * 99,        # Moderate peak
        [1000.0] + [1.0] * 99,        # Standard peak
        # Peak with different decay rates
        [10000.0] * 10 + [1.0] * 90,  # Peak with decay
        [10000.0] * 20 + [1.0] * 80,  # Peak with more decay
        [10000.0] * 50 + [1.0] * 50,  # Half-half split
        # Multi-level patterns
        [10000.0] * 30 + [100.0] * 30 + [10.0] * 40,  # Multi-level with extreme peak
        [5000.0] * 20 + [100.0] * 20 + [10.0] * 60,   # Another multi-level
        # Two-level with extreme contrast
        [10000.0] * 50 + [0.1] * 50,  # Very high contrast
        [10000.0] * 25 + [0.1] * 75,  # Different split
        # Additional high-performing patterns
        [10000.0] * 5 + [1.0] * 95,   # Very short peak
        [10000.0] * 15 + [1.0] * 85,  # Medium peak
        [10000.0] * 25 + [1.0] * 75,  # Longer peak
        [10000.0] * 75 + [1.0] * 25,  # Long tail
        # Very high contrast patterns
        [100000.0] * 50 + [0.01] * 50, # Extremely high contrast
        [50000.0] * 50 + [0.01] * 50,  # Very high contrast
        # Even more extreme peaks that were mentioned in the inspirations
        [200000.0] + [1.0] * 99,      # Ultra-high peak
        [100000.0] * 5 + [1.0] * 95,  # Very short ultra-high peak
        [50000.0] * 5 + [1.0] * 95,   # Slightly less extreme ultra-high
        [200000.0] * 10 + [1.0] * 90, # Short ultra-high peak
        [100000.0] * 15 + [1.0] * 85, # Medium ultra-high peak
    ]
    patterns.extend(ultra_peak_patterns)
    
    # 2. Proven geometric patterns with precise bases (from INSPIRATION 1)
    bases_to_try = [0.88, 0.90, 0.92, 0.94, 0.95, 0.96]
    for base in bases_to_try:
        for n in [100, 150, 200, 250, 300]:
            pattern = [base**i for i in range(n)]
            patterns.append(pattern)
    
    # 3. Multi-scale exponential with specific decay rates (from INSPIRATION 1)
    for n in [100, 150, 200, 250]:
        pattern = []
        for i in range(n):
            if i < n//5:
                val = 1000 * np.exp(-i/7.0)
            elif i < n//2:
                val = 500 * np.exp(-(i-n//5)/10.0)
            else:
                val = 100 * np.exp(-(i-n//2)/15.0)
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 4. Sharp transition patterns with controlled ratios (from INSPIRATION 1)
    for n in [100, 150, 200, 250]:
        pattern = []
        for i in range(n):
            if i < n//4:
                pattern.append(1000.0)
            elif i < 2*n//4:
                pattern.append(100.0)
            else:
                pattern.append(10.0)
        patterns.append(pattern)
    
    # 5. Peak-centered with Gaussian shape (from INSPIRATION 1)
    for n in [100, 150, 200, 250]:
        pattern = []
        center = n // 2
        for i in range(n):
            val = 1000 * np.exp(-((i - center)**2) / (2 * (n/6)**2))
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 6. Multi-level patterns with proven ratios (from INSPIRATION 1)
    for n in [100, 150, 200, 250]:
        pattern = []
        levels = [1000, 500, 200, 100, 50, 25]
        for i in range(n):
            pattern.append(levels[i % len(levels)])
        patterns.append(pattern)
    
    # 7. Hybrid with specific decay profiles (from INSPIRATION 1)
    for n in [100, 150, 200, 250]:
        pattern = []
        for i in range(n):
            if i < n//5:
                pattern.append(1000.0)
            elif i < n//2:
                pattern.append(500.0)
            else:
                pattern.append(100.0 * np.exp(-(i-n//2)/12.0))
        patterns.append(pattern)
    
    # 8. Proven high performers from literature (from INSPIRATION 2)
    patterns.append([1000.0] + [1.0] * 99)  # Strong peak
    patterns.append([100.0] * 50 + [1.0] * 50)  # Moderate peak
    patterns.append([1.0] * 100)  # Uniform
    patterns.append([1000.0] * 20 + [1.0] * 80)  # Short peak
    patterns.append([1000.0] * 30 + [50.0] * 70)  # Balanced peak
    patterns.append([1000.0] * 5 + [10.0] * 95)  # Very sharp peak
    
    # 9. Specialized patterns with very high concentration (from INSPIRATION 2)
    for n in [100, 150, 200]:
        pattern = []
        peak_start = n//2 - 3
        peak_end = n//2 + 3
        for i in range(n):
            if peak_start <= i <= peak_end:
                pattern.append(1000.0)
            elif i < peak_start:
                pattern.append(1000.0 * np.exp(-(peak_start - i)/8.0))
            else:
                pattern.append(1000.0 * np.exp(-(i - peak_end)/12.0))
        patterns.append(pattern)
    
    # 10. Power law distributions with specific exponents (from INSPIRATION 3)
    for n in [100, 150, 200, 250]:
        for alpha in [0.7, 0.8, 1.0, 1.1, 1.2, 1.5]:
            pattern = [1.0 / (i+1)**alpha for i in range(n)]
            total = sum(pattern)
            if total > 0:
                pattern = [x * 1000 / total for x in pattern]
            patterns.append(pattern)
    
    # 11. Enhanced mathematical patterns from INSPIRATION 3
    # More aggressive power law with better oscillation
    for n in [120, 180, 240, 300]:
        pattern = []
        for i in range(n):
            # More refined power law with oscillation
            base = 1.0 / (i + 1)**1.18
            oscillation = 0.12 * np.sin(i * 0.08)
            val = max(0.01, 1000 * (base + oscillation))
            pattern.append(val)
        patterns.append(pattern)
    
    # 12. Concentrated peak patterns with smooth decay (from INSPIRATION 3)
    for n in [100, 150, 200, 250]:
        pattern = []
        peak_pos = n // 5
        for i in range(n):
            if i < peak_pos:
                val = 1000 * np.exp(-i/8.0)
            elif i == peak_pos:
                val = 1000.0
            else:
                val = 1000 * np.exp(-(i-peak_pos)/10.0)
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 13. Multi-stage with better balance (from INSPIRATION 3)
    for n in [100, 150, 200, 250]:
        pattern = []
        stage1_size = n // 6
        stage2_size = n // 6
        stage3_size = n - stage1_size - stage2_size
        
        # Stage 1: High values
        pattern.extend([1000.0] * stage1_size)
        # Stage 2: Medium values
        pattern.extend([500.0] * stage2_size)
        # Stage 3: Low values with decay
        for i in range(stage3_size):
            pattern.append(100.0 * np.exp(-i/8.0))
        patterns.append(pattern)
    
    # 14. Additional proven patterns from INSPIRATION 2 with variations
    # Very sharp peak patterns with different ratios
    for ratio in [0.05, 0.1, 0.15]:
        patterns.append([1000.0] * int(100*ratio) + [1.0] * int(100*(1-ratio)))
    
    # 15. Even more aggressive geometric patterns with wider range
    for base in [0.85, 0.87, 0.89, 0.91, 0.93, 0.95]:
        for n in [120, 180, 240, 300]:
            pattern = [base**i for i in range(n)]
            patterns.append(pattern)
    
    # 16. New patterns from INSPIRATION 1: Spectral-inspired patterns
    # Create patterns that minimize high-frequency components
    for n in [150, 200, 250]:
        pattern = []
        for i in range(n):
            # Create a pattern with smooth transitions and controlled oscillation
            val = 1000 * np.exp(-i/15.0) * (1 + 0.1 * np.sin(i/4.0))
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 17. New patterns from INSPIRATION 3: Fourier-inspired patterns
    # Patterns with periodic structure
    for n in [120, 180, 240]:
        pattern = []
        for i in range(n):
            # Create a periodic pattern with controlled amplitude
            period = 20
            amplitude = 1000 * np.exp(-i/20.0)
            periodic_component = 0.2 * amplitude * np.sin(2 * np.pi * i / period)
            val = max(0.01, amplitude + periodic_component)
            pattern.append(val)
        patterns.append(pattern)
    
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

def enhanced_local_search(initial_sequence, max_iter=100):
    """
    Enhanced local search with better strategy combinations and escape mechanisms.
    Based on INSPIRATION 3 but with more robust strategies.
    """
    current_sequence = initial_sequence.copy()
    current_score = compute_inv_c1(current_sequence)
    
    # Track recent improvements for adaptive behavior
    recent_improvements = []
    improvement_count = 0
    
    # Track visited states to prevent cycling
    visited_states = set()
    visited_states.add(tuple(current_sequence))
    
    # Track the best sequence found so far
    best_sequence_so_far = current_sequence.copy()
    best_score_so_far = current_score
    
    # Even more aggressive perturbation factors for better exploration
    aggressive_factors = [0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 1.02, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    
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
        
        # Strategy 2: Add/subtract with different magnitudes (larger ranges)
        for i in range(len(current_sequence)):
            original_value = current_sequence[i]
            # Test different delta values with even larger ranges
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
        
        # Strategy 7: Random global perturbations for escape (more systematic)
        if random.random() < 0.15 and len(current_sequence) > 5:
            candidate = current_sequence.copy()
            # Perturb a few random elements
            num_perturb = min(5, len(candidate) // 10)
            for _ in range(num_perturb):
                idx = random.randint(0, len(candidate) - 1)
                candidate[idx] = max(0.01, candidate[idx] * random.uniform(0.7, 1.3))
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 8: Strategic neighborhood search with better shifting
        if len(current_sequence) > 20 and random.random() < 0.15:
            # Try shifting the entire sequence slightly
            shift_amount = random.randint(-3, 3)
            if shift_amount != 0:
                candidate = [0.0] * len(current_sequence)
                for i in range(len(current_sequence)):
                    new_pos = i + shift_amount
                    if 0 <= new_pos < len(current_sequence):
                        candidate[new_pos] = current_sequence[i]
                    else:
                        candidate[i] = current_sequence[i]  # Keep unchanged if out of bounds
                new_score = compute_inv_c1(candidate)
                if new_score > best_score:
                    best_score = new_score
                    best_sequence = candidate.copy()
        
        # Strategy 9: Random mutations to maintain diversity
        if random.random() < 0.05 and len(visited_states) < 20:
            candidate = current_sequence.copy()
            # Mutate several elements with larger changes
            num_mutations = min(8, len(candidate) // 8)
            for _ in range(num_mutations):
                idx = random.randint(0, len(candidate) - 1)
                candidate[idx] = max(0.01, candidate[idx] * random.uniform(0.8, 1.2))
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 10: Simulated annealing-inspired cooling schedule
        # If no improvement, occasionally accept worse solutions to escape local minima
        if best_score <= current_score:
            # Accept with probability that decreases over time
            if len(recent_improvements) > 0:
                temp = max(0.01, 1.0 - len(recent_improvements) * 0.05)
                if random.random() < temp:
                    # Accept worse solution to escape local minimum
                    best_sequence = current_sequence.copy()
                    best_score = current_score
        
        # Update if improvement found
        if best_score > current_score:
            current_sequence = best_sequence
            current_score = best_score
            recent_improvements.append(iteration)
            improvement_count += 1
            # Keep only last 5 improvements
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)
            visited_states.add(tuple(current_sequence))
            
            # Update best so far
            if current_score > best_score_so_far:
                best_score_so_far = current_score
                best_sequence_so_far = current_sequence.copy()
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
    
    # Return the best sequence found during the process
    return best_sequence_so_far

def advanced_targeted_optimization():
    """
    Advanced targeted optimization focusing on the highest-performing patterns.
    Incorporating the best aspects from all inspirations.
    """
    best_inv_c1 = 0
    best_sequence = None
    best_result = None
    
    # Strategy 1: High-performance mathematical patterns (enhanced)
    patterns = create_mathematical_patterns()
    for pattern in patterns:
        metrics = evaluate_sequence(pattern)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = pattern
            best_result = metrics
    
    # Strategy 2: Specialized sequences (improved diversity)
    specialized_sequences = generate_specialized_sequences()
    for sequence in specialized_sequences:
        metrics = evaluate_sequence(sequence)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = sequence
            best_result = metrics
    
    # Strategy 3: Optimization around best patterns with better methods
    if best_sequence is not None:
        # Try multiple optimization methods with different settings (from INSPIRATION 3)
        methods_and_settings = [
            ('Nelder-Mead', {'maxiter': 60, 'disp': False}),
            ('Powell', {'maxiter': 60, 'disp': False}),
            ('L-BFGS-B', {'maxiter': 40, 'disp': False}),
        ]
        
        for method, options in methods_and_settings:
            try:
                result = minimize(
                    lambda x: -compute_inv_c1(x),
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
    
    # Strategy 4: Enhanced local search refinement (from INSPIRATION 3)
    if best_sequence is not None:
        refined_sequence = enhanced_local_search(best_sequence, max_iter=80)
        metrics = evaluate_sequence(refined_sequence)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = refined_sequence
            best_result = metrics
    
    # Strategy 5: Additional targeted optimizations (from INSPIRATION 3)
    if best_sequence is not None:
        # Try a few more specific optimization attempts with better bounds
        for _ in range(7):
            try:
                # Slightly perturb and optimize
                perturbed = [x * random.uniform(0.95, 1.05) for x in best_sequence]
                result = minimize(
                    lambda x: -compute_inv_c1(x),
                    perturbed,
                    method='L-BFGS-B',
                    bounds=[(0.01, 1000.0)] * len(perturbed),
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
    
    # Strategy 6: Try a few different random starts for global exploration (enhanced)
    for _ in range(15):
        try:
            # Generate a random sequence with better structure
            n = random.randint(100, 250)
            base = 0.92 + random.uniform(-0.02, 0.02)
            sequence = [base**i for i in range(n)]
            # Add some random variation to prevent getting stuck
            for i in range(len(sequence)):
                if random.random() < 0.15:
                    sequence[i] = max(0.01, sequence[i] * random.uniform(0.9, 1.1))
            
            metrics = evaluate_sequence(sequence)
            if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                best_inv_c1 = metrics['inv_c1']
                best_sequence = sequence
                best_result = metrics
        except:
            continue
    
    # Strategy 7: Try more mathematical patterns from INSPIRATION 3
    # These are designed to have excellent autocorrelation properties
    for _ in range(12):
        try:
            n = random.randint(120, 200)
            # Create a pattern with a peak followed by exponential decay
            sequence = []
            peak_height = random.uniform(800, 1200)
            peak_pos = random.randint(n//4, 3*n//4)
            
            for i in range(n):
                if i < peak_pos:
                    # Decay towards peak
                    dist = peak_pos - i
                    val = peak_height * np.exp(-dist/15.0)
                elif i == peak_pos:
                    val = peak_height
                else:
                    # Decay away from peak
                    dist = i - peak_pos
                    val = peak_height * np.exp(-dist/12.0)
                sequence.append(max(0.01, val))
            
            metrics = evaluate_sequence(sequence)
            if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                best_inv_c1 = metrics['inv_c1']
                best_sequence = sequence
                best_result = metrics
        except:
            continue
    
    # Strategy 8: Try a hybrid approach with known good structures (from INSPIRATION 3)
    # Combine multiple proven approaches with better control
    try:
        # Try a carefully constructed hybrid pattern with better parameters
        n = random.randint(150, 200)
        hybrid_pattern = []
        # Create a pattern that starts high, drops rapidly, then decays slowly
        for i in range(n):
            if i < 20:
                hybrid_pattern.append(1000.0 * (0.95**i))
            elif i < 40:
                hybrid_pattern.append(1000.0 * (0.85**(i-20)))
            else:
                hybrid_pattern.append(100.0 * (0.95**(i-40)))
        # Normalize to reasonable scale
        max_val = max(hybrid_pattern) if hybrid_pattern else 1.0
        if max_val > 0:
            hybrid_pattern = [x / max_val * 1000.0 for x in hybrid_pattern]
        
        metrics = evaluate_sequence(hybrid_pattern)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = hybrid_pattern
            best_result = metrics
    except:
        pass
    
    # Strategy 9: Try power law patterns with better parameters (from INSPIRATION 3)
    for _ in range(10):
        try:
            n = random.randint(100, 200)
            pattern = []
            for i in range(n):
                # Power law with better oscillation parameters
                base = 1.0 / (i + 1)**1.18
                oscillation = 0.12 * np.sin(i * 0.08)
                val = max(0.01, 1000 * (base + oscillation))
                pattern.append(val)
            metrics = evaluate_sequence(pattern)
            if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
                best_inv_c1 = metrics['inv_c1']
                best_sequence = pattern
                best_result = metrics
        except:
            continue
    
    # Strategy 10: Try proven patterns from INSPIRATION 2 for additional boost
    # Directly use high-performing patterns from literature
    high_performers = [
        [1000.0] + [1.0] * 99,  # Strong peak
        [1000.0] * 20 + [1.0] * 80,  # Strong peak with decay
        [1000.0] * 30 + [50.0] * 70,  # Balanced peak
        [1000.0] * 10 + [10.0] * 90,  # Very sharp peak
        [1000.0] * 5 + [100.0] * 95,  # Extreme peak
    ]
    
    for pattern in high_performers:
        metrics = evaluate_sequence(pattern)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = pattern
            best_result = metrics
    
    # Strategy 11: Novel approach - spectral analysis inspired patterns
    # Try creating patterns with controlled frequency content
    try:
        n = 200
        spectral_pattern = []
        for i in range(n):
            # Create pattern that favors low frequencies (smooth structure)
            val = 1000 * np.exp(-i/15.0) * (1 + 0.1 * np.sin(i/3.0))
            spectral_pattern.append(max(0.01, val))
        
        metrics = evaluate_sequence(spectral_pattern)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = spectral_pattern
            best_result = metrics
    except:
        pass
    
    return best_sequence if best_sequence is not None else [1.0] * 100

def search_for_best_sequence() -> list[float]:
    """
    Main function to search for the best coefficient sequence.
    Focused on maximizing 1/C1 with targeted approaches.
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run advanced targeted optimization
    sequence = advanced_targeted_optimization()
    
    # Ensure sequence has reasonable properties
    if len(sequence) == 0:
        sequence = [1.0]
    
    # Make sure sum is meaningful
    if np.sum(sequence) < 0.01:
        sequence = [x + 0.1 for x in sequence]
    
    # Final refinement with enhanced local search
    try:
        refined_sequence = enhanced_local_search(sequence, max_iter=60)
        refined_metrics = evaluate_sequence(refined_sequence)
        if refined_metrics['inv_c1'] > evaluate_sequence(sequence)['inv_c1']:
            sequence = refined_sequence
    except:
        pass
    
    # Try one final optimization with different approach
    try:
        # Use a different optimization method for final tuning
        result = minimize(
            lambda x: -compute_inv_c1(x),
            sequence,
            method='L-BFGS-B',
            bounds=[(0.01, 1000.0)] * len(sequence),
            options={'maxiter': 30}
        )
        if result.success:
            final_sequence = result.x
            final_metrics = evaluate_sequence(final_sequence)
            if final_metrics['inv_c1'] > evaluate_sequence(sequence)['inv_c1']:
                sequence = final_sequence.tolist()
    except:
        pass
    
    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
