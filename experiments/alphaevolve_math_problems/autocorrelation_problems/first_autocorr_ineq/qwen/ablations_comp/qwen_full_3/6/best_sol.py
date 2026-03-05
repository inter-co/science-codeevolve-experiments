# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import fftconvolve, convolve
from scipy.optimize import minimize
import time

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
    
    # Use FFT for efficient convolution when sequence is large (> 100 elements)
    # For smaller sequences, use direct convolution for numerical precision
    if len(sequence) > 100:
        conv = fftconvolve(sequence, sequence, mode='full')
    else:
        conv = convolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    
    # Avoid division by very small numbers
    if max_conv <= 1e-15:
        return float('inf')
    
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

def create_high_performance_patterns():
    """Create patterns known to produce high-quality results based on mathematical analysis."""
    patterns = []
    
    # 1. Optimized geometric decay with expanded base range and more diverse lengths
    for base in [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]:
        for n in [80, 100, 120, 150, 180, 200, 250, 300]:
            pattern = [base**i for i in range(n)]
            patterns.append(pattern)
    
    # 2. Multi-scale exponential with stronger peak emphasis
    for n in [100, 120, 150, 180, 200]:
        pattern = []
        for i in range(n):
            if i < n//4:
                val = 1000 * np.exp(-i/6.0)  # Even faster decay at start
            elif i < n//2:
                val = 500 * np.exp(-(i-n//4)/10.0)  # Mid-range
            else:
                val = 100 * np.exp(-(i-n//2)/15.0)  # Slower decay at end
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 3. Sharp transition patterns with more aggressive transitions
    for n in [100, 120, 150, 180, 200]:
        # High values at start, then drop sharply
        pattern = []
        for i in range(n):
            if i < n//3:
                pattern.append(1000.0)
            elif i < 2*n//3:
                pattern.append(100.0)
            else:
                pattern.append(10.0)
        patterns.append(pattern)
        
        # Alternative: very aggressive drop
        pattern = []
        for i in range(n):
            if i < n//4:
                pattern.append(1000.0)
            elif i < n//2:
                pattern.append(200.0)
            else:
                pattern.append(10.0)
        patterns.append(pattern)
        
        # Alternative: more gradual drop
        pattern = []
        for i in range(n):
            if i < n//4:
                pattern.append(1000.0)
            elif i < n//2:
                pattern.append(500.0)
            else:
                pattern.append(100.0)
        patterns.append(pattern)
    
    # 4. Peak-centered with Gaussian shape (tighter variance)
    for n in [100, 120, 150, 180, 200]:
        pattern = []
        center = n // 2
        for i in range(n):
            # Gaussian-like peak with sharper drop-off
            val = 1000 * np.exp(-((i - center)**2) / (2 * (n/4)**2))
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 5. Multi-level with specific ratios
    for n in [100, 120, 150, 180, 200]:
        # Original pattern
        pattern = []
        levels = [1000, 500, 250, 100, 50]
        for i in range(n):
            pattern.append(levels[i % len(levels)])
        patterns.append(pattern)
        
        # Alternative: more balanced levels
        levels = [1000, 700, 400, 200, 100]
        for i in range(n):
            pattern.append(levels[i % len(levels)])
        patterns.append(pattern)
        
        # Alternative: higher initial concentration
        levels = [1000, 800, 600, 400, 200]
        for i in range(n):
            pattern.append(levels[i % len(levels)])
        patterns.append(pattern)
    
    # 6. Hybrid with high early values and controlled decay
    for n in [100, 120, 150, 180, 200]:
        # Standard hybrid
        pattern = []
        for i in range(n):
            if i < n//4:
                pattern.append(1000.0)
            elif i < n//2:
                pattern.append(500.0)
            else:
                pattern.append(100.0 * np.exp(-(i-n//2)/12.0))
        patterns.append(pattern)
        
        # Alternative: different decay rate
        pattern = []
        for i in range(n):
            if i < n//3:
                pattern.append(1000.0)
            elif i < 2*n//3:
                pattern.append(500.0)
            else:
                pattern.append(100.0 * np.exp(-(i-2*n//3)/18.0))
        patterns.append(pattern)
        
        # Alternative: very fast decay
        pattern = []
        for i in range(n):
            if i < n//3:
                pattern.append(1000.0)
            elif i < 2*n//3:
                pattern.append(500.0)
            else:
                pattern.append(100.0 * np.exp(-(i-2*n//3)/10.0))
        patterns.append(pattern)
    
    # 7. Specific high performers from previous tests (with more variations)
    patterns.append([1000.0] + [1.0] * 99)  # Strong peak
    patterns.append([100.0] * 50 + [1.0] * 50)  # Moderate peak
    patterns.append([1.0] * 100)  # Uniform
    patterns.append([1000.0] * 20 + [1.0] * 80)  # Short peak
    patterns.append([1000.0] * 30 + [50.0] * 70)  # Balanced peak
    patterns.append([1000.0] * 10 + [100.0] * 90)  # Concentrated peak
    
    # 8. Optimized peak patterns with known good ratios
    for n in [100, 120, 150, 180, 200]:
        # Base pattern with standard decay
        pattern = [0.0] * n
        peak_height = 1000.0
        for i in range(n):
            if i < 20:
                pattern[i] = peak_height
            else:
                pattern[i] = peak_height * (0.95 ** (i - 20))
        patterns.append(pattern)
        
        # Alternative: different decay rate
        pattern = [0.0] * n
        peak_height = 1000.0
        for i in range(n):
            if i < 20:
                pattern[i] = peak_height
            else:
                pattern[i] = peak_height * (0.93 ** (i - 20))
        patterns.append(pattern)
        
        # Alternative: faster decay
        pattern = [0.0] * n
        peak_height = 1000.0
        for i in range(n):
            if i < 20:
                pattern[i] = peak_height
            else:
                pattern[i] = peak_height * (0.9 ** (i - 20))
        patterns.append(pattern)
        
        # Alternative: slower decay
        pattern = [0.0] * n
        peak_height = 1000.0
        for i in range(n):
            if i < 20:
                pattern[i] = peak_height
            else:
                pattern[i] = peak_height * (0.97 ** (i - 20))
        patterns.append(pattern)
        
    # 9. High-performance hybrid patterns based on mathematical analysis
    for n in [120, 150, 180, 200]:
        # Create a pattern with a sharp initial peak followed by rapid decay
        pattern = []
        for i in range(n):
            if i < 10:
                pattern.append(1000.0)
            elif i < 25:
                pattern.append(1000.0 * 0.8**(i-10))
            elif i < 50:
                pattern.append(100.0 * 0.9**(i-25))
            else:
                pattern.append(10.0 * 0.95**(i-50))
        patterns.append(pattern)
        
        # Alternative: different decay rates
        pattern = []
        for i in range(n):
            if i < 15:
                pattern.append(1000.0)
            elif i < 35:
                pattern.append(1000.0 * 0.85**(i-15))
            elif i < 65:
                pattern.append(100.0 * 0.9**(i-35))
            else:
                pattern.append(10.0 * 0.92**(i-65))
        patterns.append(pattern)
        
        # Alternative: very concentrated early decay
        pattern = []
        for i in range(n):
            if i < 5:
                pattern.append(1000.0)
            elif i < 20:
                pattern.append(1000.0 * 0.9**(i-5))
            elif i < 45:
                pattern.append(100.0 * 0.9**(i-20))
            else:
                pattern.append(10.0 * 0.95**(i-45))
        patterns.append(pattern)
    
    # 10. Power-law decay with strategic enhancement
    for n in [100, 120, 150, 180, 200]:
        pattern = []
        for i in range(n):
            # Power law decay with added oscillation for better structure
            base = 1.0 / (i + 1)**1.2
            oscillation = 0.1 * np.sin(i * 0.1)
            val = max(0.01, 1000 * (base + oscillation))
            pattern.append(val)
        patterns.append(pattern)
        
        # Alternative: different exponent and oscillation
        pattern = []
        for i in range(n):
            base = 1.0 / (i + 1)**1.5
            oscillation = 0.05 * np.sin(i * 0.15)
            val = max(0.01, 1000 * (base + oscillation))
            pattern.append(val)
        patterns.append(pattern)
        
        # Alternative: higher oscillation
        pattern = []
        for i in range(n):
            base = 1.0 / (i + 1)**1.3
            oscillation = 0.2 * np.sin(i * 0.12)
            val = max(0.01, 1000 * (base + oscillation))
            pattern.append(val)
        patterns.append(pattern)
    
    # 11. Advanced patterns inspired by literature
    # Very aggressive peak with controlled tail
    for n in [100, 120, 150, 180, 200]:
        pattern = [0.0] * n
        peak_height = 1000.0
        for i in range(n):
            if i < 15:
                pattern[i] = peak_height
            elif i < 30:
                pattern[i] = peak_height * (0.95**(i-15))
            else:
                pattern[i] = peak_height * (0.9**(i-30)) * 0.1
        patterns.append(pattern)
        
        # Alternative: different peak location
        pattern = [0.0] * n
        peak_height = 1000.0
        for i in range(n):
            if i < 10:
                pattern[i] = peak_height
            elif i < 25:
                pattern[i] = peak_height * (0.9**(i-10))
            else:
                pattern[i] = peak_height * (0.92**(i-25)) * 0.15
        patterns.append(pattern)
        
        # Alternative: even more aggressive peak
        pattern = [0.0] * n
        peak_height = 1000.0
        for i in range(n):
            if i < 8:
                pattern[i] = peak_height
            elif i < 20:
                pattern[i] = peak_height * (0.95**(i-8))
            else:
                pattern[i] = peak_height * (0.9**(i-20)) * 0.1
        patterns.append(pattern)
    
    # 12. Multi-peak pattern with mathematical optimization
    for n in [120, 150, 180, 200]:
        pattern = []
        num_peaks = 3
        peak_positions = [n//5, n//2, 4*n//5]
        peak_heights = [1000.0, 800.0, 600.0]
        for i in range(n):
            min_dist = min(abs(i - pos) for pos in peak_positions)
            peak_idx = np.argmin([abs(i - pos) for pos in peak_positions])
            peak_height = peak_heights[peak_idx]
            # Gaussian-like decay from peak
            val = peak_height * np.exp(-min_dist/8.0)
            pattern.append(max(0.01, val))
        patterns.append(pattern)
        
        # Alternative: more evenly spaced peaks
        peak_positions = [n//6, n//2, 5*n//6]
        pattern = []
        for i in range(n):
            min_dist = min(abs(i - pos) for pos in peak_positions)
            peak_idx = np.argmin([abs(i - pos) for pos in peak_positions])
            peak_height = peak_heights[peak_idx]
            val = peak_height * np.exp(-min_dist/6.0)
            pattern.append(max(0.01, val))
        patterns.append(pattern)
        
        # Alternative: different peak heights
        peak_heights = [1200.0, 900.0, 600.0]
        pattern = []
        for i in range(n):
            min_dist = min(abs(i - pos) for pos in peak_positions)
            peak_idx = np.argmin([abs(i - pos) for pos in peak_positions])
            peak_height = peak_heights[peak_idx]
            val = peak_height * np.exp(-min_dist/8.0)
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 13. Optimized "spike and decay" pattern - critical for beating benchmarks
    for n in [120, 150, 180, 200]:
        # Start with a large spike, then decay quickly
        pattern = []
        for i in range(n):
            if i < 10:
                pattern.append(1000.0)
            elif i < 25:
                pattern.append(1000.0 * (0.9**(i-10)))
            elif i < 50:
                pattern.append(100.0 * (0.9**(i-25)))
            else:
                pattern.append(10.0 * (0.95**(i-50)))
        patterns.append(pattern)
        
        # Alternative: different timing
        pattern = []
        for i in range(n):
            if i < 5:
                pattern.append(1000.0)
            elif i < 15:
                pattern.append(1000.0 * (0.95**(i-5)))
            elif i < 40:
                pattern.append(100.0 * (0.9**(i-15)))
            else:
                pattern.append(10.0 * (0.9**(i-40)))
        patterns.append(pattern)
        
        # Alternative: more aggressive initial decay
        pattern = []
        for i in range(n):
            if i < 8:
                pattern.append(1000.0)
            elif i < 20:
                pattern.append(1000.0 * (0.9**(i-8)))
            elif i < 45:
                pattern.append(100.0 * (0.9**(i-20)))
            else:
                pattern.append(10.0 * (0.95**(i-45)))
        patterns.append(pattern)
    
    # 14. "Heavy tail" patterns for better convolution properties
    for n in [100, 120, 150, 180, 200]:
        # Heavy tail with early rapid decay
        pattern = []
        for i in range(n):
            if i < 20:
                pattern.append(1000.0)
            elif i < 40:
                pattern.append(1000.0 * (0.8**(i-20)))
            else:
                # Very long tail
                pattern.append(100.0 * (0.95**(i-40)))
        patterns.append(pattern)
        
        # Alternative: different decay rates
        pattern = []
        for i in range(n):
            if i < 15:
                pattern.append(1000.0)
            elif i < 35:
                pattern.append(1000.0 * (0.85**(i-15)))
            else:
                pattern.append(100.0 * (0.9**(i-35)))
        patterns.append(pattern)
        
        # Alternative: even longer tail
        pattern = []
        for i in range(n):
            if i < 10:
                pattern.append(1000.0)
            elif i < 25:
                pattern.append(1000.0 * (0.9**(i-10)))
            else:
                pattern.append(100.0 * (0.92**(i-25)))
        patterns.append(pattern)
        
        # Alternative: ultra-long tail
        pattern = []
        for i in range(n):
            if i < 12:
                pattern.append(1000.0)
            elif i < 30:
                pattern.append(1000.0 * (0.85**(i-12)))
            else:
                pattern.append(100.0 * (0.9**(i-30)))
        patterns.append(pattern)
    
    # 15. New patterns specifically designed to maximize 1/C1
    # Patterns with very concentrated energy at start
    for n in [120, 150, 180, 200]:
        pattern = [0.0] * n
        # Concentrate high values at the beginning
        for i in range(min(30, n)):
            pattern[i] = 1000.0 * (0.95**i)
        patterns.append(pattern)
        
        # Concentrated with a sharper peak
        pattern = [0.0] * n
        pattern[0] = 1000.0
        if n > 1:
            pattern[1] = 800.0
        if n > 2:
            pattern[2] = 600.0
        for i in range(3, n):
            pattern[i] = 100.0 * (0.9**(i-3))
        patterns.append(pattern)
        
        # Very concentrated with extremely fast decay
        pattern = [0.0] * n
        pattern[0] = 1000.0
        if n > 1:
            pattern[1] = 900.0
        if n > 2:
            pattern[2] = 800.0
        for i in range(3, n):
            pattern[i] = 100.0 * (0.85**(i-3))
        patterns.append(pattern)
    
    # 16. Additional patterns focusing on specific mathematical constructions
    # Golden ratio-inspired pattern
    golden_ratio = (1 + np.sqrt(5)) / 2
    for n in [120, 150, 180]:
        pattern = []
        for i in range(n):
            pattern.append(1000.0 * (1/golden_ratio)**i)
        patterns.append(pattern)
    
    return patterns

def generate_specialized_sequences():
    """Generate sequences with mathematical properties that tend to work well."""
    sequences = []
    
    # Create sequences with specific mathematical properties
    for _ in range(50):  # Increased count for more candidates
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

def enhanced_local_search(initial_sequence, max_iter=150):
    """
    Enhanced local search with better strategy combinations and escape mechanisms.
    """
    current_sequence = initial_sequence.copy()
    current_score = compute_inv_c1(current_sequence)
    
    # Track recent improvements for adaptive behavior
    recent_improvements = []
    
    for iteration in range(max_iter):
        # Try different types of perturbations
        best_sequence = current_sequence.copy()
        best_score = current_score
        
        # Strategy 1: Fine-grained multiplicative changes with wider range
        for i in range(len(current_sequence)):
            original_value = current_sequence[i]
            # Test a wide range of factors for fine-tuning
            factors = [0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.15, 1.2, 1.25, 1.3]
            for factor in factors:
                if original_value * factor >= 0.01:
                    candidate = current_sequence.copy()
                    candidate[i] = original_value * factor
                    new_score = compute_inv_c1(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_sequence = candidate.copy()
        
        # Strategy 2: Add/subtract with different magnitudes
        for i in range(len(current_sequence)):
            original_value = current_sequence[i]
            # Test different delta values
            deltas = [-2.0, -1.5, -1.0, -0.8, -0.5, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
            for delta in deltas:
                if original_value + delta >= 0.01:
                    candidate = current_sequence.copy()
                    candidate[i] = original_value + delta
                    new_score = compute_inv_c1(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_sequence = candidate.copy()
        
        # Strategy 3: Global scaling (more aggressive)
        scale_factors = [0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 1.03, 1.05, 1.07, 1.1, 1.15, 1.2, 1.25, 1.3]
        for sf in scale_factors:
            candidate = [x * sf for x in current_sequence]
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 4: Small averaging to reduce noise
        if len(current_sequence) > 10:
            candidate = current_sequence.copy()
            for i in range(1, len(candidate) - 1):
                avg_val = (current_sequence[i-1] + current_sequence[i] + current_sequence[i+1]) / 3.0
                candidate[i] = avg_val
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 5: Window averaging for better smoothing
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
        
        # Strategy 6: Random global perturbations for escape
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
        
        # Strategy 7: Strategic neighborhood search with shift
        if len(current_sequence) > 20 and random.random() < 0.2:
            # Try shifting the entire sequence slightly
            shift_amount = random.randint(-5, 5)
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
        
        # Strategy 8: Swap two elements for diversity
        if len(current_sequence) > 10 and random.random() < 0.1:
            idx1, idx2 = random.sample(range(len(current_sequence)), 2)
            candidate = current_sequence.copy()
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            new_score = compute_inv_c1(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 9: Add small random noise for further exploration
        if random.random() < 0.08:
            candidate = current_sequence.copy()
            for i in range(len(candidate)):
                if random.random() < 0.2:  # Apply noise to 20% of elements
                    noise_factor = random.uniform(0.9, 1.1)
                    candidate[i] = max(0.01, candidate[i] * noise_factor)
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
            # Adaptive escape mechanism with better probability calculation
            if len(recent_improvements) > 0:
                # If we haven't improved recently, be more aggressive
                escape_prob = 0.2 + 0.15 * len(recent_improvements)
            else:
                escape_prob = 0.15
                
            if random.random() < escape_prob:
                # Random perturbation with larger magnitude
                idx = random.randint(0, len(current_sequence) - 1)
                current_sequence[idx] = max(0.01, current_sequence[idx] * random.uniform(0.6, 1.4))
                current_score = compute_inv_c1(current_sequence)
    
    return current_sequence

def advanced_targeted_optimization():
    """
    Advanced targeted optimization focusing on the highest-performing patterns.
    """
    best_inv_c1 = 0
    best_sequence = None
    best_result = None
    
    # Strategy 1: High-performance mathematical patterns
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
    
    # Strategy 3: Optimization around best patterns with better methods
    if best_sequence is not None:
        # Try multiple optimization methods with different settings
        methods_and_settings = [
            ('Nelder-Mead', {'maxiter': 60, 'disp': False}),
            ('Powell', {'maxiter': 60, 'disp': False}),
            ('L-BFGS-B', {'maxiter': 50, 'disp': False}),
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
    
    # Strategy 4: Enhanced local search refinement
    if best_sequence is not None:
        refined_sequence = enhanced_local_search(best_sequence, max_iter=120)
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
            from scipy.optimize import differential_evolution
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=30,
                popsize=25,
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
        for _ in range(7):
            try:
                # Slightly perturb and optimize with bounds
                perturbed = [x * random.uniform(0.95, 1.05) for x in best_sequence]
                result = minimize(
                    lambda x: -compute_inv_c1(x),
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 40, 'disp': False}
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
    
    # Strategy 6: Try a few different random starts for global exploration
    for _ in range(12):
        try:
            # Generate a random sequence with better structure
            n = random.randint(100, 200)
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
    
    # Strategy 7: Try a few more specific mathematical patterns
    # These are designed to have excellent autocorrelation properties
    for _ in range(10):
        try:
            n = random.randint(120, 180)
            # Create a pattern with a peak followed by exponential decay
            sequence = []
            peak_height = random.uniform(800, 1200)
            peak_pos = random.randint(n//3, 2*n//3)
            
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
    
    # Strategy 8: Try a hybrid approach with known good structures
    # Combine multiple proven approaches
    try:
        # Try a carefully constructed hybrid pattern
        n = random.randint(150, 200)
        hybrid_pattern = []
        # Create a pattern that starts high, drops rapidly, then decays slowly
        for i in range(n):
            if i < 30:
                hybrid_pattern.append(1000.0 * (0.95**i))
            elif i < 60:
                hybrid_pattern.append(1000.0 * (0.85**(i-30)))
            else:
                hybrid_pattern.append(100.0 * (0.95**(i-60)))
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
    
    # Strategy 9: Try mathematical patterns with specific structure
    try:
        # Try a multi-peak pattern
        n = random.randint(120, 180)
        pattern = []
        peak_positions = [n//4, n//2, 3*n//4]
        peak_heights = [1000.0, 800.0, 600.0]
        for i in range(n):
            min_dist = min(abs(i - pos) for pos in peak_positions)
            peak_idx = np.argmin([abs(i - pos) for pos in peak_positions])
            peak_height = peak_heights[peak_idx]
            # Gaussian-like decay from peak
            val = peak_height * np.exp(-min_dist/10.0)
            pattern.append(max(0.01, val))
        
        metrics = evaluate_sequence(pattern)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = pattern
            best_result = metrics
    except:
        pass
    
    # Strategy 10: Try a very specific "heavy tail" pattern that's been shown to work well
    try:
        n = 180
        heavy_tail_pattern = []
        # Heavy tail with early rapid decay
        for i in range(n):
            if i < 20:
                heavy_tail_pattern.append(1000.0)
            elif i < 50:
                heavy_tail_pattern.append(1000.0 * (0.8**(i-20)))
            else:
                # Very long tail - this is the key for better C1
                heavy_tail_pattern.append(100.0 * (0.95**(i-50)))
        
        metrics = evaluate_sequence(heavy_tail_pattern)
        if metrics['inv_c1'] > best_inv_c1 and metrics['sum'] > 0.01:
            best_inv_c1 = metrics['inv_c1']
            best_sequence = heavy_tail_pattern
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
        refined_sequence = enhanced_local_search(sequence, max_iter=100)
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
            options={'maxiter': 50}
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
