# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple
import math
from collections import deque

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def evaluate_sequence(individual: List[float]) -> float:
    """
    Evaluate a sequence by computing 1/C₁ where C₁ = 2n * max(convolution) / (sum)^2
    We want to maximize 1/C₁, so we return the value directly.
    """
    if len(individual) == 0:
        return 0.0
    
    # Remove zeros to avoid degenerate cases and clip values
    individual = [max(min(x, 1000.0), 1e-10) for x in individual]
    
    # Compute sum and normalize
    s = sum(individual)
    if s < 0.01:
        return 0.0
    
    # Use FFT for fast convolution - more efficient for larger sequences
    a = np.array(individual)
    
    # Compute convolution using FFT (faster than direct method)
    b = fftconvolve(a, a, mode='full')
    # Take only the relevant part (the actual convolution, not full)
    b = b[:len(a)*2-1]
    
    # Find maximum value in convolution
    max_b = np.max(b)
    
    # Calculate C₁
    n = len(individual)
    if max_b <= 0:
        return 0.0
    
    c1 = 2 * n * max_b / (s ** 2)
    
    # We want to maximize 1/C₁, so return 1/c1
    return 1.0 / c1 if c1 > 0 else 0.0

def generate_geometric_sequence(length: int, ratio: float = 0.92) -> List[float]:
    """Generate a geometrically decreasing sequence."""
    return [ratio**i for i in range(length)]

def generate_sparse_sequence(length: int, sparsity: float = 0.15) -> List[float]:
    """Generate a sparse sequence with few non-zero elements."""
    sequence = [0.0] * length
    num_nonzero = max(1, int(length * sparsity))
    indices = random.sample(range(length), num_nonzero)
    for idx in indices:
        sequence[idx] = random.uniform(100, 1000)
    return sequence

def generate_symmetric_sequence(length: int) -> List[float]:
    """Generate a symmetric sequence."""
    sequence = []
    mid = length // 2
    for i in range(mid + 1):
        val = random.uniform(100, 1000)
        sequence.append(val)
    if length % 2 == 0:
        sequence.extend(reversed(sequence[:-1]))
    else:
        sequence.extend(reversed(sequence))
    return sequence

def generate_chebyshev_sequence(length: int) -> List[float]:
    """Generate a sequence based on Chebyshev polynomial roots."""
    # Chebyshev nodes scaled to [0, 1]
    nodes = [(1 - np.cos(np.pi * i / (length - 1))) / 2 for i in range(length)]
    # Map to [100, 1000] and make it decreasing
    sequence = [1000 - 900 * node for node in nodes]
    return sequence

def generate_power_law_sequence(length: int, alpha: float = 1.5) -> List[float]:
    """Generate a power-law decreasing sequence."""
    sequence = []
    for i in range(length):
        # Power law: x^(-alpha) where alpha > 0
        val = 1000.0 * (i + 1) ** (-alpha)
        sequence.append(max(val, 1e-10))
    return sequence

def generate_gaussian_sequence(length: int, std_dev_factor: float = 0.2) -> List[float]:
    """Generate a Gaussian-like sequence centered around middle."""
    sequence = []
    center = length // 2
    std_dev = std_dev_factor * length
    for i in range(length):
        # Gaussian centered at center
        val = 1000.0 * np.exp(-0.5 * ((i - center) / std_dev) ** 2)
        sequence.append(max(val, 1e-10))
    return sequence

def generate_sine_sequence(length: int, frequency: float = 0.5) -> List[float]:
    """Generate a sine wave-like sequence."""
    sequence = []
    for i in range(length):
        val = 500.0 + 500.0 * np.sin(2 * np.pi * i * frequency / length)
        sequence.append(max(val, 1e-10))
    return sequence

def generate_concentrated_sequence(length: int, concentration: float = 0.2) -> List[float]:
    """Generate a sequence concentrated at the beginning."""
    sequence = [0.0] * length
    start_idx = int(length * (1 - concentration))
    for i in range(start_idx, length):
        # Gradually increase from low to high values
        sequence[i] = 1000.0 * (i - start_idx + 1) / (length - start_idx)
    return sequence

def generate_focused_sequence(length: int) -> List[float]:
    """Generate a sequence with a very focused peak."""
    sequence = [0.0] * length
    peak_pos = length // 2
    # Create a very sharp peak
    sequence[peak_pos] = 1000.0
    # Add a few neighbors to help with convolution
    if peak_pos > 0:
        sequence[peak_pos-1] = 200.0
    if peak_pos < length-1:
        sequence[peak_pos+1] = 200.0
    return sequence

def generate_adaptive_sequence(length: int) -> List[float]:
    """Generate an adaptive sequence based on mathematical insights."""
    # Start with a concentrated pattern that tends to perform well
    sequence = [0.0] * length
    
    # Place a strong peak near the beginning
    peak_start = max(1, length // 4)
    peak_end = min(length - 1, 3 * length // 4)
    
    # Create a sharp peak in the middle
    peak_pos = length // 2
    sequence[peak_pos] = 1000.0
    
    # Add surrounding values to enhance convolution effect
    if peak_pos > 0:
        sequence[peak_pos-1] = 500.0
    if peak_pos < length-1:
        sequence[peak_pos+1] = 500.0
    
    # Add some concentration at the beginning to boost sum
    for i in range(min(peak_start, 10)):
        sequence[i] = 1000.0 - (i * 50)
    
    return sequence

def generate_mixed_sequence(length: int) -> List[float]:
    """Generate a mixed pattern sequence."""
    # Mix different patterns
    half_length = length // 2
    part1 = generate_power_law_sequence(half_length, alpha=1.5)
    part2 = generate_gaussian_sequence(length - half_length, std_dev_factor=0.2)
    return part1 + part2

def generate_mathematical_patterns():
    """Create mathematically sound patterns based on proven high-performing sequences."""
    patterns = []
    
    # 1. Optimized geometric decay (very effective) - base 0.92, 0.93, 0.94
    for base in [0.92, 0.93, 0.94]:
        for n in [100, 150, 200, 250, 300]:
            pattern = [base**i for i in range(n)]
            patterns.append(pattern)
    
    # 2. Multi-scale exponential with strong peak (inspiration from good patterns)
    for n in [100, 150, 200, 250]:
        pattern = []
        for i in range(n):
            if i < n//4:
                val = 1000 * np.exp(-i/10.0)  # Rapid decay at start
            elif i < n//2:
                val = 500 * np.exp(-(i-n//4)/15.0)  # Mid-range
            else:
                val = 100 * np.exp(-(i-n//2)/20.0)  # Slow decay at end
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 3. Sharp transition patterns
    for n in [100, 150, 200, 250]:
        pattern = []
        # High values at start, then drop sharply
        for i in range(n):
            if i < n//3:
                pattern.append(1000.0)
            elif i < 2*n//3:
                pattern.append(100.0)
            else:
                pattern.append(10.0)
        patterns.append(pattern)
    
    # 4. Peak-centered with Gaussian shape
    for n in [100, 150, 200, 250]:
        pattern = []
        center = n // 2
        for i in range(n):
            # Gaussian-like peak with sharper drop-off
            val = 1000 * np.exp(-((i - center)**2) / (2 * (n/6)**2))
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 5. Multi-level with specific ratios (based on proven good values)
    for n in [100, 150, 200, 250]:
        pattern = []
        levels = [1000, 500, 250, 100, 50]
        for i in range(n):
            pattern.append(levels[i % len(levels)])
        patterns.append(pattern)
    
    # 6. Hybrid with high early values and controlled decay
    for n in [100, 150, 200, 250]:
        pattern = []
        for i in range(n):
            if i < n//4:
                pattern.append(1000.0)
            elif i < n//2:
                pattern.append(500.0)
            else:
                pattern.append(100.0 * np.exp(-(i-n//2)/20.0))
        patterns.append(pattern)
    
    # 7. Specific high performers from previous tests
    patterns.append([1000.0] + [1.0] * 99)  # Strong peak
    patterns.append([100.0] * 50 + [1.0] * 50)  # Moderate peak
    patterns.append([1.0] * 100)  # Uniform
    patterns.append([1000.0] * 20 + [1.0] * 80)  # Short peak
    
    # 8. Optimized peak patterns with known good ratios
    for n in [100, 150, 200, 250]:
        pattern = [0.0] * n
        # Place a strong peak at the beginning with gradual decay
        peak_height = 1000.0
        for i in range(n):
            if i < 20:
                pattern[i] = peak_height
            else:
                pattern[i] = peak_height * (0.95 ** (i - 20))
        patterns.append(pattern)
        
    # 9. High-performance hybrid patterns based on mathematical analysis
    for n in [120, 180, 240]:
        # Create a pattern with a sharp initial peak followed by rapid decay
        pattern = []
        for i in range(n):
            if i < 10:
                pattern.append(1000.0)
            elif i < 30:
                pattern.append(1000.0 * 0.8**(i-10))
            elif i < 60:
                pattern.append(100.0 * 0.9**(i-30))
            else:
                pattern.append(10.0 * 0.95**(i-60))
        patterns.append(pattern)
    
    # 10. Power-law decay with strategic enhancement
    for n in [100, 150, 200, 250]:
        pattern = []
        for i in range(n):
            # Power law decay with added oscillation for better structure
            base = 1.0 / (i + 1)**1.2
            oscillation = 0.1 * np.sin(i * 0.1)
            val = max(0.01, 1000 * (base + oscillation))
            pattern.append(val)
        patterns.append(pattern)
    
    # 11. Optimized "spike and decay" patterns (inspired by best results)
    for n in [150, 200, 250]:
        pattern = [0.0] * n
        # Strong spike at beginning, then exponential decay
        pattern[0] = 1000.0
        for i in range(1, n):
            pattern[i] = 1000.0 * (0.93 ** i)
        patterns.append(pattern)
    
    # 12. Alternating high-low patterns (good for reducing convolution max)
    for n in [100, 150, 200, 250]:
        pattern = []
        for i in range(n):
            if i % 2 == 0:
                pattern.append(1000.0)
            else:
                pattern.append(100.0)
        patterns.append(pattern)
    
    # 13. Optimized multi-peak patterns with strategic spacing
    for n in [120, 180, 240]:
        pattern = []
        # Create a pattern with multiple peaks
        peak_positions = [n//5, n//2, 4*n//5]
        peak_heights = [1000.0, 800.0, 600.0]
        for i in range(n):
            min_dist = min(abs(i - pos) for pos in peak_positions)
            peak_idx = np.argmin([abs(i - pos) for pos in peak_positions])
            peak_height = peak_heights[peak_idx]
            # Gaussian-like decay from peak
            val = peak_height * np.exp(-min_dist/10.0)
            pattern.append(max(0.01, val))
        patterns.append(pattern)
    
    # 14. Heavy-tail patterns with aggressive early decay
    for n in [100, 150, 200, 250]:
        pattern = []
        # Heavy tail with early rapid decay
        for i in range(n):
            if i < 20:
                pattern.append(1000.0)
            elif i < 50:
                pattern.append(1000.0 * (0.8**(i-20)))
            else:
                # Very long tail - this is the key for better C1
                pattern.append(100.0 * (0.95**(i-50)))
        patterns.append(pattern)
    
    return patterns

def enhanced_local_search(initial_sequence, max_iter=100):
    """
    Enhanced local search with better strategy combinations and escape mechanisms.
    """
    current_sequence = initial_sequence.copy()
    current_score = evaluate_sequence(current_sequence)
    
    # Track recent improvements for adaptive behavior
    recent_improvements = []
    improvement_count = 0
    
    for iteration in range(max_iter):
        # Try different types of perturbations
        best_sequence = current_sequence.copy()
        best_score = current_score
        
        # Strategy 1: Fine-grained multiplicative changes with wider range
        for i in range(len(current_sequence)):
            original_value = current_sequence[i]
            # Test a wide range of factors for fine-tuning
            factors = [0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.15, 1.2]
            for factor in factors:
                if original_value * factor >= 0.01:
                    candidate = current_sequence.copy()
                    candidate[i] = original_value * factor
                    new_score = evaluate_sequence(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_sequence = candidate.copy()
        
        # Strategy 2: Add/subtract with different magnitudes
        for i in range(len(current_sequence)):
            original_value = current_sequence[i]
            # Test different delta values
            deltas = [-2.0, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 2.0]
            for delta in deltas:
                if original_value + delta >= 0.01:
                    candidate = current_sequence.copy()
                    candidate[i] = original_value + delta
                    new_score = evaluate_sequence(candidate)
                    if new_score > best_score:
                        best_score = new_score
                        best_sequence = candidate.copy()
        
        # Strategy 3: Global scaling (more aggressive)
        scale_factors = [0.85, 0.9, 0.93, 0.96, 0.98, 1.02, 1.04, 1.07, 1.1, 1.15]
        for sf in scale_factors:
            candidate = [x * sf for x in current_sequence]
            new_score = evaluate_sequence(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 4: Small averaging to reduce noise
        if len(current_sequence) > 10:
            candidate = current_sequence.copy()
            for i in range(1, len(candidate) - 1):
                avg_val = (current_sequence[i-1] + current_sequence[i] + current_sequence[i+1]) / 3.0
                candidate[i] = avg_val
            new_score = evaluate_sequence(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 5: Random global perturbations for escape
        if random.random() < 0.15 and len(current_sequence) > 5:
            candidate = current_sequence.copy()
            # Perturb a few random elements
            num_perturb = min(5, len(candidate) // 10)
            for _ in range(num_perturb):
                idx = random.randint(0, len(candidate) - 1)
                candidate[idx] = max(0.01, candidate[idx] * random.uniform(0.7, 1.3))
            new_score = evaluate_sequence(candidate)
            if new_score > best_score:
                best_score = new_score
                best_sequence = candidate.copy()
        
        # Strategy 6: Strategic neighborhood search
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
                new_score = evaluate_sequence(candidate)
                if new_score > best_score:
                    best_score = new_score
                    best_sequence = candidate.copy()
        
        # Strategy 7: Window averaging for better smoothing
        if len(current_sequence) > 20:
            window_size = min(5, len(current_sequence) // 10)
            if window_size > 1:
                candidate = current_sequence.copy()
                for i in range(window_size, len(candidate) - window_size):
                    window_sum = sum(current_sequence[i-window_size:i+window_size+1])
                    avg_val = window_sum / (2 * window_size + 1)
                    candidate[i] = avg_val
                new_score = evaluate_sequence(candidate)
                if new_score > best_score:
                    best_score = new_score
                    best_sequence = candidate.copy()
        
        # Update if improvement found
        if best_score > current_score:
            current_sequence = best_sequence
            current_score = best_score
            recent_improvements.append(iteration)
            improvement_count += 1
            # Keep only last 5 improvements
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)
        else:
            # Adaptive escape mechanism - increase probability with lack of progress
            escape_prob = 0.05 + 0.15 * min(len(recent_improvements), 3)
            if random.random() < escape_prob:
                # Random perturbation with larger magnitude
                idx = random.randint(0, len(current_sequence) - 1)
                current_sequence[idx] = max(0.01, current_sequence[idx] * random.uniform(0.6, 1.4))
                current_score = evaluate_sequence(current_sequence)
    
    return current_sequence

def optimize_with_local_search(initial_sequence: List[float], max_iter: int = 300) -> List[float]:
    """Use local search to refine an initial sequence."""
    def objective(x):
        # Minimize negative of our target function (to maximize 1/C₁)
        return -evaluate_sequence(list(x))
    
    # Use L-BFGS-B for local refinement
    bounds = [(1e-10, 1000.0) for _ in range(len(initial_sequence))]
    
    try:
        res = minimize(objective, initial_sequence, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': max_iter})
        
        if res.success:
            return list(res.x)
    except:
        pass
    
    return initial_sequence

def optimize_with_differential_evolution(sequence: List[float], max_iter: int = 1000) -> List[float]:
    """Use differential evolution to optimize a sequence."""
    def objective(x):
        # Minimize negative of our target function (to maximize 1/C₁)
        return -evaluate_sequence(list(x))
    
    # Bounds for optimization (values between 0 and 1000)
    bounds = [(1e-10, 1000.0) for _ in range(len(sequence))]
    
    # Use differential evolution for robust global search
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_iter,
            popsize=min(20, len(sequence)),
            seed=42,
            disp=False,
            strategy='best1bin'
        )
        
        if result.success:
            return list(result.x)
    except:
        pass
    
    return sequence

def advanced_targeted_optimization():
    """
    Advanced targeted optimization focusing on the highest-performing patterns.
    """
    best_inv_c1 = 0
    best_sequence = None
    best_result = None
    
    # Strategy 1: High-performance mathematical patterns
    patterns = generate_mathematical_patterns()
    for pattern in patterns:
        metrics = evaluate_sequence(pattern)
        inv_c1 = metrics  # Already returns 1/C1
        if inv_c1 > best_inv_c1 and sum(pattern) > 0.01:
            best_inv_c1 = inv_c1
            best_sequence = pattern
            best_result = metrics
    
    # Strategy 2: Specialized sequences
    specialized_sequences = []
    # Add more mathematical patterns from inspiration 2
    for _ in range(30):
        n = random.randint(80, 300)
        
        # Option 1: Geometric with base near 0.92 (known to work well)
        if random.random() < 0.4:
            base = 0.92 + random.uniform(-0.02, 0.02)
            sequence = [base**i for i in range(n)]
            specialized_sequences.append(sequence)
        
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
            specialized_sequences.append(sequence[:n])
        
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
            specialized_sequences.append(sequence)
    
    for sequence in specialized_sequences:
        metrics = evaluate_sequence(sequence)
        inv_c1 = metrics  # Already returns 1/C1
        if inv_c1 > best_inv_c1 and sum(sequence) > 0.01:
            best_inv_c1 = inv_c1
            best_sequence = sequence
            best_result = metrics
    
    # Strategy 3: Optimization around best patterns with better methods
    if best_sequence is not None:
        # Try multiple optimization methods with different settings
        methods_and_settings = [
            ('Nelder-Mead', {'maxiter': 50, 'disp': False}),
            ('Powell', {'maxiter': 50, 'disp': False}),
            ('L-BFGS-B', {'maxiter': 50}),
        ]
        
        for method, options in methods_and_settings:
            try:
                result = minimize(
                    lambda x: -evaluate_sequence(x),
                    best_sequence,
                    method=method,
                    options=options,
                    bounds=[(0.01, 1000.0)] * len(best_sequence)
                )
                if result.success:
                    test_sequence = result.x
                    metrics = evaluate_sequence(test_sequence)
                    inv_c1 = metrics  # Already returns 1/C1
                    if inv_c1 > best_inv_c1 and sum(test_sequence) > 0.01:
                        best_inv_c1 = inv_c1
                        best_sequence = test_sequence.tolist()
                        best_result = metrics
            except Exception as e:
                continue
    
    # Strategy 4: Enhanced local search refinement
    if best_sequence is not None:
        refined_sequence = enhanced_local_search(best_sequence, max_iter=100)
        metrics = evaluate_sequence(refined_sequence)
        inv_c1 = metrics  # Already returns 1/C1
        if inv_c1 > best_inv_c1 and sum(refined_sequence) > 0.01:
            best_inv_c1 = inv_c1
            best_sequence = refined_sequence
            best_result = metrics
    
    # Strategy 5: Additional targeted optimizations
    if best_sequence is not None:
        # Try a few more specific optimization attempts
        for _ in range(5):
            try:
                # Slightly perturb and optimize
                perturbed = [x * random.uniform(0.95, 1.05) for x in best_sequence]
                result = minimize(
                    lambda x: -evaluate_sequence(x),
                    perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 40},
                    bounds=[(0.01, 1000.0)] * len(perturbed)
                )
                if result.success:
                    test_sequence = result.x
                    metrics = evaluate_sequence(test_sequence)
                    inv_c1 = metrics  # Already returns 1/C1
                    if inv_c1 > best_inv_c1 and sum(test_sequence) > 0.01:
                        best_inv_c1 = inv_c1
                        best_sequence = test_sequence.tolist()
                        best_result = metrics
            except:
                continue
    
    # Strategy 6: Try a few different random starts for global exploration
    for _ in range(10):
        try:
            # Generate a random sequence with better structure
            n = random.randint(120, 250)
            base = 0.92 + random.uniform(-0.02, 0.02)
            sequence = [base**i for i in range(n)]
            # Add some random variation to prevent getting stuck
            for i in range(len(sequence)):
                if random.random() < 0.15:
                    sequence[i] = max(0.01, sequence[i] * random.uniform(0.9, 1.1))
            
            metrics = evaluate_sequence(sequence)
            inv_c1 = metrics  # Already returns 1/C1
            if inv_c1 > best_inv_c1 and sum(sequence) > 0.01:
                best_inv_c1 = inv_c1
                best_sequence = sequence
                best_result = metrics
        except:
            continue
    
    # Strategy 7: Try a few more specific mathematical patterns
    # These are designed to have excellent autocorrelation properties
    for _ in range(8):
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
            inv_c1 = metrics  # Already returns 1/C1
            if inv_c1 > best_inv_c1 and sum(sequence) > 0.01:
                best_inv_c1 = inv_c1
                best_sequence = sequence
                best_result = metrics
        except:
            continue
    
    # Strategy 8: Try a hybrid approach with known good structures
    # Combine multiple proven approaches
    try:
        # Try a carefully constructed hybrid pattern
        n = random.randint(150, 250)
        hybrid_pattern = []
        # Create a pattern that starts high, drops rapidly, then decays slowly
        for i in range(n):
            if i < 20:
                hybrid_pattern.append(1000.0 * (0.95**i))
            elif i < 50:
                hybrid_pattern.append(1000.0 * (0.85**(i-20)))
            else:
                hybrid_pattern.append(100.0 * (0.95**(i-50)))
        # Normalize to reasonable scale
        max_val = max(hybrid_pattern) if hybrid_pattern else 1.0
        if max_val > 0:
            hybrid_pattern = [x / max_val * 1000.0 for x in hybrid_pattern]
        
        metrics = evaluate_sequence(hybrid_pattern)
        inv_c1 = metrics  # Already returns 1/C1
        if inv_c1 > best_inv_c1 and sum(hybrid_pattern) > 0.01:
            best_inv_c1 = inv_c1
            best_sequence = hybrid_pattern
            best_result = metrics
    except:
        pass
    
    # Strategy 9: Focus on specific high-performing mathematical bases
    # Try different geometric bases that have shown promise
    for base in [0.91, 0.915, 0.92, 0.925, 0.93]:
        try:
            n = random.randint(180, 250)
            sequence = [base**i for i in range(n)]
            metrics = evaluate_sequence(sequence)
            inv_c1 = metrics  # Already returns 1/C1
            if inv_c1 > best_inv_c1 and sum(sequence) > 0.01:
                best_inv_c1 = inv_c1
                best_sequence = sequence
                best_result = metrics
        except:
            continue
    
    # Strategy 10: Try a very specific pattern that has shown to beat benchmarks
    # Based on the best results from inspirations
    try:
        # Create a pattern that combines a strong initial spike with heavy tail
        n = 200
        pattern = []
        # Strong initial spike
        for i in range(15):
            pattern.append(1000.0)
        # Then decay with longer tail
        for i in range(15, n):
            pattern.append(100.0 * (0.95**(i-15)))
        
        metrics = evaluate_sequence(pattern)
        inv_c1 = metrics  # Already returns 1/C1
        if inv_c1 > best_inv_c1 and sum(pattern) > 0.01:
            best_inv_c1 = inv_c1
            best_sequence = pattern
            best_result = metrics
    except:
        pass
    
    return best_sequence if best_sequence is not None else [1.0] * 100

def search_for_best_sequence() -> List[float]:
    """
    Use a sophisticated hybrid approach combining mathematical constructions and optimization
    """
    # Run advanced targeted optimization
    sequence = advanced_targeted_optimization()
    
    # Ensure sequence has reasonable properties
    if len(sequence) == 0:
        sequence = [1.0]
    
    # Make sure sum is meaningful
    if sum(sequence) < 0.01:
        sequence = [x + 0.1 for x in sequence]
    
    # Final refinement with enhanced local search
    try:
        refined_sequence = enhanced_local_search(sequence, max_iter=60)
        refined_metrics = evaluate_sequence(refined_sequence)
        current_metrics = evaluate_sequence(sequence)
        if refined_metrics > current_metrics:
            sequence = refined_sequence
    except:
        pass
    
    # Try one final optimization with different approach
    try:
        # Use a different optimization method for final tuning
        result = minimize(
            lambda x: -evaluate_sequence(x),
            sequence,
            method='L-BFGS-B',
            bounds=[(0.01, 1000.0)] * len(sequence),
            options={'maxiter': 30}
        )
        if result.success:
            final_sequence = result.x
            final_metrics = evaluate_sequence(final_sequence)
            current_metrics = evaluate_sequence(sequence)
            if final_metrics > current_metrics:
                sequence = final_sequence.tolist()
    except:
        pass
    
    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
