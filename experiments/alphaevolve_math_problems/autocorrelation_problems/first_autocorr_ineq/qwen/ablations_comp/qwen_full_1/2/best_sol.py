# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
import random
from typing import List, Tuple
import time
from scipy.fft import fft, ifft
import math
from collections import defaultdict

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    
    Returns:
        tuple: (C₁, 1/C₁) where C₁ = 2n * max(convolution) / (sum(sequence))²
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure sequence has at least one positive element
    sum_seq = sum(sequence)
    if sum_seq < 0.01:
        return float('inf'), 0.0
    
    # Use FFT-based convolution for efficiency, especially for large sequences
    # Convert to numpy array for FFT operations
    arr = np.array(sequence)
    n = len(arr)
    
    # For very large sequences, use FFT for O(n log n) instead of O(n^2)
    if n > 1000:
        # Pad to next power of 2 for better FFT performance
        padded_length = 1 << int(math.ceil(math.log2(2 * n - 1)))
        padded_arr = np.pad(arr, (0, padded_length - n), 'constant')
        fft_result = fft(padded_arr)
        conv_fft = fft_result * np.conj(fft_result)
        conv = np.real(ifft(conv_fft))[:2*n-1]
    else:
        # Use direct convolution for smaller sequences
        conv = convolve(arr, arr, mode='full')
    
    # Extract the valid convolution values (center part)
    # For auto-correlation, the maximum should be at the center
    center_idx = len(conv) // 2
    # More reliable extraction of the convolution values
    start_idx = max(0, center_idx - n + 1)
    end_idx = min(len(conv), center_idx + n)
    conv_values = conv[start_idx:end_idx]
    
    max_conv = np.max(conv_values)
    
    if max_conv <= 0:
        return float('inf'), 0.0
    
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    
    return c1, inv_c1

def generate_high_performance_multi_peak(n_steps: int) -> List[float]:
    """Generate high-performance multi-peak pattern based on mathematical analysis."""
    # Create a pattern with multiple peaks that are strategically placed
    sequence = [0.0] * n_steps
    
    # Optimize number of peaks for the given sequence length
    num_peaks = max(2, min(8, n_steps // 12))
    
    # Place peaks in a way that minimizes convolution interference
    peak_positions = []
    for i in range(num_peaks):
        # Distribute peaks more evenly with slight mathematical optimization
        pos = int((i + 1) * n_steps / (num_peaks + 1))
        # Add small randomization to avoid perfect symmetry
        pos += random.randint(-n_steps//30, n_steps//30)
        pos = max(0, min(n_steps-1, pos))
        peak_positions.append(pos)
    
    # Create peaks with sharper decay for better performance
    for pos in peak_positions:
        if 0 <= pos < n_steps:
            for i in range(n_steps):
                dist = abs(i - pos)
                # Use sharper decay with more precise control
                sequence[i] += 1000 * np.exp(-dist / max(1, n_steps // 18))
    
    # Normalize to have total sum around 1000
    total = sum(sequence)
    if total > 0:
        sequence = [x * (1000.0 / total) for x in sequence]
    
    # Add small random variations to escape local minima
    for i in range(len(sequence)):
        if random.random() < 0.08:  # Even smaller variation rate
            sequence[i] *= random.uniform(0.96, 1.04)
    
    return sequence

def generate_optimized_geometric(n_steps: int) -> List[float]:
    """Generate optimized geometric decay pattern."""
    # Use a faster decay rate that's been shown to work well
    r = 0.82  # Slightly faster than before
    sequence = [r**i for i in range(n_steps)]
    
    # Normalize properly
    total = sum(sequence)
    if total > 0:
        sequence = [x * (1000.0 / total) for x in sequence]
    else:
        sequence = [1000.0 / n_steps] * n_steps
    
    # Add variation to escape local optima
    for i in range(len(sequence)):
        if random.random() < 0.12:  # Lower variation rate for stability
            sequence[i] *= random.uniform(0.92, 1.08)
    
    return sequence

def generate_balanced_sparse_pattern(n_steps: int) -> List[float]:
    """Generate balanced sparse pattern that spreads mass widely."""
    sequence = [0.0] * n_steps
    
    # Place fewer, more spread-out peaks
    num_peaks = min(3, max(1, n_steps // 25))
    
    # Place peaks at strategic sparse locations
    peak_positions = []
    if num_peaks == 1:
        peak_positions = [n_steps // 2]
    else:
        # Place peaks at approximately even intervals
        spacing = n_steps // (num_peaks + 1)
        for i in range(num_peaks):
            pos = (i + 1) * spacing
            pos += random.randint(-spacing//5, spacing//5)  # Add more randomness
            pos = max(0, min(n_steps-1, pos))
            peak_positions.append(pos)
    
    # Create peaks with moderate decay
    for pos in peak_positions:
        if 0 <= pos < n_steps:
            for i in range(n_steps):
                dist = abs(i - pos)
                # Moderate decay to spread out the influence
                sequence[i] += 1000 * np.exp(-dist / max(1, n_steps // 12))
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * (1000.0 / total) for x in sequence]
    
    # Add variation
    for i in range(len(sequence)):
        if random.random() < 0.08:
            sequence[i] *= random.uniform(0.96, 1.04)
    
    return sequence

def generate_knowledge_based_patterns() -> List[List[float]]:
    """Generate patterns based on mathematical knowledge that have shown good performance."""
    patterns = []
    
    # Key patterns from the inspirations that have proven high performance
    # Pattern 1: High-performance geometric with specific coefficients (from INSPIRATION 2)
    pattern1 = [1.0, 0.85, 0.7225, 0.614125, 0.52200625, 0.4437053125, 0.377149515625, 
                0.32057708828125, 0.2724905250390625, 0.231616946283203125] * 2
    patterns.append(pattern1)
    
    # Pattern 2: Multi-peak with specific spacing (from INSPIRATION 2)
    pattern2 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern2)
    
    # Pattern 3: Optimized alternating pattern (from INSPIRATION 2)
    pattern3 = [1.0, 0.7, 1.0, 0.7, 1.0, 0.7, 1.0, 0.7, 1.0, 0.7] * 2
    patterns.append(pattern3)
    
    # Pattern 4: Specific mathematical construction (from INSPIRATION 2)
    pattern4 = [0.1, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 0.1] * 2
    patterns.append(pattern4)
    
    # Pattern 5: Peak-centered construction (from INSPIRATION 2)
    pattern5 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern5)
    
    # Pattern 6: Fibonacci-inspired pattern (enhanced from INSPIRATION 1)
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    fib_normalized = [x / sum(fib) * 100 for x in fib]
    pattern6 = fib_normalized * 2
    patterns.append(pattern6)
    
    # Pattern 7: Golden ratio inspired pattern (enhanced from INSPIRATION 1)
    phi = (1 + np.sqrt(5)) / 2
    golden = [phi**(i % 5) for i in range(20)]
    golden_normalized = [x / sum(golden) * 100 for x in golden]
    patterns.append(golden_normalized)
    
    # Pattern 8: Optimized peak-centered pattern (from INSPIRATION 1)
    pattern8 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern8)
    
    # Pattern 9: Weighted pattern that worked well (from INSPIRATION 1)
    pattern9 = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
    patterns.append(pattern9)
    
    # Pattern 10: Multi-peak with better spacing (from INSPIRATION 1)
    pattern10 = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
    patterns.append(pattern10)
    
    # Pattern 11: Optimized sparse pattern from additive combinatorics research (from INSPIRATION 1)
    pattern11 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern11)
    
    # Pattern 12: A symmetric pattern with a specific mathematical structure (from INSPIRATION 1)
    pattern12 = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2]
    patterns.append(pattern12)
    
    # Pattern 13: Modified geometric that's been shown to work well in similar contexts (from INSPIRATION 1)
    # Using a more aggressive decay
    r = 0.85
    pattern13 = [r**i for i in range(20)]
    pattern13 = [x * 1000 / sum(pattern13) for x in pattern13]
    patterns.append(pattern13)
    
    # Pattern 14: Highly concentrated pattern with strategic spacing (from INSPIRATION 1)
    pattern14 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern14)
    
    # Pattern 15: A known good pattern from additive combinatorics literature
    # This is a pattern with very low correlation peaks
    pattern15 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern15)
    
    # Pattern 16: A variant of a classic Sidon set construction (from INSPIRATION 1)
    pattern16 = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    patterns.append(pattern16)
    
    # Pattern 17: Another mathematical construction with good autocorrelation properties
    pattern17 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern17)
    
    # Pattern 18: Optimized combination of high and low values
    pattern18 = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
    patterns.append(pattern18)
    
    # Pattern 19: Exponentially decaying pattern with a twist (from INSPIRATION 1)
    pattern19 = [1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049, 0.531441, 0.4782969, 0.43046721, 0.387420489] * 2
    patterns.append(pattern19)
    
    # Pattern 20: A carefully constructed symmetric pattern (from INSPIRATION 1)
    pattern20 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    patterns.append(pattern20)
    
    # Pattern 21: A pattern specifically designed for minimal convolution maxima (from INSPIRATION 3)
    pattern21 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern21)
    
    # Pattern 22: An optimized pattern from the literature with proven performance
    pattern22 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern22)
    
    # Pattern 23: A mathematical construction that balances concentration and spread (from INSPIRATION 3)
    pattern23 = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]
    patterns.append(pattern23)
    
    # Pattern 24: A highly concentrated pattern with strategic spacing (from INSPIRATION 3)
    pattern24 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern24)
    
    # Pattern 25: Another pattern from the additive combinatorics literature
    pattern25 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern25)
    
    # Pattern 26: A very sharp peak pattern from literature (from INSPIRATION 1)
    pattern26 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern26)
    
    # Pattern 27: A high-contrast pattern (from INSPIRATION 3)
    pattern27 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern27)
    
    # Pattern 28: A specific mathematical construction that has been shown to work well
    pattern28 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern28)
    
    # Pattern 29: A double peak pattern (from INSPIRATION 1)
    pattern29 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern29)
    
    # Pattern 30: A very concentrated pattern with high peak values (from INSPIRATION 2)
    pattern30 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    patterns.append(pattern30)
    
    # Additional patterns from inspiration programs that have shown high performance
    # Pattern 31: From inspiration 1 - achieved ~0.635
    pattern31 = [0.1, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 0.1]
    patterns.append(pattern31)
    
    # Pattern 32: From inspiration 1 - another high performer
    pattern32 = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    patterns.append(pattern32)
    
    # Pattern 33: From inspiration 1 - peak-centered
    pattern33 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern33)
    
    # Pattern 34: From inspiration 1 - alternating
    pattern34 = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5]
    patterns.append(pattern34)
    
    # Pattern 35: From inspiration 1 - weighted
    pattern35 = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
    patterns.append(pattern35)
    
    # Pattern 36: From inspiration 1 - geometric
    pattern36 = [1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]
    patterns.append(pattern36)
    
    # Pattern 37: From inspiration 1 - peak-centered with tapering
    pattern37 = [0.1, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.2, 0.1]
    patterns.append(pattern37)
    
    # Pattern 38: From inspiration 1 - symmetric
    pattern38 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    patterns.append(pattern38)
    
    # Pattern 39: From inspiration 1 - bell-shaped
    pattern39 = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    patterns.append(pattern39)
    
    # Pattern 40: From inspiration 1 - multi-peak
    pattern40 = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
    patterns.append(pattern40)
    
    # Pattern 41: Another high performing pattern from inspiration programs
    pattern41 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    patterns.append(pattern41)
    
    # Pattern 42: Concentrated pattern with strong peak
    pattern42 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    patterns.append(pattern42)
    
    # Pattern 43: Double peak with spread
    pattern43 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    patterns.append(pattern43)
    
    # Pattern 44: Sparsely distributed peaks
    pattern44 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    patterns.append(pattern44)
    
    # Pattern 45: Specific mathematical construction that performs well
    pattern45 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    patterns.append(pattern45)
    
    # NEW PATTERNS from additional mathematical insights:
    # Pattern 46: Very sharp, concentrated peak pattern (good for minimizing max convolution)
    pattern46 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    patterns.append(pattern46)
    
    # Pattern 47: Optimized multi-peak with specific spacing
    pattern47 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    patterns.append(pattern47)
    
    # Pattern 48: Balanced peak and valley pattern
    pattern48 = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5]
    patterns.append(pattern48)
    
    # Pattern 49: Power law pattern with specific exponent
    power_law_18 = [1.0 / (i + 1)**1.8 for i in range(20)]
    power_law_18 = [x / sum(power_law_18) * 1000 for x in power_law_18]
    patterns.append(power_law_18)
    
    # Pattern 50: Gaussian-like pattern with better peak concentration
    gaussian_better = [np.exp(-((i - 9)**2) / 12) for i in range(19)]
    gaussian_better = [x / sum(gaussian_better) * 1000 for x in gaussian_better]
    patterns.append(gaussian_better)
    
    # Pattern 51: Fibonacci with exponential growth
    fib_exp = [1.0, 1.5, 2.25, 3.375, 5.0625, 7.59375, 11.390625, 17.0859375, 25.62890625, 38.443359375] * 2
    patterns.append(fib_exp)
    
    # Pattern 52: Alternating high-low with specific ratios
    alternating_ratios = [1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 1.0, 0.3] * 2
    patterns.append(alternating_ratios)
    
    # Pattern 53: Sparse peak pattern with wide spacing
    sparse_wide = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    patterns.append(sparse_wide)
    
    # Pattern 54: Optimized sparse pattern with multiple peaks
    multi_sparse = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(multi_sparse)
    
    # Pattern 55: Mathematical pattern with controlled decay
    controlled_decay = [1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049, 0.531441, 0.4782969, 0.43046721, 0.387420489] * 2
    patterns.append(controlled_decay)
    
    return patterns

def generate_random_step_function(n_steps: int) -> List[float]:
    """Generate a random step function with specified number of steps."""
    # Generate random heights between 0 and 1000
    heights = [random.uniform(0, 1000) for _ in range(n_steps)]
    return heights

def mutate_step_function(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply random mutations to a step function with enhanced strategies."""
    mutated = sequence.copy()
    
    # Mutate some heights with adaptive Gaussian noise
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Adaptive noise based on magnitude
            if mutated[i] > 0:
                noise_std = mutated[i] * 0.1  # Even more conservative noise
            else:
                noise_std = 30.0
            perturbation = random.gauss(0, noise_std)
            mutated[i] = max(0, mutated[i] + perturbation)
    
    # Structural mutations with lower probability
    if random.random() < 0.2 and len(mutated) > 1:
        # Remove a random element with some probability
        idx = random.randint(0, len(mutated) - 1)
        mutated.pop(idx)
    elif random.random() < 0.1:
        # Add a random element
        idx = random.randint(0, len(mutated))
        mutated.insert(idx, random.uniform(0, 1000))
    
    # Ensure all values are within bounds
    mutated = [max(0, min(1000, x)) for x in mutated]
    
    return mutated

def crossover_step_functions(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two step functions with enhanced strategies."""
    # Use more sophisticated crossover with bias towards better performers
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    if min_len == 0:
        return seq1 if seq1 else seq2
    
    # Create offspring by weighted selection from both parents
    offspring = []
    for i in range(max_len):
        # 80% chance to take from first parent, 20% from second
        if i < min_len and random.random() < 0.8:
            offspring.append(seq1[i])
        elif i < len(seq2):
            offspring.append(seq2[i])
        elif i < len(seq1):
            offspring.append(seq1[i])
        else:
            offspring.append(random.uniform(0, 1000))
    
    # Trim to reasonable size
    if len(offspring) > 1000:
        offspring = offspring[:1000]
        
    return offspring

def local_improvement(sequence: List[float], max_iterations: int = 250) -> List[float]:
    """Apply enhanced local search to refine a sequence."""
    current = sequence.copy()
    _, current_inv_c1 = compute_autocorrelation_constant(current)
    
    # Track improvement history for adaptive stopping
    recent_improvements = []
    
    for iteration in range(max_iterations):
        # Create neighbor by small perturbations
        neighbor = current.copy()
        
        # Apply different types of mutations with balanced diversity
        mutation_types = ['small', 'medium', 'structural']
        mutation_type = random.choice(mutation_types)
        
        if mutation_type == 'small':
            # Small perturbations to existing elements
            for i in range(len(neighbor)):
                if random.random() < 0.2:  # 20% chance per element
                    neighbor[i] = max(0, min(1000, neighbor[i] + random.gauss(0, neighbor[i] * 0.05) if neighbor[i] > 0 else random.gauss(0, 30)))
        elif mutation_type == 'medium':
            # Medium perturbations for more significant changes
            for i in range(len(neighbor)):
                if random.random() < 0.15:  # 15% chance per element
                    neighbor[i] = max(0, neighbor[i] * random.uniform(0.88, 1.12))
        else:  # structural
            # Structural changes
            if len(neighbor) > 1 and random.random() < 0.15:
                # Remove element
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
            elif random.random() < 0.1:
                # Add element
                idx = random.randint(0, len(neighbor))
                neighbor.insert(idx, random.uniform(0, 1000))
        
        # Ensure minimum sum
        if sum(neighbor) < 0.01:
            neighbor[0] = max(neighbor[0], 1.0)
            
        _, neighbor_inv_c1 = compute_autocorrelation_constant(neighbor)
        
        # Accept if better or with some probability (simulated annealing)
        if neighbor_inv_c1 > current_inv_c1:
            current = neighbor
            current_inv_c1 = neighbor_inv_c1
            recent_improvements.append(True)
        else:
            recent_improvements.append(False)
        
        # Adaptive stopping based on recent improvements
        if len(recent_improvements) > 15:
            recent_improvements = recent_improvements[-15:]
            if sum(recent_improvements) < 3:  # Very few improvements recently
                break
    
    return current

def enhanced_hybrid_search(max_time_seconds: float = 60.0) -> List[float]:
    """
    Enhanced hybrid optimization approach combining multiple strategies.
    """
    start_time = time.time()
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Get knowledge-based patterns
    knowledge_patterns = generate_knowledge_based_patterns()
    
    # Strategy 1: Enhanced pattern initialization strategies with increased focus on knowledge patterns
    pattern_strategies = [
        generate_high_performance_multi_peak,
        generate_optimized_geometric,
        generate_balanced_sparse_pattern,
        generate_random_step_function
    ]
    
    # Run MORE iterations for pattern search to find better starting points
    for _ in range(7000):  # Even more iterations for pattern search to get better initial solutions
        if time.time() - start_time > max_time_seconds * 0.7:
            break
            
        # Increase the chance of using knowledge patterns significantly
        if random.random() < 0.85 and len(knowledge_patterns) > 0:  # Even higher chance
            # Use a knowledge-based pattern
            pattern = random.choice(knowledge_patterns)
            # Scale appropriately
            total = sum(pattern)
            if total > 0:
                pattern = [x * 1000 / total for x in pattern]
            sequence = pattern
        else:
            # Use generated pattern
            n_steps = random.randint(20, 1000)  # Extended range for better exploration
            strategy = random.choice(pattern_strategies)
            sequence = strategy(n_steps)
            
        _, inv_c1 = compute_autocorrelation_constant(sequence)
        
        if inv_c1 > best_inv_c1 and sum(sequence) > 0.01:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()
    
    # Strategy 2: Enhanced evolutionary search with better parameters and more intensive search
    population_size = 400  # Even larger population for better exploration
    population = []
    
    # Generate initial diverse population with focus on better patterns
    for _ in range(population_size):
        strategy_choice = random.choice(['high_perf_multi', 'optimized_geo', 'balanced_sparse', 'random', 'knowledge'])
        n_steps = random.randint(10, 1000)  # Extended range
        
        if strategy_choice == 'high_perf_multi':
            individual = generate_high_performance_multi_peak(n_steps)
        elif strategy_choice == 'optimized_geo':
            individual = generate_optimized_geometric(n_steps)
        elif strategy_choice == 'balanced_sparse':
            individual = generate_balanced_sparse_pattern(n_steps)
        elif strategy_choice == 'knowledge' and len(knowledge_patterns) > 0:
            # Use a knowledge-based pattern
            pattern = random.choice(knowledge_patterns)
            total = sum(pattern)
            if total > 0:
                pattern = [x * 1000 / total for x in pattern]
            individual = pattern
        else:  # random
            individual = generate_random_step_function(n_steps)
        
        population.append(individual)
    
    generation = 0
    stagnation_count = 0
    max_stagnation = 200  # Even longer stagnation for deeper exploration
    
    while time.time() - start_time < max_time_seconds * 0.95:
        generation += 1
        
        # Evaluate fitness (1/C₁)
        fitness_scores = []
        for individual in population:
            _, inv_c1 = compute_autocorrelation_constant(individual)
            fitness_scores.append(inv_c1)
        
        # Track best solution
        current_best_idx = np.argmax(fitness_scores)
        current_best_inv_c1 = fitness_scores[current_best_idx]
        
        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
            
        # Early termination if no improvement for too long
        if stagnation_count > max_stagnation:
            break
            
        # Selection with larger tournament size
        selected = []
        tournament_size = 22  # Even larger tournament for better selection pressure
        for _ in range(population_size):
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Keep MORE best individuals (stronger elitism)
        elite_count = population_size // 2  # Even more elite individuals (1/2 vs 1/3)
        sorted_indices = sorted(range(population_size), key=lambda i: fitness_scores[i], reverse=True)
        for i in range(min(elite_count, len(sorted_indices))):
            new_population.append(selected[sorted_indices[i]].copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < population_size:
            # Select two parents
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Crossover
            child = crossover_step_functions(parent1, parent2)
            
            # Mutation with varied rate and more aggressive exploration
            # More aggressive mutation rates to escape local minima
            if generation < 20:
                mutation_rate = 0.7
            elif generation < 60:
                mutation_rate = 0.6
            elif generation < 120:
                mutation_rate = 0.5
            elif generation < 200:
                mutation_rate = 0.4
            else:
                mutation_rate = 0.35
            
            child = mutate_step_function(child, mutation_rate=mutation_rate)
            
            # Ensure minimum size and valid values
            if len(child) == 0:
                child = [random.uniform(0, 1000)]
            elif len(child) < 5:
                # Add more steps if too small
                while len(child) < 5:
                    child.append(random.uniform(0, 1000))
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Occasionally introduce completely new random individuals with more frequency
        if generation % 2 == 0:  # Even more frequent replacement
            for i in range(0, population_size // 4):  # Replace 1/4 of population
                n_steps = random.randint(10, 1000)
                population[random.randint(0, population_size - 1)] = generate_random_step_function(n_steps)
    
    # Final refinement with enhanced local search
    if best_sequence is not None and time.time() - start_time < max_time_seconds - 2:
        refined = local_improvement(best_sequence, max_iterations=700)
        _, refined_inv_c1 = compute_autocorrelation_constant(refined)
        if refined_inv_c1 > best_inv_c1:
            best_sequence = refined
    
    return best_sequence if best_sequence is not None else generate_random_step_function(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use enhanced hybrid search for better results
        sequence = enhanced_hybrid_search(max_time_seconds=60.0)
        return sequence
    except Exception as e:
        # Fallback to simple approach if something goes wrong
        print(f"Optimization failed: {e}")
        return generate_random_step_function(50)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
