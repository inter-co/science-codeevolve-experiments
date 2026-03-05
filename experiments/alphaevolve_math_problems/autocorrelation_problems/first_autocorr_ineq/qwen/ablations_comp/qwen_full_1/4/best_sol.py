# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution
import time
from typing import List, Tuple
import math
import itertools

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    
    Returns:
        tuple: (C₁, 1/C₁) where C₁ = 2n * max(convolution) / (sum(sequence))²
    """
    if not sequence or sum(sequence) < 0.01:
        return float('inf'), 0.0
    
    # Ensure all values are non-negative and clip to reasonable bounds
    sequence = np.clip(sequence, 0, 1000)
    
    n = len(sequence)
    # Use FFT for efficient convolution
    conv = fftconvolve(sequence, sequence, mode='full')
    # Take only the relevant part (the actual convolution)
    conv = conv[n-1:2*n-1]
    
    max_conv = np.max(conv)
    sum_seq = sum(sequence)
    
    if sum_seq < 0.01:
        return float('inf'), 0.0
    
    c1 = (2 * n * max_conv) / (sum_seq ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    
    return c1, inv_c1

def evaluate_sequence_gradient_free(individual):
    """Evaluate a sequence and return 1/C₁ value - gradient-free version"""
    # Convert individual to numpy array and ensure it's valid
    a = np.array(individual)
    
    # Ensure minimum sum constraint
    if np.sum(a) < 0.01:
        return 0.0  # Invalid sequence
    
    # Compute convolution using FFT for efficiency
    b = fftconvolve(a, a, mode='full')[:len(a)*2-1]
    
    # Compute C₁
    max_b = np.max(b)
    sum_a_squared = np.sum(a)**2
    
    if sum_a_squared == 0:
        return 0.0
    
    c1 = 2 * len(a) * max_b / sum_a_squared
    
    # Return inverse of C₁ (we want to maximize this)
    return 1.0 / c1 if c1 > 0 else 0.0

def create_mathematical_patterns_from_inspiration() -> List[List[float]]:
    """Create mathematical patterns based on the most successful approaches from inspirations"""
    patterns = []
    
    # From INSPIRATION 1 - High performing mathematical constructions
    # Fibonacci-like with golden ratio proportions
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    fib_norm = [x / sum(fib) * 1000 for x in fib]
    patterns.append(fib_norm)
    
    # Golden ratio pattern
    phi = (1 + np.sqrt(5)) / 2
    golden = [phi**(i % 5) for i in range(20)]
    golden_norm = [x / sum(golden) * 1000 for x in golden]
    patterns.append(golden_norm)
    
    # High-performance geometric with specific coefficients (from INSPIRATION 2)
    pattern1 = [1.0, 0.85, 0.7225, 0.614125, 0.52200625, 0.4437053125, 0.377149515625, 
                0.32057708828125, 0.2724905250390625, 0.231616946283203125] * 2
    patterns.append(pattern1)
    
    # Multi-peak with specific spacing (from INSPIRATION 3)
    pattern2 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern2)
    
    # Optimized alternating pattern (from INSPIRATION 2)
    pattern3 = [1.0, 0.7, 1.0, 0.7, 1.0, 0.7, 1.0, 0.7, 1.0, 0.7] * 2
    patterns.append(pattern3)
    
    # Specific mathematical construction (from INSPIRATION 1)
    pattern4 = [0.1, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 0.1] * 2
    patterns.append(pattern4)
    
    # Peak-centered construction (from INSPIRATION 1)
    pattern5 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern5)
    
    # Optimized peak-centered pattern (from INSPIRATION 1)
    pattern6 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern6)
    
    # Weighted pattern that worked well (from INSPIRATION 1)
    pattern7 = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
    patterns.append(pattern7)
    
    # Multi-peak with better spacing (from INSPIRATION 1)
    pattern8 = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
    patterns.append(pattern8)
    
    # Optimized sparse pattern from additive combinatorics research (from INSPIRATION 1)
    pattern9 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern9)
    
    # A symmetric pattern with a specific mathematical structure (from INSPIRATION 1)
    pattern10 = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2]
    patterns.append(pattern10)
    
    # Modified geometric that's been shown to work well (from INSPIRATION 1)
    r = 0.85
    pattern11 = [r**i for i in range(20)]
    pattern11 = [x * 1000 / sum(pattern11) for x in pattern11]
    patterns.append(pattern11)
    
    # Highly concentrated pattern with strategic spacing (from INSPIRATION 1)
    pattern12 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern12)
    
    # New optimized pattern from research - very sharp peaks
    pattern13 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern13)
    
    # Alternative mathematical pattern - alternating with emphasis on peaks
    pattern14 = [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2]
    patterns.append(pattern14)
    
    # Concentrated central peak with surrounding low values
    pattern15 = [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    patterns.append(pattern15)
    
    # Enhanced Fibonacci pattern with better scaling
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    fib_normalized = [x / sum(fib) * 1000 for x in fib]
    pattern16 = fib_normalized * 2
    patterns.append(pattern16)
    
    # Optimized symmetric pattern with peak in middle
    pattern17 = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    patterns.append(pattern17)
    
    # High contrast pattern for maximum separation
    pattern18 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern18)
    
    # Very sharp peak pattern (from INSPIRATION 1)
    pattern19 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern19)
    
    # Double peak pattern (from INSPIRATION 1)
    pattern20 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(pattern20)
    
    # Optimized exponential decay pattern (from INSPIRATION 2)
    pattern21 = [1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728] * 2
    patterns.append(pattern21)
    
    # Linear pattern (from INSPIRATION 3)
    pattern22 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] * 2
    patterns.append(pattern22)
    
    # Spike pattern (from INSPIRATION 1)
    pattern23 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(pattern23)
    
    return patterns

def create_advanced_patterns() -> List[List[float]]:
    """Create advanced patterns with more mathematical sophistication"""
    patterns = []
    
    # Multi-peak optimized for low convolution peaks
    multi_peak = [0.0] * 30
    peak_positions = [3, 8, 15, 22, 27]
    for pos in peak_positions:
        for i in range(30):
            dist = abs(i - pos)
            multi_peak[i] += 1000 * np.exp(-dist / 2.5)
    multi_peak_norm = [x / sum(multi_peak) * 1000 for x in multi_peak]
    patterns.append(multi_peak_norm)
    
    # Balanced sparse pattern with strategic distribution
    sparse = [0.0] * 25
    positions = [4, 9, 14, 19, 24]
    for pos in positions:
        for i in range(25):
            dist = abs(i - pos)
            sparse[i] += 1000 * np.exp(-dist / 3.0)
    sparse_norm = [x / sum(sparse) * 1000 for x in sparse]
    patterns.append(sparse_norm)
    
    # Optimized geometric with higher decay rate
    geo_fast = [0.82**i for i in range(25)]
    geo_fast_norm = [x / sum(geo_fast) * 1000 for x in geo_fast]
    patterns.append(geo_fast_norm)
    
    # Oscillating pattern with decay
    oscillating = []
    for i in range(25):
        oscillation = 1.0 if i % 2 == 0 else 0.8
        decay = 0.92 ** i
        oscillating.append(oscillation * decay)
    oscillating_norm = [x / sum(oscillating) * 1000 for x in oscillating]
    patterns.append(oscillating_norm)
    
    # Hybrid symmetric and asymmetric pattern
    hybrid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1] * 2
    hybrid_norm = [x / sum(hybrid) * 1000 for x in hybrid]
    patterns.append(hybrid_norm)
    
    # Peak-centered pattern with sharper decay
    peak_centered = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(peak_centered)
    
    # High-contrast pattern
    high_contrast = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(high_contrast)
    
    # Concentrated pattern
    concentrated = [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    patterns.append(concentrated)
    
    # Multi-peak optimized pattern
    multi_peak_opt = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    patterns.append(multi_peak_opt)
    
    # Enhanced Fibonacci pattern
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    fib_normalized = [x / sum(fib) * 1000 for x in fib]
    pattern20 = fib_normalized * 2
    patterns.append(pattern20)
    
    return patterns

def create_analytical_sequence(length: int, pattern_type: str = "geometric") -> List[float]:
    """Create sequences with mathematical patterns that tend to perform well"""
    
    if pattern_type == "geometric":
        # Geometric progression with decay - optimized for performance
        base = 1000.0
        ratio = 0.85  # This ratio has shown good performance
        sequence = [base * (ratio ** i) for i in range(length)]
        # Normalize to ensure reasonable sum
        total = sum(sequence)
        if total > 0:
            sequence = [x * 1000 / total for x in sequence]
        return sequence
    
    elif pattern_type == "inverse_square":
        # Inverse square law pattern - tested for good performance
        sequence = [1.0 / ((i + 1) ** 2) for i in range(length)]
        # Normalize
        total = sum(sequence)
        if total > 0:
            sequence = [x * 1000 / total for x in sequence]
        return sequence
    
    elif pattern_type == "exponential_decay":
        # Exponential decay pattern - with slightly different rate
        sequence = [np.exp(-i * 0.15) for i in range(length)]  # Slightly faster decay
        # Normalize
        total = sum(sequence)
        if total > 0:
            sequence = [x * 1000 / total for x in sequence]
        return sequence
    
    elif pattern_type == "linear_decay":
        # Linear decay pattern
        sequence = [max(0, 1000 - i * 10) for i in range(length)]
        # Normalize
        total = sum(sequence)
        if total > 0:
            sequence = [x * 1000 / total for x in sequence]
        return sequence
    
    else:
        # Default: random with structure but more controlled
        sequence = []
        for i in range(length):
            if i < 10:
                sequence.append(random.uniform(500, 1000))
            elif i < 30:
                sequence.append(max(100, 1000 * np.exp(-i * 0.12)))  # Slightly faster decay
            else:
                sequence.append(max(0, 100 * np.exp(-i * 0.07)))  # Slower decay for tail
        return sequence

def create_multi_pattern_sequence(length: int) -> List[float]:
    """Create a sequence combining multiple mathematical patterns"""
    # Mix of different mathematical approaches with better weighting
    sequence = []
    
    # Create segments with different patterns
    segment_length = max(1, length // 4)
    
    # Segment 1: Exponential decay with optimized parameters
    exp_segment = [np.exp(-i * 0.15) for i in range(segment_length)]
    exp_total = sum(exp_segment)
    if exp_total > 0:
        exp_segment = [x * 250 / exp_total for x in exp_segment]
    
    # Segment 2: Inverse square with better normalization
    inv_segment = [1.0 / ((i + 1) ** 1.8) for i in range(segment_length)]
    inv_total = sum(inv_segment)
    if inv_total > 0:
        inv_segment = [x * 250 / inv_total for x in inv_segment]
    
    # Segment 3: Linear decay with controlled shape
    lin_segment = [max(0, 1000 - i * 12) for i in range(segment_length)]
    lin_total = sum(lin_segment)
    if lin_total > 0:
        lin_segment = [x * 250 / lin_total for x in lin_segment]
    
    # Segment 4: Constant (for balance)
    const_segment = [1000.0] * segment_length
    
    # Combine segments
    sequence.extend(exp_segment)
    sequence.extend(inv_segment)
    sequence.extend(lin_segment)
    sequence.extend(const_segment)
    
    # Truncate or pad to exact length
    if len(sequence) > length:
        sequence = sequence[:length]
    elif len(sequence) < length:
        # Fill with last element
        sequence.extend([sequence[-1]] * (length - len(sequence)))
    
    # Normalize to sum to 1000
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def create_optimized_patterns() -> List[List[float]]:
    """Create highly optimized patterns based on extensive testing"""
    patterns = []
    
    # Optimized geometric pattern
    geo = [0.85**i for i in range(20)]
    geo_norm = [x / sum(geo) * 1000 for x in geo]
    patterns.append(geo_norm)
    
    # Optimized multi-peak pattern with specific spacing
    multi_peak = [0.0] * 25
    positions = [3, 8, 12, 17, 22]
    for pos in positions:
        for i in range(25):
            dist = abs(i - pos)
            multi_peak[i] += 1000 * np.exp(-dist / 2.0)
    multi_peak_norm = [x / sum(multi_peak) * 1000 for x in multi_peak]
    patterns.append(multi_peak_norm)
    
    # Optimized sparse pattern
    sparse = [0.0] * 20
    for i in range(0, 20, 3):
        sparse[i] = 1000
    sparse_norm = [x / sum(sparse) * 1000 for x in sparse]
    patterns.append(sparse_norm)
    
    # Optimized alternating pattern
    alt = [1.0, 0.7, 1.0, 0.7, 1.0, 0.7, 1.0, 0.7, 1.0, 0.7] * 2
    alt_norm = [x / sum(alt) * 1000 for x in alt]
    patterns.append(alt_norm)
    
    # Optimized symmetric pattern
    symm = [0.1, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 0.1] * 2
    symm_norm = [x / sum(symm) * 1000 for x in symm]
    patterns.append(symm_norm)
    
    # Optimized peak-centered pattern
    peak_centered = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    patterns.append(peak_centered)
    
    return patterns

def optimize_with_direct_search() -> List[float]:
    """Use enhanced direct search with more comprehensive pattern generation"""
    best_sequence = None
    best_score = -float('inf')
    
    # Try mathematical patterns from inspirations
    knowledge_patterns = create_mathematical_patterns_from_inspiration()
    
    # Try advanced patterns
    advanced_patterns = create_advanced_patterns()
    
    # Try optimized patterns
    optimized_patterns = create_optimized_patterns()
    
    # Try knowledge patterns first (with more aggressive selection)
    for pattern in knowledge_patterns:
        score = evaluate_sequence_gradient_free(pattern)
        if score > best_score and score > 0.01:
            best_score = score
            best_sequence = pattern[:]
    
    # Try advanced patterns
    for pattern in advanced_patterns:
        score = evaluate_sequence_gradient_free(pattern)
        if score > best_score and score > 0.01:
            best_score = score
            best_sequence = pattern[:]
    
    # Try optimized patterns
    for pattern in optimized_patterns:
        score = evaluate_sequence_gradient_free(pattern)
        if score > best_score and score > 0.01:
            best_score = score
            best_sequence = pattern[:]
    
    # Try different pattern types with different lengths (more comprehensive)
    pattern_types = ["geometric", "exponential_decay", "linear_decay", "inverse_square"]
    lengths = [10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500]
    
    for pattern_type in pattern_types:
        for length in lengths:
            # Try a few different parameters for each pattern type
            if pattern_type == "geometric":
                # Try multiple ratios with more precision
                ratios = [0.75, 0.8, 0.82, 0.85, 0.88, 0.9]
                for ratio in ratios:
                    sequence = [1000 * (ratio ** i) for i in range(length)]
                    total = sum(sequence)
                    if total > 0:
                        sequence = [x * 1000 / total for x in sequence]
                    score = evaluate_sequence_gradient_free(sequence)
                    if score > best_score and score > 0.01:
                        best_score = score
                        best_sequence = sequence[:]
            elif pattern_type == "exponential_decay":
                # Try different decay rates with more granularity
                rates = [0.08, 0.1, 0.12, 0.15, 0.18, 0.2]
                for rate in rates:
                    sequence = [np.exp(-i * rate) for i in range(length)]
                    total = sum(sequence)
                    if total > 0:
                        sequence = [x * 1000 / total for x in sequence]
                    score = evaluate_sequence_gradient_free(sequence)
                    if score > best_score and score > 0.01:
                        best_score = score
                        best_sequence = sequence[:]
            else:
                sequence = create_analytical_sequence(length, pattern_type)
                score = evaluate_sequence_gradient_free(sequence)
                if score > best_score and score > 0.01:
                    best_score = score
                    best_sequence = sequence[:]
    
    # Try multi-pattern approach with more variation
    for length in [15, 20, 25, 30, 40, 50, 75, 100, 150, 200]:
        sequence = create_multi_pattern_sequence(length)
        score = evaluate_sequence_gradient_free(sequence)
        if score > best_score and score > 0.01:
            best_score = score
            best_sequence = sequence[:]
    
    return best_sequence if best_sequence is not None else [1.0] * 100

def optimize_with_scipy_methods(initial_sequence: List[float]) -> List[float]:
    """Use scipy's optimization methods with enhanced parameters"""
    # Define objective function to minimize (negative of our target)
    def objective(x):
        # Ensure non-negativity
        x = np.maximum(x, 0)
        score = evaluate_sequence_gradient_free(x)
        return -score if score > 0 else 1000  # Large penalty for invalid
    
    # Use differential evolution for global optimization with better settings
    try:
        bounds = [(0, 1000) for _ in range(len(initial_sequence))]
        # More thorough optimization with better parameters
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=400,      # More iterations for better convergence
            popsize=30,       # Larger population for better exploration
            seed=42,          # Fixed seed for reproducibility
            polish=True,      # Polish with local optimizer
            disp=False,       # Suppress output
            tol=0.0005        # Tighter tolerance
        )
        return result.x.tolist()
    except Exception as e:
        # Fallback to simpler optimization if needed
        return initial_sequence

def local_improvement_refinement(sequence: List[float], max_iterations: int = 400) -> List[float]:
    """Enhanced local improvement with more sophisticated strategies"""
    current = sequence.copy()
    current_score = evaluate_sequence_gradient_free(current)
    
    # Track improvement history for adaptive stopping
    recent_improvements = []
    
    # More aggressive mutation strategy
    mutation_probability = 0.7
    for iteration in range(max_iterations):
        # Create neighbor by small perturbations with more controlled approach
        neighbor = current.copy()
        
        # Apply mixed mutation strategies with better balance
        for i in range(len(neighbor)):
            if random.random() < mutation_probability:
                # 70% chance of small perturbation
                if random.random() < 0.7:
                    # Small Gaussian perturbation
                    if neighbor[i] > 0:
                        std_dev = neighbor[i] * 0.08  # 8% of current value
                    else:
                        std_dev = 15.0
                    neighbor[i] = max(0, neighbor[i] + random.gauss(0, std_dev))
                # 30% chance of larger adjustment
                else:
                    # Larger multiplicative change
                    if neighbor[i] > 0:
                        factor = random.uniform(0.85, 1.15)
                        neighbor[i] = max(0, neighbor[i] * factor)
        
        # Ensure minimum sum constraint
        if sum(neighbor) < 0.01:
            neighbor[0] = max(neighbor[0], 1.0)
            
        neighbor_score = evaluate_sequence_gradient_free(neighbor)
        
        # Accept if better or sometimes accept worse (with probability based on difference)
        if neighbor_score > current_score:
            current = neighbor
            current_score = neighbor_score
            recent_improvements.append(True)
        else:
            recent_improvements.append(False)
        
        # Adaptive stopping based on recent improvements
        if len(recent_improvements) > 25:
            recent_improvements = recent_improvements[-25:]
            if sum(recent_improvements) < 5:  # Very few improvements recently
                break
    
    return current

def search_for_best_sequence():
    """Use an enhanced hybrid approach combining analytical construction and optimization"""
    
    print("Starting enhanced hybrid search...")
    
    # Strategy 1: Comprehensive pattern initialization
    print("Strategy 1: Advanced pattern initialization")
    best_sequence = optimize_with_direct_search()
    
    # Strategy 2: Enhanced scipy optimization with more thorough search
    print("Strategy 2: Enhanced scipy optimization")
    scipy_sequence = optimize_with_scipy_methods(best_sequence)
    score1 = evaluate_sequence_gradient_free(best_sequence)
    score2 = evaluate_sequence_gradient_free(scipy_sequence)
    
    if score2 > score1:
        best_sequence = scipy_sequence
        print(f"Improved score from scipy optimization: {score2:.6f}")
    
    # Strategy 3: Local refinement with enhanced methods
    print("Strategy 3: Local refinement")
    refined_sequence = local_improvement_refinement(best_sequence, max_iterations=300)
    score3 = evaluate_sequence_gradient_free(refined_sequence)
    
    if score3 > score2:
        best_sequence = refined_sequence
        print(f"Improved score from local refinement: {score3:.6f}")
    
    # Strategy 4: Try additional pattern variations
    print("Strategy 4: Additional pattern variations")
    # Try with a different initial pattern based on the best found so far
    alt_sequence = create_multi_pattern_sequence(len(best_sequence))
    score4 = evaluate_sequence_gradient_free(alt_sequence)
    
    if score4 > score3:
        best_sequence = alt_sequence
        print(f"Improved score from alternative pattern: {score4:.6f}")
    
    # Strategy 5: Final fine-tuning with multiple local searches
    print("Strategy 5: Final fine-tuning")
    final_sequence = local_improvement_refinement(best_sequence, max_iterations=200)
    score5 = evaluate_sequence_gradient_free(final_sequence)
    
    if score5 > score4:
        best_sequence = final_sequence
        print(f"Improved score from final tuning: {score5:.6f}")
    
    # Final validation
    final_score = evaluate_sequence_gradient_free(best_sequence)
    if final_score < 0.01 or np.sum(best_sequence) < 0.01:
        # Return a fallback sequence
        print("Returning fallback sequence")
        return [1.0] * 100
    
    print(f"Final best score achieved: {final_score:.6f}")
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
