# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import fftconvolve
from scipy.optimize import minimize, linprog
import time
from typing import List, Tuple
import math
from numba import jit

def compute_autocorrelation_constant(sequence: List[float]) -> float:
    """
    Compute the first autocorrelation constant C1 for a given sequence.
    Returns 1/C1 where we want to maximize this value.
    """
    if len(sequence) == 0:
        return 0.0
    
    # Ensure we have a valid sequence with sum > 0.01
    sum_a = sum(sequence)
    if sum_a < 0.01:
        return 0.0
    
    # Compute convolution using FFT for efficiency
    a = np.array(sequence)
    conv_result = fftconvolve(a, a, mode='full')
    
    # The convolution result has length 2*n - 1
    # We want the maximum value among the middle terms (the actual autoconvolution)
    # The center of the full convolution corresponds to index n-1
    max_conv = np.max(conv_result)
    
    # Compute C1 = 2*n * max(b) / (sum(a))^2
    n = len(sequence)
    c1 = 2 * n * max_conv / (sum_a ** 2)
    
    # Return 1/C1 as our objective (we want to maximize this)
    return 1.0 / c1 if c1 > 0 else 0.0

def solve_convolution_lp(f_sequence, rhs):
    """
    Solves the convolution LP for a given sequence and RHS.
    Based on the mathematical approach from successful inspirations.
    """
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    
    # Build constraint matrix for convolution constraints
    # For each k, we want sum_{i+j=k} f[i]*f[j] <= rhs
    # This translates to: sum_{j=0}^{n-1} f[j] * (1 if j+k-i<n else 0) <= rhs
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: g_i >= 0
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    try:
        result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', 
                        options={'disp': False, 'presolve': True, 'maxiter': 10000})
    except:
        try:
            result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='simplex', 
                            options={'disp': False, 'maxiter': 10000})
        except:
            return None

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        return None

def get_good_direction_to_move_into(sequence):
    """
    Returns the direction to move into the sequence based on LP optimization.
    This is the key optimization technique from successful inspirations.
    """
    n = len(sequence)
    if n == 0:
        return None
        
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None
        
    # Normalize the sequence appropriately (as done in INSPIRATION PROGRAM 3)
    # Using sqrt(2*n) normalization for consistency with mathematical formulations
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    
    # Compute the RHS for the LP constraint (maximum convolution value)
    conv_result = np.convolve(normalized_sequence, normalized_sequence)
    rhs = np.max(conv_result)
    
    # Solve the LP to find a better direction
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    
    if g_fun is None:
        # If LP fails, try a more sophisticated fallback approach
        # This enhanced approach applies more targeted peak reduction
        try:
            # Create a more refined pattern that focuses on reducing peak convolution
            new_sequence = sequence.copy()
            
            # First, identify and reduce significant peaks more aggressively
            peak_positions = []
            # Find approximate peak positions in the sequence
            for i in range(1, len(sequence)-1):
                if sequence[i] > sequence[i-1] and sequence[i] > sequence[i+1]:
                    peak_positions.append(i)
            
            # Apply more aggressive reduction to top peaks
            sorted_peaks = sorted(peak_positions, key=lambda p: sequence[p], reverse=True)
            for i, pos in enumerate(sorted_peaks[:2]):  # Focus on top 2 peaks
                # More aggressive reduction for top peaks
                for j in range(max(0, pos-3), min(len(sequence), pos+4)):
                    if j != pos:
                        # Apply different reduction factors based on proximity to peak
                        distance = abs(j - pos)
                        if distance <= 1:
                            new_sequence[j] *= 0.91  # Very aggressive near peak
                        elif distance <= 2:
                            new_sequence[j] *= 0.94  # Aggressive
                        else:
                            new_sequence[j] *= 0.97  # Mild
            
            # Also apply smoothing to reduce sharp transitions
            smoothed = new_sequence.copy()
            for i in range(1, len(new_sequence)-1):
                smoothed[i] = 0.15 * new_sequence[i-1] + 0.7 * new_sequence[i] + 0.15 * new_sequence[i+1]
            new_sequence = smoothed
            
            # Ensure non-negativity
            new_sequence = [max(0, x) for x in new_sequence]
            return new_sequence
        except:
            # Last resort: return a simple peak reduction approach
            try:
                adjusted = sequence.copy()
                # Apply uniform reduction with more careful consideration
                avg_val = np.mean(adjusted)
                for i in range(len(adjusted)):
                    if adjusted[i] > avg_val * 1.2:
                        adjusted[i] *= 0.93  # Slightly more aggressive for outliers
                return adjusted
            except:
                return None
        
    # Normalize the resulting sequence
    sum_g = np.sum(g_fun)
    if sum_g < 0.01:
        return None
        
    # Scale back to original magnitude using sqrt(2*n) normalization
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]
    
    # Mix with original sequence to create new candidate
    # Slightly more aggressive mixing to encourage exploration
    t = 0.04  # Slightly higher mixing factor for more exploration
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
    
    return new_sequence

@jit(nopython=True)
def fast_convolution(a):
    """Fast convolution computation using Numba"""
    n = len(a)
    b = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            b[i + j] += a[i] * a[j]
    return b

def compute_autocorrelation_constant_fast(sequence: List[float]) -> float:
    """
    Fast computation of autocorrelation constant using numba-optimized convolution
    """
    if len(sequence) == 0:
        return 0.0
    
    # Ensure we have a valid sequence with sum > 0.01
    sum_a = sum(sequence)
    if sum_a < 0.01:
        return 0.0
    
    # Compute convolution using fast Numba implementation
    a = np.array(sequence)
    conv_result = fast_convolution(a)
    
    # Get maximum value in convolution
    max_conv = np.max(conv_result)
    
    # Compute C1 = 2*n * max(b) / (sum(a))^2
    n = len(sequence)
    c1 = 2 * n * max_conv / (sum_a ** 2)
    
    # Return 1/C1 as our objective (we want to maximize this)
    return 1.0 / c1 if c1 > 0 else 0.0

def generate_advanced_mathematical_pattern(n: int) -> List[float]:
    """
    Generate an advanced mathematical pattern inspired by optimal sequences
    This pattern is specifically tuned to minimize convolution peaks while maintaining good energy distribution
    """
    sequence = [0.0] * n
    
    # Create a sophisticated pattern that combines:
    # 1. Fast initial decay to concentrate energy early
    # 2. Oscillatory components to distribute energy evenly
    # 3. Careful tapering to prevent large final convolution contributions
    
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        
        # Base exponential decay with controlled rate
        base_decay = math.exp(-2.5 * t)
        
        # Multiple oscillatory components with different frequencies and amplitudes
        osc1 = 0.25 * math.sin(8 * math.pi * t)      # High frequency
        osc2 = 0.2 * math.cos(10 * math.pi * t)     # Medium-high frequency
        osc3 = 0.15 * math.sin(14 * math.pi * t)    # Medium frequency
        osc4 = 0.1 * math.cos(18 * math.pi * t)     # Low-medium frequency
        
        # Additional component for fine-tuning
        extra = 0.05 * math.sin(24 * math.pi * t) * math.cos(6 * math.pi * t)
        
        # Combined amplitude with careful weighting to avoid convolution spikes
        amplitude = 1000 * (base_decay + 0.3 * osc1 + 0.25 * osc2 + 0.2 * osc3 + 0.15 * osc4 + 0.05 * extra)
        
        # Apply smoothing to avoid sharp transitions
        if i > 0 and i < n - 1:
            # Simple averaging with neighbors for smoothing
            smooth_factor = 0.2
            prev_val = sequence[i-1]
            next_val = sequence[i+1] if i+1 < n else 0
            amplitude = (1 - smooth_factor) * amplitude + smooth_factor * (prev_val + next_val) / 2
        
        sequence[i] = max(0, amplitude)
    
    # Normalize to ensure reasonable sum
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_balanced_sequence(n: int) -> List[float]:
    """
    Generate a balanced sequence that spreads energy evenly.
    """
    sequence = [0.0] * n
    
    # Create a pattern with gradual decay and oscillation
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        
        # Gradual exponential decay with oscillation to distribute energy
        base = math.exp(-2 * t)
        oscillation = 0.2 * math.sin(6 * math.pi * t) + 0.15 * math.cos(8 * math.pi * t)
        amplitude = 1000 * (base + 0.3 * oscillation)
        
        sequence[i] = max(0, amplitude)
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_final_pattern(n: int) -> List[float]:
    """
    Generate the final optimized pattern that closely matches what worked in inspirations
    """
    # Pattern based on mathematical analysis that works well for this problem
    sequence = []
    
    # Create a pattern that combines:
    # 1. Initial high values to concentrate early energy
    # 2. Controlled decay to reduce later convolution contributions
    # 3. Oscillations to spread energy evenly without creating spikes
    
    for i in range(n):
        # Early part: rapid decay with oscillation
        if i < n // 3:
            # Quick initial decay with oscillation
            value = 1000 * (0.9 + 0.1 * math.sin(6 * math.pi * i / (n/3)))
        # Middle part: slower decay with oscillation
        elif i < 2 * n // 3:
            t = (i - n//3) / (n//3)
            base = math.exp(-1.5 * t)
            oscillation = 0.1 * math.sin(8 * math.pi * t) + 0.05 * math.cos(10 * math.pi * t)
            value = 1000 * (base + 0.15 * oscillation)
        # Late part: tapering with oscillation
        else:
            t = (i - 2*n//3) / (n//3)
            value = 1000 * math.exp(-3 * t) * (0.2 + 0.1 * math.sin(4 * math.pi * t))
        
        sequence.append(max(0, value))
    
    # Normalize to ensure good scale
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_oscillating_pattern(n: int) -> List[float]:
    """
    Generate an oscillating pattern that can help reduce convolution peaks
    """
    sequence = []
    for i in range(n):
        # Create a pattern with multiple oscillation frequencies
        freq1 = math.sin(0.5 * math.pi * i / n)
        freq2 = math.cos(0.8 * math.pi * i / n)
        freq3 = math.sin(1.2 * math.pi * i / n) * math.cos(0.6 * math.pi * i / n)
        
        # Combine with exponential decay to focus energy early
        decay = math.exp(-0.02 * i)
        amplitude = 1000 * (0.7 * decay + 0.2 * freq1 + 0.1 * freq2 + 0.05 * freq3)
        sequence.append(max(0, amplitude))
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_composite_pattern(n: int) -> List[float]:
    """
    Generate a composite pattern combining multiple mathematical approaches
    """
    sequence = [0.0] * n
    
    # Mix different approaches: exponential decay, oscillation, and geometric patterns
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        
        # Exponential decay component
        exp_component = 1000 * math.exp(-2 * t)
        
        # Oscillation component  
        osc_component = 100 * math.sin(8 * math.pi * t) * math.cos(4 * math.pi * t)
        
        # Geometric-like component
        geo_component = 500 * math.exp(-0.5 * t) * (1 - t)**2
        
        amplitude = exp_component + osc_component + geo_component + 20
        
        sequence[i] = max(0, amplitude)
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_better_pattern(n: int) -> List[float]:
    """
    Generate an enhanced pattern that combines multiple effective strategies
    """
    # Use a hybrid approach that balances energy distribution with structural properties
    sequence = []
    
    # Create pattern with multiple phases
    for i in range(n):
        # Phase 1: Early high values with oscillation
        if i < n // 4:
            value = 1000 * (0.9 + 0.1 * math.sin(4 * math.pi * i / (n/4)))
        # Phase 2: Middle with controlled decay and oscillation
        elif i < 3 * n // 4:
            t = (i - n//4) / (n//2)
            base = math.exp(-2 * t)
            oscillation = 0.15 * math.sin(6 * math.pi * t) + 0.1 * math.cos(8 * math.pi * t)
            value = 1000 * (base + 0.2 * oscillation)
        # Phase 3: Late tapering
        else:
            t = (i - 3*n//4) / (n//4)
            value = 1000 * math.exp(-3 * t) * (0.3 + 0.2 * math.sin(2 * math.pi * t))
        
        sequence.append(max(0, value))
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_optimized_pattern(n: int) -> List[float]:
    """
    Generate a highly optimized pattern incorporating the best features from all inspirations
    """
    sequence = []
    
    # Create a sophisticated pattern with:
    # 1. Initial spike for concentration
    # 2. Controlled decay with oscillations
    # 3. Strategic tapering
    # 4. Smooth transitions
    
    for i in range(n):
        # Early rapid decay with oscillation
        if i < n // 4:
            # Strong initial values with oscillation
            value = 1000 * (0.95 + 0.05 * math.sin(10 * math.pi * i / (n/4)))
        # Middle controlled decay
        elif i < n // 2:
            t = (i - n//4) / (n//4)
            # Exponential decay with oscillation
            base = math.exp(-1.8 * t)
            oscillation = 0.1 * math.sin(12 * math.pi * t) + 0.05 * math.cos(14 * math.pi * t)
            value = 1000 * (base + 0.15 * oscillation)
        # Later tapering with oscillation
        elif i < 3 * n // 4:
            t = (i - n//2) / (n//4)
            # Tapering with oscillation
            base = math.exp(-2.5 * t)
            oscillation = 0.08 * math.sin(8 * math.pi * t) + 0.03 * math.cos(10 * math.pi * t)
            value = 1000 * (base + 0.1 * oscillation)
        # Final tapering
        else:
            t = (i - 3*n//4) / (n//4)
            # Very slow decay with gentle oscillation
            value = 1000 * math.exp(-3.5 * t) * (0.1 + 0.05 * math.sin(4 * math.pi * t))
        
        sequence.append(max(0, value))
    
    # Apply enhanced smoothing to reduce sharp transitions
    smoothed_sequence = sequence.copy()
    for i in range(1, n-1):
        # Use a more sophisticated smoothing kernel
        smoothed_sequence[i] = 0.2 * sequence[i-1] + 0.6 * sequence[i] + 0.2 * sequence[i+1]
    
    # Normalize
    total = sum(smoothed_sequence)
    if total > 0:
        smoothed_sequence = [x * 1000 / total for x in smoothed_sequence]
    
    return smoothed_sequence

def generate_mathematically_optimized_pattern(n: int) -> List[float]:
    """
    Generate a mathematically optimized pattern based on advanced insights
    from mathematical literature and optimization theory.
    """
    sequence = []
    
    # Use a pattern that incorporates:
    # 1. Golden ratio-based decay for optimal energy distribution
    # 2. Strategic oscillations to avoid convolution spikes
    # 3. Smooth transitions to reduce discontinuities
    
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    
    for i in range(n):
        # Use golden ratio for exponential decay with oscillation
        # This tends to produce optimal energy distributions for such problems
        base_decay = math.exp(-0.05 * i) * (1 - 0.01 * i)  # Slow decay with slight linear decrease
        oscillation = 0.1 * math.sin(2 * math.pi * i / (n/5)) + 0.05 * math.cos(4 * math.pi * i / (n/3))
        # Incorporate golden ratio properties
        golden_factor = 1.0 / (1 + 0.02 * i)  # Gradually decreasing influence
        
        amplitude = 1000 * (base_decay + 0.2 * oscillation + 0.1 * golden_factor)
        sequence.append(max(0, amplitude))
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_fibonacci_pattern(n: int) -> List[float]:
    """
    Generate a pattern based on Fibonacci sequence properties
    """
    # Generate Fibonacci-like sequence
    fib = [1, 1]
    for i in range(n - 2):
        fib.append(fib[-1] + fib[-2])
    
    # Normalize and scale to 1000
    fib = [x * 1000 / max(fib) for x in fib]
    
    # Apply exponential decay to reduce later convolution contributions
    sequence = []
    for i in range(n):
        # Apply exponential decay to the Fibonacci pattern
        decay_factor = math.exp(-0.03 * i)
        sequence.append(max(0, fib[i] * decay_factor))
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_golden_ratio_pattern(n: int) -> List[float]:
    """
    Generate a pattern based on golden ratio properties
    """
    phi = (1 + math.sqrt(5)) / 2
    sequence = []
    
    # Golden ratio decay with oscillation
    for i in range(n):
        # Exponential decay with golden ratio influence
        base = math.exp(-0.04 * i) * (1 - 0.005 * i)
        oscillation = 0.1 * math.sin(2 * math.pi * i / (n/4)) + 0.05 * math.cos(4 * math.pi * i / (n/3))
        value = 1000 * (base + 0.15 * oscillation)
        sequence.append(max(0, value))
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def advanced_local_search(initial_sequence: List[float], max_iter: int = 2000) -> List[float]:
    """
    Enhanced local search with better neighborhood exploration and convergence
    """
    current_seq = initial_sequence.copy()
    current_score = compute_autocorrelation_constant(current_seq)
    
    # Track improvement for early stopping
    last_improvement = 0
    consecutive_no_improvement = 0
    max_consecutive_no_improvement = 500
    
    # More diverse neighborhood operators with better weight distribution
    operators = [
        ("small_changes", 0.16),      # Small random changes
        ("moderate_changes", 0.16),   # Moderate changes
        ("scaling", 0.13),            # Global scaling
        ("strategic_adjustments", 0.13), # Strategic position adjustments
        ("segment_shift", 0.11),       # Segment shifts
        ("adaptive_changes", 0.11),   # Adaptive changes
        ("random_segment", 0.09),     # Random segments
        ("pattern_based", 0.09),      # Pattern-based mutations
        ("gaussian_noise", 0.09),     # Gaussian noise for fine tuning
        ("weighted_avg", 0.06),       # Weighted average modification
        ("peak_reduction", 0.06),     # Targeted peak reduction
        ("energy_balance", 0.06),     # Energy balance adjustments
        ("gradient_like", 0.06),      # Gradient-like adjustments
        ("global_smooth", 0.05),      # Global smoothing
    ]
    
    for iteration in range(max_iter):
        # Create multiple neighbor types for thorough exploration
        neighbors = []
        
        # Generate neighbors with different strategies
        for name, prob in operators:
            if random.random() < prob:
                neighbor = current_seq.copy()
                
                if name == "small_changes":
                    # Small random changes for fine tuning
                    num_changes = min(8, len(neighbor) // 4)
                    for _ in range(num_changes):
                        idx = random.randint(0, len(neighbor) - 1)
                        change = random.gauss(0, 15)
                        neighbor[idx] = max(0, neighbor[idx] + change)
                        
                elif name == "moderate_changes":
                    # Moderate changes for larger adjustments
                    num_changes = min(4, len(neighbor) // 6)
                    for _ in range(num_changes):
                        idx = random.randint(0, len(neighbor) - 1)
                        change = random.gauss(0, 50)
                        neighbor[idx] = max(0, neighbor[idx] + change)
                        
                elif name == "scaling":
                    # Global scaling
                    scale_factor = random.uniform(0.95, 1.05)
                    neighbor = [max(0, x * scale_factor) for x in neighbor]
                    
                elif name == "strategic_adjustments":
                    # Adjust some strategic positions
                    positions_to_adjust = random.sample(range(len(neighbor)), min(3, len(neighbor) // 10))
                    for idx in positions_to_adjust:
                        neighbor[idx] = max(0, neighbor[idx] + random.gauss(0, 30))
                        
                elif name == "segment_shift":
                    # Apply a shift to a segment
                    start_idx = random.randint(0, len(neighbor) - 5)
                    end_idx = min(start_idx + 5, len(neighbor))
                    shift_amount = random.gauss(0, 20)
                    for i in range(start_idx, end_idx):
                        neighbor[i] = max(0, neighbor[i] + shift_amount)
                        
                elif name == "adaptive_changes":
                    # Adaptive changes based on current values
                    for i in range(len(neighbor)):
                        if neighbor[i] > 500:
                            change = random.gauss(0, 10)
                        else:
                            change = random.gauss(0, 30)
                        neighbor[i] = max(0, neighbor[i] + change)
                        
                elif name == "random_segment":
                    # Randomized segment adjustments
                    segment_size = max(1, len(neighbor) // 10)
                    start_idx = random.randint(0, len(neighbor) - segment_size)
                    for i in range(start_idx, min(start_idx + segment_size, len(neighbor))):
                        change = random.gauss(0, 25)
                        neighbor[i] = max(0, neighbor[i] + change)
                        
                elif name == "pattern_based":
                    # Pattern-based mutations with enhanced structure awareness
                    for i in range(len(neighbor)):
                        # If at beginning or end, make more significant changes
                        if i < 5 or i > len(neighbor) - 5:
                            change = random.gauss(0, 40)
                        else:
                            change = random.gauss(0, 15)
                        neighbor[i] = max(0, neighbor[i] + change)
                        
                elif name == "gaussian_noise":
                    # Add Gaussian noise to all elements
                    for i in range(len(neighbor)):
                        noise = random.gauss(0, 20)
                        neighbor[i] = max(0, neighbor[i] + noise)
                        
                elif name == "weighted_avg":
                    # Apply weighted average with neighbors to smooth transitions
                    if len(neighbor) >= 3:
                        new_neighbor = neighbor.copy()
                        for i in range(1, len(neighbor) - 1):
                            new_neighbor[i] = 0.3 * neighbor[i-1] + 0.4 * neighbor[i] + 0.3 * neighbor[i+1]
                        neighbor = new_neighbor
                        
                elif name == "peak_reduction":
                    # Targeted peak reduction to reduce convolution spikes
                    if len(neighbor) > 10:
                        # Find peak locations and reduce nearby values
                        peak_positions = []
                        for i in range(1, len(neighbor)-1):
                            if neighbor[i] > neighbor[i-1] and neighbor[i] > neighbor[i+1]:
                                peak_positions.append(i)
                        
                        # Reduce values near peaks with more aggressive approach
                        for pos in peak_positions[:3]:  # Limit to top 3 peaks
                            start = max(0, pos - 2)
                            end = min(len(neighbor), pos + 3)
                            for i in range(start, end):
                                if i != pos:
                                    # Apply different reduction rates based on distance
                                    distance = abs(i - pos)
                                    if distance == 0:
                                        continue  # Skip the peak itself
                                    elif distance <= 1:
                                        neighbor[i] *= 0.92  # Very aggressive near peak
                                    elif distance <= 2:
                                        neighbor[i] *= 0.95  # Aggressive
                                    else:
                                        neighbor[i] *= 0.97  # Mild
                        
                elif name == "energy_balance":
                    # Energy balance adjustment to maintain good distribution
                    # Calculate current distribution statistics
                    total_energy = sum(neighbor)
                    if total_energy > 0:
                        avg_energy = total_energy / len(neighbor)
                        # Adjust to make distribution more uniform
                        for i in range(len(neighbor)):
                            # If too high, reduce; if too low, increase
                            if neighbor[i] > 2 * avg_energy:
                                neighbor[i] *= 0.93  # More aggressive reduction for outliers
                            elif neighbor[i] < 0.5 * avg_energy:
                                neighbor[i] *= 1.02  # Slightly more aggressive increase
                        
                elif name == "gradient_like":
                    # Gradient-like adjustment to reduce peaks more aggressively
                    # This is a heuristic that mimics gradient descent behavior
                    try:
                        # Identify positions that contribute most to high convolution
                        # Simple heuristic: reduce values that are significantly higher than average
                        avg_val = np.mean(neighbor)
                        for i in range(len(neighbor)):
                            if neighbor[i] > avg_val * 1.5:
                                neighbor[i] *= 0.91  # Even more aggressive reduction for outliers
                    except:
                        pass
                        
                elif name == "global_smooth":
                    # Apply global smoothing to reduce sharp transitions
                    if len(neighbor) >= 5:
                        new_neighbor = neighbor.copy()
                        # Use a stronger smoothing kernel
                        for i in range(2, len(neighbor) - 2):
                            new_neighbor[i] = (
                                0.1 * neighbor[i-2] + 
                                0.2 * neighbor[i-1] + 
                                0.4 * neighbor[i] + 
                                0.2 * neighbor[i+1] + 
                                0.1 * neighbor[i+2]
                            )
                        neighbor = new_neighbor
                        
                neighbors.append((name, neighbor))
        
        # Evaluate all neighbors
        best_neighbor = current_seq
        best_score = current_score
        
        for name, neighbor in neighbors:
            neighbor_score = compute_autocorrelation_constant(neighbor)
            if neighbor_score > best_score:
                best_score = neighbor_score
                best_neighbor = neighbor
        
        # Accept the best neighbor
        if best_score > current_score:
            current_seq = best_neighbor
            current_score = best_score
            last_improvement = iteration
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
            # Occasionally accept worse solutions to escape local minima
            if consecutive_no_improvement > 15 and random.random() < 0.05:
                current_seq = best_neighbor
                current_score = best_score
                last_improvement = iteration
                consecutive_no_improvement = 0
        
        # Early stopping if no improvement for a while
        if iteration - last_improvement > max_consecutive_no_improvement:
            break
    
    return current_seq

def iterative_improvement_search(max_iterations: int = 1200) -> List[float]:
    """
    Iterative improvement approach inspired by successful inspirations
    """
    # Start with the best pattern from our analysis
    n = 200  # Larger size for better optimization potential
    sequence = generate_optimized_pattern(n)
    
    best_sequence = sequence[:]
    best_inv_c1 = compute_autocorrelation_constant(best_sequence)
    
    print(f"Initial score: {best_inv_c1:.6f}")
    
    # Track convergence history for early stopping
    convergence_history = []
    patience = 0
    max_patience = 100
    
    # Apply iterative improvement using the LP-based approach
    for iteration in range(max_iterations):
        # Try to improve with the LP-based direction
        improved_sequence = get_good_direction_to_move_into(best_sequence)
        
        if improved_sequence is not None:
            inv_c1 = compute_autocorrelation_constant(improved_sequence)
            if inv_c1 > best_inv_c1:
                best_sequence = improved_sequence
                best_inv_c1 = inv_c1
                print(f"Iteration {iteration}: Improved to {best_inv_c1:.6f}")
                convergence_history = []  # Reset history when improvement found
                patience = 0
            else:
                convergence_history.append(best_inv_c1)
                patience += 1
        else:
            convergence_history.append(best_inv_c1)
            patience += 1
        
        # Early stopping based on convergence
        if len(convergence_history) >= 10:
            recent_scores = convergence_history[-10:]
            if len(set(recent_scores)) == 1:  # No change in last 10 iterations
                print(f"Converged after {iteration} iterations")
                break
                
        # Occasionally try a different approach to escape local optima
        if iteration % 50 == 0 and iteration > 0:
            # Try a pattern with more oscillation for diversity
            oscillation_pattern = []
            for i in range(n):
                # More aggressive oscillation
                value = 1000 * (0.8 + 0.2 * math.sin(12 * math.pi * i / n))
                oscillation_pattern.append(max(0, value))
            
            # Normalize
            total = sum(oscillation_pattern)
            if total > 0:
                oscillation_pattern = [x * 1000 / total for x in oscillation_pattern]
                
            inv_c1 = compute_autocorrelation_constant(oscillation_pattern)
            if inv_c1 > best_inv_c1:
                best_sequence = oscillation_pattern
                best_inv_c1 = inv_c1
                print(f"Iteration {iteration}: Oscillation pattern to {best_inv_c1:.6f}")
                convergence_history = []
                patience = 0
        
        # Occasionally do a global restart with different pattern
        if iteration % 200 == 0 and iteration > 0:
            # Try different pattern types
            restart_patterns = [
                generate_advanced_mathematical_pattern(n),
                generate_balanced_sequence(n),
                generate_oscillating_pattern(n),
                generate_composite_pattern(n),
                generate_better_pattern(n),
                generate_mathematically_optimized_pattern(n),
                generate_fibonacci_pattern(n),
                generate_golden_ratio_pattern(n),
                [1000.0] * n,  # Constant
                [1000 * math.exp(-0.02 * i) for i in range(n)]  # Exponential decay
            ]
            
            for pattern in restart_patterns:
                inv_c1 = compute_autocorrelation_constant(pattern)
                if inv_c1 > best_inv_c1:
                    best_sequence = pattern
                    best_inv_c1 = inv_c1
                    print(f"Iteration {iteration}: Restarted pattern to {best_inv_c1:.6f}")
                    convergence_history = []
                    patience = 0
                    break
        
        # Check patience for early stopping
        if patience > max_patience:
            print(f"Patience exceeded after {iteration} iterations")
            break
    
    return best_sequence

def enhanced_systematic_search() -> List[float]:
    """
    Enhanced systematic search with more diverse mathematical patterns
    """
    best_score = 0.0
    best_sequence = []
    
    # Test several different mathematical structures
    structures_to_test = [
        # Advanced mathematical patterns
        lambda n: generate_advanced_mathematical_pattern(n),
        lambda n: generate_balanced_sequence(n),
        lambda n: generate_oscillating_pattern(n),
        lambda n: generate_composite_pattern(n),
        lambda n: generate_better_pattern(n),
        lambda n: generate_optimized_pattern(n),
        lambda n: generate_mathematically_optimized_pattern(n),
        lambda n: generate_fibonacci_pattern(n),
        lambda n: generate_golden_ratio_pattern(n),
        # Traditional patterns from inspiration 1
        lambda n: [1000 * math.exp(-0.02 * i) for i in range(n)],  # Exponential decay
        lambda n: [1000.0] * n,  # Constant
        # Symmetric patterns
        lambda n: [1.0] * (n//2) + [1.0] * (n - n//2) if n % 2 == 0 else [1.0] * (n//2) + [1.0] + [1.0] * (n - n//2 - 1),
    ]
    
    # Test different sequence lengths more thoroughly
    lengths_to_test = [100, 150, 200, 250, 300]
    
    for length in lengths_to_test:
        for struct_func in structures_to_test:
            try:
                seq = struct_func(length)
                score = compute_autocorrelation_constant(seq)
                
                if score > best_score:
                    best_score = score
                    best_sequence = seq
                    
                # Also run local optimization on this structure
                optimized = advanced_local_search(seq, 500)
                optimized_score = compute_autocorrelation_constant(optimized)
                
                if optimized_score > best_score:
                    best_score = optimized_score
                    best_sequence = optimized
                    
            except Exception as e:
                continue  # Skip problematic cases
    
    return best_sequence

def search_for_best_sequence() -> List[float]:
    """
    Main search function implementing the most effective approach from inspirations
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("Starting enhanced hybrid optimization search...")
    
    # Strategy 1: Enhanced systematic search
    print("Strategy 1: Enhanced systematic search")
    try:
        systematic_result = enhanced_systematic_search()
        systematic_score = compute_autocorrelation_constant(systematic_result)
        print(f"Enhanced systematic search result: score = {systematic_score:.6f}")
    except Exception as e:
        print(f"Enhanced systematic search failed: {e}")
        systematic_result = []
        systematic_score = 0.0
    
    # Strategy 2: Iterative improvement search
    print("Strategy 2: Iterative improvement search")
    try:
        ii_result = iterative_improvement_search(1200)
        ii_score = compute_autocorrelation_constant(ii_result)
        print(f"Iterative improvement result: score = {ii_score:.6f}")
    except Exception as e:
        print(f"Iterative improvement failed: {e}")
        ii_result = []
        ii_score = 0.0
    
    # Strategy 3: Final intensive refinement
    print("Strategy 3: Final intensive refinement")
    try:
        # Start with the best of the previous strategies
        candidates = [
            (systematic_result, systematic_score),
            (ii_result, ii_score)
        ]
        
        # Filter valid results
        valid_candidates = [(seq, score) for seq, score in candidates if seq and score > 0]
        
        if valid_candidates:
            best_candidate = max(valid_candidates, key=lambda x: x[1])
            start_seq = best_candidate[0]
        else:
            start_seq = generate_optimized_pattern(200)
        
        # Apply intensive local search with increased iterations
        final_result = advanced_local_search(start_seq, 2000)
        final_score = compute_autocorrelation_constant(final_result)
        print(f"Final intensive refinement result: score = {final_score:.6f}")
    except Exception as e:
        print(f"Final refinement failed: {e}")
        final_result = []
        final_score = 0.0
    
    # Choose the best overall result
    all_results = [
        (systematic_result, systematic_score),
        (ii_result, ii_score),
        (final_result, final_score)
    ]
    
    # Filter out invalid results
    valid_results = [(seq, score) for seq, score in all_results if seq and score > 0]
    
    if valid_results:
        best_result = max(valid_results, key=lambda x: x[1])
        print(f"Best final result score: {best_result[1]:.6f}")
        return best_result[0]
    else:
        # Fallback to a robust mathematical pattern
        print("Falling back to default pattern")
        return generate_optimized_pattern(200)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
