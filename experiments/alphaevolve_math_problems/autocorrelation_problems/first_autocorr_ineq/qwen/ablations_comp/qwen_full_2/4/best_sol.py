# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import time
import random
from typing import List, Tuple

# Global constants for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def compute_c1(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute C1 and 1/C1 for a given sequence using FFT for efficiency.
    Returns (C1, 1/C1)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Use FFT for efficient autoconvolution
    n = len(sequence)
    # Pad to avoid circular convolution effects
    padded_length = 2 * n - 1
    padded_seq = np.pad(sequence, (0, padded_length - n), mode='constant')
    
    # Compute autoconvolution using FFT
    fft_result = np.fft.fft(padded_seq)
    conv_fft = fft_result * np.conj(fft_result)
    autoconv = np.fft.ifft(conv_fft).real[:n]
    
    # Normalize to match the definition
    seq_sum = sum(sequence)
    if seq_sum < 0.01:
        return float('inf'), 0.0
    
    max_conv = np.max(autoconv)
    c1 = (2 * n * max_conv) / (seq_sum ** 2)
    
    # Return both C1 and its reciprocal
    return c1, 1.0 / c1 if c1 > 0 else 0.0

def compute_inv_c1(sequence: List[float]) -> float:
    """Compute just 1/C1 value for a given sequence."""
    c1, _ = compute_c1(sequence)
    if c1 == float('inf') or c1 == 0:
        return 0  # Return 0 for invalid sequences
    return 1.0 / c1 if c1 > 0 else 0

def get_good_direction_to_move_into(
    sequence: List[float],
) -> List[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None
        
    # Normalize the sequence properly for the LP
    normalized_sequence = [x / sum_sequence for x in sequence]
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
    
    # Solve the LP to find a better sequence
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_sequence = np.sum(g_fun)
    if sum_sequence < 0.01:
        return None
    # Normalize the result
    normalized_g_fun = [x / sum_sequence for x in g_fun]
    
    # Move towards the better sequence (but don't go too far)
    t = 0.05  # Slightly larger step
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    
    # Build constraint matrix for convolution
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
    b_ub_nonneg = np.zeros(n)  # Zero vector

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    # Use method='highs' which is more reliable for this type of problem
    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
    except:
        # Fallback to default method
        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub)
        except:
            return None

    if result.success:
        g_sequence = result.x
        # Ensure all values are non-negative (numerical precision issues)
        g_sequence = np.maximum(g_sequence, 0)
        return g_sequence
    else:
        return None

def generate_extreme_spike_sequence():
    """Generate an extremely concentrated spike sequence with maximum values."""
    sequence = [0.0] * 100
    
    # Create a spike with maximum possible concentration and value
    # Place very high value at center with minimal support
    sequence[48] = 50000000.0  # Very high value
    sequence[49] = 100000000.0  # Maximum possible value!
    sequence[50] = 200000000.0  # Maximum possible value!
    sequence[51] = 100000000.0
    sequence[52] = 50000000.0
    
    return sequence

def generate_ultra_concentrated_sequence():
    """Generate ultra-concentrated sequence with maximum peak."""
    sequence = [0.0] * 100
    
    # Extremely sharp peak with maximum concentration
    sequence[49] = 100000000.0  # Maximum value
    sequence[50] = 500000000.0  # Maximum possible value!
    sequence[51] = 100000000.0
    
    return sequence

def generate_maximum_concentration_pattern():
    """Generate pattern with maximum possible concentration."""
    sequence = [0.0] * 100
    
    # Create maximum concentration at center
    peak_positions = [48, 49, 50, 51, 52]
    peak_values = [100000000.0, 200000000.0, 500000000.0, 200000000.0, 100000000.0]
    
    for i, (pos, val) in enumerate(zip(peak_positions, peak_values)):
        sequence[pos] = val
    
    return sequence

def generate_mathematical_optimal_spike():
    """Generate a mathematically optimal spike sequence."""
    sequence = [0.0] * 100
    
    # Create a precisely optimized spike with maximum concentration
    spike_pos = 50
    sequence[spike_pos] = 10000000.0  # Very high value
    
    # Nearly perfect spike with minimal spread
    for i in range(max(0, spike_pos-1), min(100, spike_pos+2)):
        distance = abs(i - spike_pos)
        if distance == 0:
            sequence[i] = 10000000.0
        else:
            sequence[i] = 10000000.0 * (1 - distance * 0.95)  # Nearly perfect drop-off
    
    return sequence

def generate_analytical_spike():
    """Generate an analytically optimized spike using mathematical insights."""
    sequence = [0.0] * 100
    
    # Use a very sharp spike at center with optimal decay
    spike_pos = 50
    sequence[spike_pos] = 15000000.0  # Even higher value
    
    # Sharp decay with mathematical precision
    for i in range(max(0, spike_pos-1), min(100, spike_pos+2)):
        distance = abs(i - spike_pos)
        if distance == 0:
            sequence[i] = 15000000.0
        else:
            # Use 0.99 decay factor for nearly perfect spike
            sequence[i] = 15000000.0 * (1 - distance * 0.99)
    
    return sequence

def generate_extreme_hyper_spike():
    """Generate an extreme hyper optimized spike sequence."""
    sequence = [0.0] * 100
    
    # Create an extremely sharp and very high-value spike
    spike_pos = 50
    sequence[spike_pos] = 20000000.0  # Extremely high value
    
    # Very narrow and sharp spike
    for i in range(max(0, spike_pos-1), min(100, spike_pos+2)):
        distance = abs(i - spike_pos)
        if distance == 0:
            sequence[i] = 20000000.0
        else:
            sequence[i] = 20000000.0 * (1 - distance * 0.95)  # Very sharp drop-off
    
    return sequence

def generate_super_spike():
    """Generate a super optimized spike sequence."""
    sequence = [0.0] * 100
    
    # Create an extremely sharp and high-value spike
    spike_pos = 50
    sequence[spike_pos] = 50000000.0  # Even higher value
    
    # Super narrow and sharp spike
    for i in range(max(0, spike_pos-1), min(100, spike_pos+2)):
        distance = abs(i - spike_pos)
        if distance == 0:
            sequence[i] = 50000000.0
        else:
            sequence[i] = 50000000.0 * (1 - distance * 0.9)  # Very sharp drop-off
    
    return sequence

def generate_high_precision_spike():
    """Generate a high precision mathematical spike."""
    sequence = [0.0] * 100
    
    # Create a precise mathematical spike with maximum concentration
    spike_pos = 50
    sequence[spike_pos] = 100000000.0  # Very high value
    
    # Extremely sharp spike with mathematical precision
    for i in range(max(0, spike_pos-1), min(100, spike_pos+2)):
        distance = abs(i - spike_pos)
        if distance == 0:
            sequence[i] = 100000000.0
        else:
            sequence[i] = 100000000.0 * (1 - distance * 0.99)  # Nearly perfect spike
    
    return sequence

def generate_optimized_geometric_sequence():
    """Generate an optimized geometric decay sequence."""
    # Very aggressive geometric decay
    sequence = [0.999**i * 100000 for i in range(100)]
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x / total for x in sequence]
    
    return sequence

def generate_bell_shaped_sequence():
    """Generate a bell-shaped sequence that often performs well."""
    sequence = [np.exp(-((i-50)**2)/(2*1**2)) * 100000 for i in range(100)]
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x / total for x in sequence]
    
    return sequence

def generate_sparse_optimized_sequence():
    """Generate a sparse sequence with optimized high-value elements."""
    sequence = [0.0] * 100
    
    # Place a few optimally spaced high-value elements
    positions = [20, 50, 80]
    for pos in positions:
        sequence[pos] = 10000000.0  # Very high value
        
        # Add narrow support around each peak
        for j in range(max(0, pos-1), min(100, pos+2)):
            distance = abs(j - pos)
            if distance == 0:
                sequence[j] = 10000000.0
            else:
                sequence[j] = 10000000.0 * (1 - distance * 0.9)
    
    return sequence

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    best_sequence = None
    best_inv_c1 = 0.0
    start_time = time.time()
    
    # Strategy 1: Try extreme spike constructions (inspired by highest performing approaches)
    extreme_patterns = [
        generate_extreme_spike_sequence(),
        generate_ultra_concentrated_sequence(), 
        generate_maximum_concentration_pattern(),
        generate_mathematical_optimal_spike(),
        generate_analytical_spike(),
        generate_extreme_hyper_spike(),
        generate_super_spike(),
        generate_high_precision_spike()
    ]
    
    for pattern in extreme_patterns:
        if time.time() - start_time > 45:  # Time limit
            break
            
        # Normalize the pattern
        total = sum(pattern)
        if total > 0:
            sequence = [x / total for x in pattern]
        else:
            continue
            
        # Apply extremely aggressive optimization with maximum iterations
        max_iterations = 1500  # Much higher than before
        for iteration in range(max_iterations):
            if time.time() - start_time > 45:  # Time limit
                break
            h_function = get_good_direction_to_move_into(sequence)
            if h_function is not None:
                sequence = h_function
            else:
                # If optimization fails, do a small random perturbation with high variance
                sequence = [x * (0.999 + 0.002 * random.random()) for x in sequence]
                total = sum(sequence)
                if total > 0:
                    sequence = [x / total for x in sequence]
        
        # Check result
        inv_c1 = compute_inv_c1(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()
    
    # Strategy 2: Enhanced mathematical patterns with more aggressive optimization
    # Use the same patterns as inspiration #3 but with more aggressive optimization
    patterns = [
        # Strong geometric decays (these are the most promising)
        [0.9**i for i in range(100)],
        [0.85**i for i in range(100)],
        [0.95**i for i in range(100)],
        # Harmonic decay
        [1.0/(i+1) for i in range(100)],
        # Uniform distribution
        [1.0/100]*100,
        # Concave-like shape
        [1.0 if i < 20 else 0.0 for i in range(100)],
        # Slight exponential decay with adjustment
        [0.9**i * (1.0 - 0.01*i) for i in range(100)],
        # Another geometric variant
        [0.8**i for i in range(100)],
        # Different geometric base
        [0.88**i for i in range(100)],
        # Exponential decay with different rate
        [0.75**i for i in range(100)],
        # Very concentrated peak
        [1.0 if i < 5 else 0.0 for i in range(100)],
        # Smooth bell-shaped
        [np.exp(-((i-50)**2)/(2*10**2)) for i in range(100)],
        # Optimized geometric
        generate_optimized_geometric_sequence(),
        # Bell-shaped
        generate_bell_shaped_sequence(),
        # Sparse optimized
        generate_sparse_optimized_sequence()
    ]
    
    for i, pattern in enumerate(patterns):
        if time.time() - start_time > 45:  # Time limit
            break
            
        # Normalize the pattern
        total = sum(pattern)
        if total > 0:
            sequence = [x / total for x in pattern]
        else:
            continue
        
        # Apply MORE aggressive optimization process with even more iterations
        max_iterations = 2000  # Even more aggressive iterations
        for iteration in range(max_iterations):
            if time.time() - start_time > 45:  # Time limit
                break
            h_function = get_good_direction_to_move_into(sequence)
            if h_function is not None:
                sequence = h_function
            else:
                # If optimization fails, do a small random perturbation with high variance
                sequence = [x * (0.999 + 0.002 * random.random()) for x in sequence]
                total = sum(sequence)
                if total > 0:
                    sequence = [x / total for x in sequence]
        
        # Check result
        inv_c1 = compute_inv_c1(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()
    
    # Strategy 3: Final aggressive refinement with maximum concentration
    if best_sequence is not None and time.time() - start_time < 50:
        # Try the absolute most extreme mathematical construction
        final_sequence = [0.0] * 100
        # Extremely high value peak with perfect concentration
        final_sequence[48] = 10000000.0
        final_sequence[49] = 20000000.0  # Very high peak
        final_sequence[50] = 100000000.0  # Maximum peak value!
        final_sequence[51] = 20000000.0
        final_sequence[52] = 10000000.0
        
        # Perform extremely aggressive refinement with very high iteration count
        for iteration in range(2500):  # Very aggressive refinement
            if time.time() - start_time > 50:  # Time limit
                break
            h_function = get_good_direction_to_move_into(final_sequence)
            if h_function is not None:
                # Use very aggressive step size for fast convergence
                t = 0.02
                final_sequence = [(1 - t) * x + t * y for x, y in zip(final_sequence, h_function)]
            else:
                # If optimization fails, do a small random perturbation with high variance
                final_sequence = [x * (0.9995 + 0.001 * random.random()) for x in final_sequence]
                total = sum(final_sequence)
                if total > 0:
                    final_sequence = [x / total for x in final_sequence]
        
        final_inv_c1 = compute_inv_c1(final_sequence)
        if final_inv_c1 > best_inv_c1:
            best_inv_c1 = final_inv_c1
            best_sequence = final_sequence
    
    # If we still haven't found anything good, use a very aggressive geometric sequence
    if best_sequence is None:
        sequence = [0.9999**i * 1000000 for i in range(100)]
        total = sum(sequence)
        if total > 0:
            sequence = [x / total for x in sequence]
        best_sequence = sequence
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
