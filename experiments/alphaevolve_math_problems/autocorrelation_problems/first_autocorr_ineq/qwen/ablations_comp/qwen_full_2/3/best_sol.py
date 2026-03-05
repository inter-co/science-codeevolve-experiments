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

def compute_c1_fft(sequence: np.ndarray) -> Tuple[float, float]:
    """
    Compute C1 and 1/C1 for a given sequence using FFT for efficiency.
    Returns (C1, 1/C1)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Use FFT for efficient autoconvolution
    n = len(sequence)
    
    # Check sum before proceeding
    seq_sum = np.sum(sequence)
    if seq_sum < 0.01:
        return float('inf'), 0.0
    
    # Compute autoconvolution using FFT
    conv_result = fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(conv_result[1:])  # Exclude zero-lag term
    
    c1 = (2 * n * max_conv) / (seq_sum ** 2)
    
    # Return both C1 and its reciprocal
    return c1, 1.0 / c1 if c1 > 0 else 0.0

def compute_c1(sequence: List[float]) -> Tuple[float, float]:
    """Wrapper for compute_c1_fft with list input."""
    return compute_c1_fft(np.array(sequence))

def get_good_direction_to_move_into(
    sequence: List[float],
) -> List[float] | None:
    """Returns the direction to move into the sequence using LP-based optimization."""
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

def generate_extreme_spike_sequence(length: int = 100) -> List[float]:
    """Generate an extremely concentrated spike sequence with maximum values."""
    sequence = [0.0] * length
    
    # Create a spike with maximum possible concentration and value
    # Place very high value at center with minimal support
    center = length // 2
    sequence[center-2] = 50000000.0
    sequence[center-1] = 100000000.0
    sequence[center] = 200000000.0  # Maximum possible value!
    sequence[center+1] = 100000000.0
    sequence[center+2] = 50000000.0
    
    return sequence

def generate_mathematical_spike(length: int = 100) -> List[float]:
    """Generate mathematically optimized spike with extreme concentration."""
    sequence = [0.0] * length
    
    # Create maximum concentration at center
    center = length // 2
    sequence[center-1] = 100000000.0
    sequence[center] = 500000000.0  # Maximum value
    sequence[center+1] = 100000000.0
    
    return sequence

def generate_ultra_concentrated_sequence(length: int = 100) -> List[float]:
    """Generate ultra-concentrated sequence with maximum peak."""
    sequence = [0.0] * length
    
    # Extremely sharp peak with maximum concentration
    center = length // 2
    sequence[center] = 1000000000.0  # Maximum possible value!
    
    return sequence

def generate_optimized_geometric_sequence(base: float = 0.9, length: int = 100) -> List[float]:
    """Generate optimized geometric sequence."""
    sequence = [base**i for i in range(length)]
    total = sum(sequence)
    if total > 0:
        sequence = [x / total for x in sequence]
    return sequence

def generate_peak_concentrated_sequence() -> List[float]:
    """Generate a highly concentrated peak sequence with mathematical precision."""
    sequence = [0.0] * 100
    
    # Create a very sharp and high concentration peak
    sequence[48] = 50000000.0
    sequence[49] = 100000000.0
    sequence[50] = 1000000000.0  # Maximum possible value!
    sequence[51] = 100000000.0
    sequence[52] = 50000000.0
    
    return sequence

def generate_tight_spike_sequence() -> List[float]:
    """Generate a tight spike sequence with minimal spread."""
    sequence = [0.0] * 100
    
    # Create a tight spike with maximum concentration
    sequence[49] = 100000000.0
    sequence[50] = 1000000000.0  # Maximum value!
    sequence[51] = 100000000.0
    
    return sequence

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    start_time = time.time()
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Strategy 1: Try the most extreme mathematical constructions
    extreme_patterns = [
        generate_extreme_spike_sequence(),
        generate_mathematical_spike(),
        generate_ultra_concentrated_sequence(),
        generate_peak_concentrated_sequence(),
        generate_tight_spike_sequence()
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
        max_iterations = 2500  # Even more aggressive iterations
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
        _, inv_c1 = compute_c1(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()
    
    # Strategy 2: Enhanced mathematical patterns with very aggressive optimization
    patterns = [
        # Strong geometric decays
        generate_optimized_geometric_sequence(0.999, 100),
        generate_optimized_geometric_sequence(0.9995, 100),
        generate_optimized_geometric_sequence(0.9999, 100),
        # Harmonic decay
        [1.0/(i+1) for i in range(100)],
        # Uniform distribution
        [1.0/100]*100,
        # Concave-like shape
        [1.0 if i < 20 else 0.0 for i in range(100)],
        # Slight exponential decay with adjustment
        [0.99**i * (1.0 - 0.001*i) for i in range(100)],
        # Another geometric variant with ultra-aggressive decay
        generate_optimized_geometric_sequence(0.99999, 100),
        # Very concentrated peak
        [1.0 if i < 5 else 0.0 for i in range(100)],
        # Smooth bell-shaped with high concentration
        [np.exp(-((i-50)**2)/(2*5**2)) for i in range(100)]
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
        _, inv_c1 = compute_c1(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()
    
    # Strategy 3: Final extremely aggressive refinement with maximum concentration
    if best_sequence is not None and time.time() - start_time < 55:
        # Try the absolute most extreme mathematical construction
        final_sequence = [0.0] * 100
        # Extremely high value peak with perfect concentration
        final_sequence[48] = 100000000.0
        final_sequence[49] = 200000000.0  # Very high peak
        final_sequence[50] = 1000000000.0  # Maximum peak value!
        final_sequence[51] = 200000000.0
        final_sequence[52] = 100000000.0
        
        # Perform extremely aggressive refinement with very high iteration count
        for iteration in range(3000):  # Very aggressive refinement
            if time.time() - start_time > 55:  # Time limit
                break
            h_function = get_good_direction_to_move_into(final_sequence)
            if h_function is not None:
                # Use very aggressive step size for fast convergence
                t = 0.03  # Even more aggressive step size
                final_sequence = [(1 - t) * x + t * y for x, y in zip(final_sequence, h_function)]
            else:
                # If optimization fails, do a small random perturbation with high variance
                final_sequence = [x * (0.9995 + 0.001 * random.random()) for x in final_sequence]
                total = sum(final_sequence)
                if total > 0:
                    final_sequence = [x / total for x in final_sequence]
        
        _, final_inv_c1 = compute_c1(final_sequence)
        if final_inv_c1 > best_inv_c1:
            best_inv_c1 = final_inv_c1
            best_sequence = final_sequence
    
    # If we still haven't found anything good, use a very aggressive geometric sequence
    if best_sequence is None:
        sequence = generate_optimized_geometric_sequence(0.999999, 100)
        best_sequence = sequence
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
