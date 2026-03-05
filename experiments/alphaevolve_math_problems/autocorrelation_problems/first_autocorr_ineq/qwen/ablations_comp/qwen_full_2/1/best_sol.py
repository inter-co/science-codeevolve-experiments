# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy import optimize
from scipy.fft import fft, ifft
import time
from typing import List, Tuple

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    C₁ = 2n * max(convolution) / (sum(sequence))²
    Returns (C₁, 1/C₁)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    n = len(sequence)
    sum_seq = sum(sequence)
    
    # Avoid division by very small sums
    if sum_seq < 0.01:
        return float('inf'), 0.0
    
    # Use FFT-based convolution for efficiency
    # Pad to avoid circular convolution effects
    padded_length = 2 * n - 1
    padded_seq = np.pad(sequence, (0, padded_length - n), mode='constant')
    
    # Compute autoconvolution using FFT
    fft_result = fft(padded_seq)
    conv_fft = fft_result * np.conj(fft_result)
    autoconv = ifft(conv_fft).real[:n]
    
    # The convolution should be real, but due to floating point errors, take real part
    max_conv = max(autoconv)
    
    # Calculate C₁
    c1 = (2 * n * max_conv) / (sum_seq ** 2)
    
    # Return both C₁ and its reciprocal
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    return c1, inv_c1

def get_good_direction_to_move_into(
    sequence: List[float],
) -> List[float] | None:
    """Returns the direction to move into the sequence using LP optimization."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None
        
    # Normalize the sequence properly for the LP
    normalized_sequence = [x / sum_sequence for x in sequence]
    
    # Compute the maximum auto-correlation value for the constraint
    conv = np.convolve(normalized_sequence, normalized_sequence, mode='full')
    auto_corr = conv[n-1:2*n-1]
    rhs = np.max(auto_corr)
    
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

def generate_optimized_spike_sequence(length=None):
    """Generate an optimized spike sequence with maximum possible concentration."""
    if length is None:
        length = 100
    
    sequence = [0.0] * length
    
    # Create a perfectly centered spike with maximum possible height
    # This is designed to maximize the ratio of peak convolution to total squared sum
    spike_pos = length // 2
    sequence[spike_pos] = 1000000.0  # Extremely high value
    
    # Make it extremely narrow and sharp - only 1 position around the peak
    for i in range(max(0, spike_pos-1), min(length, spike_pos+2)):
        distance = abs(i - spike_pos)
        if distance == 0:
            sequence[i] = 1000000.0
        else:
            sequence[i] = 1000000.0 * (1 - distance * 0.999)  # Nearly perfect drop-off
    
    return sequence

def generate_highly_concentrated_sequence(length=None):
    """Generate a sequence with maximum concentration and strategic distribution."""
    if length is None:
        length = 100
    
    sequence = [0.0] * length
    
    # Place a very high value at the center
    center = length // 2
    sequence[center] = 500000.0
    
    # Add supporting values around the center with precise ratios
    # This creates a very sharp peak with controlled convolution
    for i in range(max(0, center-2), min(length, center+3)):
        distance = abs(i - center)
        if distance == 0:
            sequence[i] = 500000.0
        elif distance == 1:
            sequence[i] = 300000.0  # Higher than typical
        else:
            sequence[i] = 100000.0  # Lower for stability
    
    return sequence

def generate_precise_geometric_sequence(length=None):
    """Generate a precise geometric sequence optimized for the autocorrelation property."""
    if length is None:
        length = 100
    
    # Use a very carefully chosen geometric base that works well
    # Based on empirical testing of the best bases
    base = 0.87
    sequence = [base**i for i in range(length)]
    
    # Normalize to ensure the sum is meaningful
    total = sum(sequence)
    if total > 0:
        sequence = [x / total for x in sequence]
    
    return sequence

def generate_multispike_sequence(length=None):
    """Generate a sequence with multiple strategically placed spikes."""
    if length is None:
        length = 100
    
    sequence = [0.0] * length
    
    # Place multiple spikes with precise spacing and heights
    num_spikes = 3
    for i in range(num_spikes):
        # Place spikes at roughly evenly spaced locations
        pos = (length // (num_spikes + 1)) * (i + 1)
        sequence[pos] = 100000.0 * (1.0 - i * 0.1)  # Decreasing heights
        
        # Add surrounding support
        for j in range(max(0, pos-1), min(length, pos+2)):
            distance = abs(j - pos)
            if distance == 0:
                sequence[j] = 100000.0 * (1.0 - i * 0.1)
            else:
                sequence[j] = 100000.0 * (1.0 - i * 0.1) * (1 - distance * 0.8)
    
    return sequence

def test_specific_patterns() -> List[float]:
    """Test specific mathematical patterns that have shown to work well"""
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Try the most aggressive spike-based approaches
    spike_strategies = [
        generate_optimized_spike_sequence,
        generate_highly_concentrated_sequence,
        generate_multispike_sequence,
    ]
    
    for strategy in spike_strategies:
        try:
            sequence = strategy()
            _, inv_c1 = compute_autocorrelation_constant(sequence)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = sequence.copy()
        except Exception as e:
            continue
    
    # Also test some mathematical patterns with very high concentration
    patterns = [
        # Extremely concentrated geometric
        [0.87**i * 1000000 for i in range(100)],
        # Very sharp peak
        [0.0] * 48 + [1000000.0] + [0.0] * 52,
        # Multi-peak with high values
        [0.0] * 30 + [500000.0] + [0.0] * 10 + [500000.0] + [0.0] * 10 + [500000.0] + [0.0] * 40,
        # Optimized exponential decay
        [0.88**i * 1000000 for i in range(100)],
        # Super-concentrated Gaussian
        [np.exp(-((i-50)**2)/(2*2**2)) * 1000000 for i in range(100)],
    ]
    
    for i, pattern in enumerate(patterns):
        # Normalize the pattern
        total = sum(pattern)
        if total > 0:
            sequence = [x / total for x in pattern]
        else:
            continue
        
        # Apply focused optimization process with multiple iterations
        max_iterations = 500  # More iterations for better convergence
        for iteration in range(max_iterations):
            h_function = get_good_direction_to_move_into(sequence)
            if h_function is not None:
                sequence = h_function
            else:
                # If optimization fails, do a small random perturbation
                sequence = [x * (0.999 + 0.002 * random.random()) for x in sequence]
                total = sum(sequence)
                if total > 0:
                    sequence = [x / total for x in sequence]
        
        # Check result
        _, inv_c1 = compute_autocorrelation_constant(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()
    
    # Final refinement with even more aggressive optimization
    if best_sequence is not None:
        # Give the best sequence a few extra optimization steps with even smaller steps
        final_sequence = best_sequence.copy()
        # Use even more iterations for final tuning
        for _ in range(300):  # Even more aggressive refinement
            h_function = get_good_direction_to_move_into(final_sequence)
            if h_function is not None:
                # Use a smaller step size for more precise optimization
                t = 0.005
                final_sequence = [(1 - t) * x + t * y for x, y in zip(final_sequence, h_function)]
            else:
                # If optimization fails, do a small random perturbation
                final_sequence = [x * (0.9999 + 0.0002 * random.random()) for x in final_sequence]
                total = sum(final_sequence)
                if total > 0:
                    final_sequence = [x / total for x in final_sequence]
        
        _, final_inv_c1 = compute_autocorrelation_constant(final_sequence)
        if final_inv_c1 > best_inv_c1:
            best_inv_c1 = final_inv_c1
            best_sequence = final_sequence
    
    # If we still haven't found anything good, use a precise geometric decay
    if best_sequence is None:
        n = 100
        sequence = [0.87**i for i in range(n)]
        total = sum(sequence)
        if total > 0:
            sequence = [x / total for x in sequence]
        best_sequence = sequence
    
    return best_sequence

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Use the enhanced pattern-based approach which works better than evolutionary algorithms for this problem
        best_sequence = test_specific_patterns()
        return best_sequence
    except Exception as e:
        print(f"Error in pattern search: {e}")
        # Fallback to precise geometric decay
        n = 100
        sequence = [0.87**i for i in range(n)]
        total = sum(sequence)
        if total > 0:
            sequence = [x / total for x in sequence]
        return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
