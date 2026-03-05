# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
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
    Compute C1 and 1/C1 for a given sequence.
    Uses FFT for efficiency when sequence is large.
    Returns (C1, 1/C1)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure sequence has positive sum
    seq_sum = sum(sequence)
    if seq_sum < 0.01:
        return float('inf'), 0.0
    
    n = len(sequence)
    
    # Use FFT-based convolution for better numerical stability and performance
    try:
        # Use scipy's fftconvolve for better numerical stability
        autoconv = fftconvolve(sequence, sequence, mode='full')
        # The peak of convolution occurs around the center
        max_conv = np.max(autoconv)
        
        # Alternative FFT approach if fftconvolve fails
        if np.isnan(max_conv) or max_conv <= 0:
            # Pad to avoid circular convolution effects
            padded_length = 2 * n - 1
            padded_seq = np.pad(sequence, (0, padded_length - n), mode='constant')
            
            # Compute autoconvolution using FFT
            fft_result = np.fft.fft(padded_seq)
            conv_fft = fft_result * np.conj(fft_result)
            autoconv = np.fft.ifft(conv_fft).real[:n]
            max_conv = np.max(autoconv)
    except:
        # Fallback to standard convolution for very small sequences
        if n <= 100:
            conv = np.convolve(sequence, sequence, mode='full')
            # Extract the auto-correlation values (middle n values)
            auto_corr = conv[n-1:2*n-1]
            max_conv = np.max(auto_corr)
        else:
            # If all else fails, return worst-case
            return float('inf'), 0.0
    
    # Compute C1 = 2n * max(b) / (sum(a))^2
    c1 = (2 * n * max_conv) / (seq_sum ** 2)
    
    # Return both C1 and its reciprocal
    return c1, 1.0 / c1 if c1 > 0 else 0.0

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
    
    # Compute RHS using FFT for better accuracy
    try:
        # Use scipy's fftconvolve for better numerical stability
        autoconv = fftconvolve(normalized_sequence, normalized_sequence, mode='full')
        # The peak of convolution occurs around the center
        rhs = np.max(autoconv)
    except:
        # Fallback to standard computation
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
    
    # Build constraint matrix for convolution using FFT approach
    # This is a more accurate way to construct the constraint matrix
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
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'presolve': True, 'maxiter': 1000})
    except:
        # Fallback to default method
        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, options={'presolve': True, 'maxiter': 1000})
        except:
            return None

    if result.success:
        g_sequence = result.x
        # Ensure all values are non-negative (numerical precision issues)
        g_sequence = np.maximum(g_sequence, 0)
        return g_sequence
    else:
        return None

def create_mathematical_patterns() -> List[List[float]]:
    """Create high-quality mathematical patterns based on known good constructions."""
    patterns = []
    
    # Golden ratio based pattern
    phi = (1 + np.sqrt(5)) / 2
    n = random.randint(30, 150)
    golden = [phi**i for i in range(n)]
    total = sum(golden)
    if total > 0:
        golden = [val * 1000 / total for val in golden]
    patterns.append(golden)
    
    # Fibonacci-like pattern
    n = random.randint(30, 150)
    fib = [1, 1]
    for i in range(n-2):
        fib.append(fib[-1] + fib[-2])
    total = sum(fib)
    if total > 0:
        fib = [val * 1000 / total for val in fib]
    patterns.append(fib)
    
    # Exponential decay pattern (more controlled)
    n = random.randint(30, 150)
    exp_decay = [np.exp(-i/2.5) for i in range(n)]
    total = sum(exp_decay)
    if total > 0:
        exp_decay = [val * 1000 / total for val in exp_decay]
    patterns.append(exp_decay)
    
    # Gaussian-like pattern
    n = random.randint(30, 150)
    gaussian = [np.exp(-((i - n//2)**2) / (2 * (n//5)**2)) for i in range(n)]
    total = sum(gaussian)
    if total > 0:
        gaussian = [val * 1000 / total for val in gaussian]
    patterns.append(gaussian)
    
    # Concentrated mass patterns
    n = random.randint(30, 150)
    concentrated = [0.0] * n
    concentrated[random.randint(0, n//4)] = 1000.0
    concentrated[random.randint(3*n//4, n-1)] = 500.0
    patterns.append(concentrated)
    
    # Two-peak pattern
    n = random.randint(30, 150)
    two_peak = [0.0] * n
    two_peak[n//3] = 800.0
    two_peak[2*n//3] = 600.0
    patterns.append(two_peak)
    
    # Uniform pattern with slight variation
    n = random.randint(30, 150)
    uniform = [1.0 + random.uniform(-0.1, 0.1) for _ in range(n)]
    total = sum(uniform)
    if total > 0:
        uniform = [val * 1000 / total for val in uniform]
    patterns.append(uniform)
    
    # Very concentrated pattern
    n = random.randint(30, 100)
    spike = [0.0] * n
    spike[n//2] = 1000.0
    patterns.append(spike)
    
    # Logarithmic pattern
    n = random.randint(30, 150)
    log_pattern = [np.log(i+2) for i in range(n)]
    total = sum(log_pattern)
    if total > 0:
        log_pattern = [val * 1000 / total for val in log_pattern]
    patterns.append(log_pattern)
    
    # Power law pattern
    n = random.randint(30, 150)
    power_pattern = [1.0 / ((i+1)**1.2) for i in range(n)]
    total = sum(power_pattern)
    if total > 0:
        power_pattern = [val * 1000 / total for val in power_pattern]
    patterns.append(power_pattern)
    
    # Multi-peak pattern with even spacing
    n = random.randint(30, 120)
    multi_peak = [0.0] * n
    peak_positions = [n//5, n//2, 4*n//5]
    for pos in peak_positions:
        if pos < n:
            multi_peak[pos] = 600.0 + random.uniform(0, 400)
    patterns.append(multi_peak)
    
    # Additional mathematical patterns from inspirations
    # Hyperbolic patterns
    n = random.randint(30, 150)
    hyperbolic = [1.0 / (i + 1) for i in range(n)]
    total = sum(hyperbolic)
    if total > 0:
        hyperbolic = [val * 1000 / total for val in hyperbolic]
    patterns.append(hyperbolic)
    
    # Sinusoidal modulation patterns
    n = random.randint(30, 150)
    sin_mod = [np.exp(-i/10.0) * (1 + 0.3 * np.sin(i/5.0)) for i in range(n)]
    total = sum(sin_mod)
    if total > 0:
        sin_mod = [val * 1000 / total for val in sin_mod]
    patterns.append(sin_mod)
    
    # Double exponential patterns
    n = random.randint(30, 150)
    double_exp = [np.exp(-i/3.0) + 0.5*np.exp(-i/6.0) for i in range(n)]
    total = sum(double_exp)
    if total > 0:
        double_exp = [val * 1000 / total for val in double_exp]
    patterns.append(double_exp)
    
    # Heavy-tailed distributions
    n = random.randint(30, 150)
    heavy_tail = [1.0 / ((i+1)**2.5) for i in range(n)]
    total = sum(heavy_tail)
    if total > 0:
        heavy_tail = [val * 1000 / total for val in heavy_tail]
    patterns.append(heavy_tail)
    
    # Modified geometric with optimized scaling
    n = random.randint(30, 150)
    modified_geo = [0.95**i * (1 + 0.1 * np.sin(i/10)) for i in range(n)]
    total = sum(modified_geo)
    if total > 0:
        modified_geo = [val * 1000 / total for val in modified_geo]
    patterns.append(modified_geo)
    
    # Bell-shaped pattern with controlled spread
    n = random.randint(40, 150)
    bell = [np.exp(-((i - n//2)**2) / (2 * (n//4)**2)) * (1 + 0.1*np.sin(i/10.0)) for i in range(n)]
    total = sum(bell)
    if total > 0:
        bell = [val * 1000 / total for val in bell]
    patterns.append(bell)
    
    # Power law with sharper decay
    n = random.randint(30, 150)
    sharp_power = [1.0 / ((i+1)**1.8) * (1 + 0.05*np.cos(i/5.0)) for i in range(n)]
    total = sum(sharp_power)
    if total > 0:
        sharp_power = [val * 1000 / total for val in sharp_power]
    patterns.append(sharp_power)
    
    # Alternating pattern with variation
    n = random.randint(30, 150)
    alternating = [1.0 if i % 2 == 0 else 0.7 for i in range(n)]
    # Add some noise to reduce convolution peaks
    for i in range(0, n, 5):
        if i < n:
            alternating[i] *= random.uniform(0.9, 1.1)
    patterns.append(alternating)
    
    return patterns

def aggressive_local_search(initial_sequence: List[float], max_iter: int = 100) -> List[float]:
    """Apply aggressive local search improvements to a sequence."""
    current_seq = initial_sequence[:]
    current_score = compute_c1(current_seq)[1]  # Get 1/C1
    
    # Track improvements to stop early if no progress
    last_improvement = 0
    consecutive_no_improvement = 0
    
    for iteration in range(max_iter):
        # Make more aggressive random perturbations
        new_seq = current_seq[:]
        
        # Modify more elements for better exploration
        num_changes = max(1, min(len(new_seq) // 6, 25))
        if iteration > 30:
            # Even more exploration in later iterations
            num_changes = max(1, min(len(new_seq) // 3, 40))
            
        for _ in range(num_changes):
            idx = random.randint(0, len(new_seq) - 1)
            # Larger Gaussian perturbation for more aggressive exploration
            perturbation_size = 0.15 if iteration < 30 else 0.25
            new_seq[idx] = max(0.01, new_seq[idx] + random.gauss(0, new_seq[idx] * perturbation_size))
        
        # Try adding/removing elements more systematically
        if random.random() < 0.15 and len(new_seq) > 10:
            new_seq.pop(random.randint(0, len(new_seq) - 1))
        elif random.random() < 0.15 and len(new_seq) < 500:
            new_seq.append(random.uniform(0.01, 1000))
        
        new_score = compute_c1(new_seq)[1]  # Get 1/C1
        if new_score > current_score:
            current_seq = new_seq[:]
            current_score = new_score
            last_improvement = iteration
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
            # More aggressive early stopping
            if consecutive_no_improvement > 30:
                break
    
    return current_seq

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    start_time = time.time()
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Strategy 1: Enhanced mathematical pattern library from inspirations
    patterns = create_mathematical_patterns()
    
    # Test patterns systematically with optimization
    for i, pattern in enumerate(patterns):
        if time.time() - start_time > 45:  # Leave some time for final refinement
            break
            
        # Normalize the pattern
        total = sum(pattern)
        if total > 0:
            sequence = [x / total for x in pattern]
        else:
            continue
        
        # Apply optimization with adaptive iterations
        max_iterations = max(50, 200 - len(pattern)//3)
        for iteration in range(max_iterations):
            if time.time() - start_time > 45:
                break
            h_function = get_good_direction_to_move_into(sequence)
            if h_function is not None:
                sequence = h_function
            else:
                # If optimization fails, do a small random perturbation
                sequence = [x * (0.995 + 0.01 * random.random()) for x in sequence]
                total = sum(sequence)
                if total > 0:
                    sequence = [x / total for x in sequence]
        
        # Check result
        _, inv_c1 = compute_c1(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()
    
    # Strategy 2: Aggressive local search refinement of best found pattern
    if best_sequence is not None and time.time() - start_time < 50.0:
        refined_result = aggressive_local_search(best_sequence, 100)
        _, refined_score = compute_c1(refined_result)
        if refined_score > best_inv_c1:
            best_inv_c1 = refined_score
            best_sequence = refined_result
    
    # Strategy 3: Additional random search for escape
    if time.time() - start_time < 50.0:
        try:
            # Try more targeted random searches with mathematical intuition
            for _ in range(50):
                n = random.randint(30, 200)
                # Create a more structured random sequence
                if random.random() < 0.3:
                    # Geometric pattern with controlled decay
                    rand_seq = [random.uniform(0.01, 1000) * (0.9 ** i) for i in range(n)]
                elif random.random() < 0.6:
                    # Random with some structure
                    rand_seq = [random.uniform(0.01, 1000) for _ in range(n)]
                else:
                    # Concentrated pattern
                    rand_seq = [0.0] * n
                    rand_seq[random.randint(0, n-1)] = 1000.0
                total = sum(rand_seq)
                if total > 0:
                    rand_seq = [x / total for x in rand_seq]
                _, inv_c1 = compute_c1(rand_seq)
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = rand_seq[:]
        except Exception:
            pass
    
    # Strategy 4: Final fine-tuning with more aggressive optimization
    if best_sequence is not None and time.time() - start_time < 55.0:
        try:
            # Try even more aggressive optimization with different step sizes
            final_result = best_sequence[:]
            for iteration in range(100):
                if time.time() - start_time > 55.0:
                    break
                h_function = get_good_direction_to_move_into(final_result)
                if h_function is not None:
                    # Use varying step sizes
                    t = 0.05 + 0.02 * (iteration / 100)  # Gradually increase step size
                    final_result = [(1 - t) * x + t * y for x, y in zip(final_result, h_function)]
                else:
                    # If optimization fails, do a more aggressive random perturbation
                    final_result = [x * (0.99 + 0.02 * random.random()) for x in final_result]
                    total = sum(final_result)
                    if total > 0:
                        final_result = [x / total for x in final_result]
            
            _, final_inv_c1 = compute_c1(final_result)
            if final_inv_c1 > best_inv_c1:
                best_inv_c1 = final_inv_c1
                best_sequence = final_result
        except Exception:
            pass
    
    # If we still haven't found anything good, try to construct a mathematically informed pattern
    if best_sequence is None:
        # Try to construct a pattern that might be optimal based on mathematical theory
        # Use a combination of exponential decay and carefully placed spikes
        n = 100
        sequence = [0.0] * n
        
        # Place some significant weights at strategic positions
        sequence[0] = 1000.0  # Strong initial weight
        sequence[n//2] = 500.0  # Middle weight
        sequence[n-1] = 200.0  # Final weight
        
        # Fill in with exponential decay
        for i in range(1, n-1):
            if i < n//2:
                sequence[i] = 0.9**(i) * 500.0
            else:
                sequence[i] = 0.95**(i - n//2) * 300.0
        
        total = sum(sequence)
        if total > 0:
            sequence = [x / total for x in sequence]
        best_sequence = sequence
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
