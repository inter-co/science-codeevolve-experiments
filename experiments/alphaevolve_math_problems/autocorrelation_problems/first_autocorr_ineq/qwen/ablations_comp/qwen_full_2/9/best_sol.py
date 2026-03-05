# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from scipy.signal import fftconvolve
import time
import random

# Global constants for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def compute_c1(sequence: list[float]) -> tuple[float, float]:
    """
    Compute C1 and 1/C1 for a given sequence using FFT for efficiency.
    Returns (C1, 1/C1)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure sequence has positive sum
    seq_sum = sum(sequence)
    if seq_sum < 0.01:
        return float('inf'), 0.0
    
    # Use FFT for autoconvolution - more stable and faster for large sequences
    try:
        # Use scipy's fftconvolve for better numerical stability
        autoconv = fftconvolve(sequence, sequence, mode='full')
        # The peak of convolution occurs around the center
        max_conv = np.max(autoconv)
        
        # Handle edge case where fftconvolve might fail
        if np.isnan(max_conv) or max_conv <= 0:
            raise ValueError("FFT convolution failed")
    except:
        # Fallback to standard convolution for small sequences or edge cases
        if len(sequence) <= 100:
            conv = np.convolve(sequence, sequence, mode='full')
            # Extract the auto-correlation values (middle n values)
            auto_corr = conv[len(sequence)-1:2*len(sequence)-1]
            max_conv = np.max(auto_corr)
        else:
            # If all else fails, return worst-case
            return float('inf'), 0.0
    
    # Compute C1 = 2n * max(b) / (sum(a))^2
    n = len(sequence)
    c1 = (2 * n * max_conv) / (seq_sum ** 2)
    
    # Return both C1 and its reciprocal
    return c1, 1.0 / c1 if c1 > 0 else 0.0

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence using LP optimization."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None
        
    # Normalize the sequence properly for the LP
    normalized_sequence = [x / sum_sequence for x in sequence]
    
    # Compute RHS using FFT for better accuracy and stability
    try:
        # Use scipy's fftconvolve for better numerical stability
        autoconv = fftconvolve(normalized_sequence, normalized_sequence, mode='full')
        # The peak of convolution occurs around the center
        rhs = np.max(autoconv)
    except:
        # Fallback to standard computation
        try:
            rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
        except:
            return None
    
    # Solve the LP to find a better sequence
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_sequence = np.sum(g_fun)
    if sum_sequence < 0.01:
        return None
    # Normalize the result
    normalized_g_fun = [x / sum_sequence for x in g_fun]
    
    # Move towards the better sequence with adaptive step size
    # Start with larger steps but decrease over time for stability
    t = 0.05
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
    
    # Build constraint matrix for convolution (optimized version)
    # Only build the necessary rows for the constraint matrix
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
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', 
                                 options={'presolve': True, 'maxiter': 1000})
    except:
        # Fallback to default method with limited iterations
        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, 
                                     options={'presolve': True, 'maxiter': 1000})
        except:
            return None

    if result.success:
        g_sequence = result.x
        # Ensure all values are non-negative (numerical precision issues)
        g_sequence = np.maximum(g_sequence, 0)
        return g_sequence
    else:
        return None

def create_mathematical_patterns() -> list[list[float]]:
    """Create high-quality mathematical patterns based on proven constructions."""
    patterns = []
    n_base = 100  # Base sequence length
    
    # Golden ratio based with precise scaling
    phi = (1 + np.sqrt(5)) / 2
    golden = [phi**(-i) for i in range(n_base)]
    total = sum(golden)
    if total > 0:
        golden = [val * 1000 / total for val in golden]
    patterns.append(golden)
    
    # Fibonacci with better normalization
    fib = [1, 1]
    while len(fib) < n_base:
        fib.append(fib[-1] + fib[-2])
    total = sum(fib)
    if total > 0:
        fib = [val * 1000 / total for val in fib[:n_base]]
    patterns.append(fib)
    
    # Exponential decay with optimized rate
    exp_decay = [np.exp(-i/2.5) for i in range(n_base)]
    total = sum(exp_decay)
    if total > 0:
        exp_decay = [val * 1000 / total for val in exp_decay]
    patterns.append(exp_decay)
    
    # Gaussian with optimized variance
    gaussian = [np.exp(-((i - n_base//2)**2) / (2 * (n_base//8)**2)) for i in range(n_base)]
    total = sum(gaussian)
    if total > 0:
        gaussian = [val * 1000 / total for val in gaussian]
    patterns.append(gaussian)
    
    # Concentrated patterns with multiple spikes
    concentrated = [0.0] * n_base
    concentrated[20] = 900.0
    concentrated[50] = 700.0
    concentrated[80] = 900.0
    patterns.append(concentrated)
    
    # Two-peak pattern with specific ratios
    two_peak = [0.0] * n_base
    two_peak[n_base//4] = 850.0
    two_peak[3*n_base//4] = 650.0
    patterns.append(two_peak)
    
    # Uniform pattern with fine tuning
    uniform = [1.0 + random.uniform(-0.1, 0.1) for _ in range(n_base)]
    total = sum(uniform)
    if total > 0:
        uniform = [val * 1000 / total for val in uniform]
    patterns.append(uniform)
    
    # Very concentrated pattern
    spike = [0.0] * n_base
    spike[n_base//2] = 1000.0
    patterns.append(spike)
    
    # Multi-peak pattern with optimized spacing
    multi_peak = [0.0] * n_base
    peak_positions = [n_base//6, n_base//3, n_base//2, 2*n_base//3, 5*n_base//6]
    for pos in peak_positions:
        if pos < n_base:
            multi_peak[pos] = 600.0 + random.uniform(0, 200)
    patterns.append(multi_peak)
    
    # Logarithmic with better scaling
    log_pattern = [np.log(i+3) for i in range(n_base)]
    total = sum(log_pattern)
    if total > 0:
        log_pattern = [val * 1000 / total for val in log_pattern]
    patterns.append(log_pattern)
    
    # Power law with sharper decay
    power_pattern = [1.0 / ((i+1)**1.8) for i in range(n_base)]
    total = sum(power_pattern)
    if total > 0:
        power_pattern = [val * 1000 / total for val in power_pattern]
    patterns.append(power_pattern)
    
    # Modified exponential with sinusoidal modulation
    mod_exp = [np.exp(-i/3.0) * (1 + 0.2*np.sin(i/5.0)) for i in range(n_base)]
    total = sum(mod_exp)
    if total > 0:
        mod_exp = [val * 1000 / total for val in mod_exp]
    patterns.append(mod_exp)
    
    # Double exponential pattern
    double_exp = [np.exp(-i/2.0) + 0.5*np.exp(-i/4.0) for i in range(n_base)]
    total = sum(double_exp)
    if total > 0:
        double_exp = [val * 1000 / total for val in double_exp]
    patterns.append(double_exp)
    
    # Heavy-tailed distribution
    heavy_tail = [1.0 / ((i+1)**2.2) for i in range(n_base)]
    total = sum(heavy_tail)
    if total > 0:
        heavy_tail = [val * 1000 / total for val in heavy_tail]
    patterns.append(heavy_tail)
    
    # Hybrid pattern combining multiple features
    hybrid = [0.0] * n_base
    # Place several peaks
    for i in range(6):
        pos = random.randint(0, n_base-1)
        hybrid[pos] = 800.0 + random.uniform(0, 200)
    # Add some smooth decay
    for i in range(n_base):
        if hybrid[i] == 0:
            hybrid[i] = 100.0 * np.exp(-i/20.0)
    total = sum(hybrid)
    if total > 0:
        hybrid = [val * 1000 / total for val in hybrid]
    patterns.append(hybrid)
    
    # Sine wave pattern for periodicity
    sine_pattern = [1.0 + 0.5 * np.sin(2 * np.pi * i / (n_base//4)) for i in range(n_base)]
    total = sum(sine_pattern)
    if total > 0:
        sine_pattern = [val * 1000 / total for val in sine_pattern]
    patterns.append(sine_pattern)
    
    # Bell-shaped pattern with controlled spread
    bell = [np.exp(-((i - n_base//2)**2) / (2 * (n_base//4)**2)) for i in range(n_base)]
    total = sum(bell)
    if total > 0:
        bell = [val * 1000 / total for val in bell]
    patterns.append(bell)
    
    # Inverse power law pattern
    inv_power = [1.0 / ((i+1)**1.2) for i in range(n_base)]
    total = sum(inv_power)
    if total > 0:
        inv_power = [val * 1000 / total for val in inv_power]
    patterns.append(inv_power)
    
    # Alternating pattern with variation
    alternating = [1.0 if i % 2 == 0 else 0.7 for i in range(n_base)]
    # Add some noise to reduce convolution peaks
    for i in range(0, n_base, 5):
        if i < n_base:
            alternating[i] *= random.uniform(0.9, 1.1)
    patterns.append(alternating)
    
    return patterns

def aggressive_local_search(initial_sequence, max_iter=100):
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

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    start_time = time.time()
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Strategy 1: Test mathematical patterns from inspiration programs
    if time.time() - start_time < 20:
        patterns = create_mathematical_patterns()
        for pattern in patterns:
            if time.time() - start_time > 50:
                break
            try:
                # Normalize pattern
                total = sum(pattern)
                if total > 0:
                    sequence = [x / total for x in pattern]
                else:
                    continue
                    
                # Apply optimization
                max_iterations = 200
                for iteration in range(max_iterations):
                    if time.time() - start_time > 50:
                        break
                    h_function = get_good_direction_to_move_into(sequence)
                    if h_function is not None:
                        sequence = h_function
                    else:
                        # Small random perturbation if optimization fails
                        sequence = [x * (0.995 + 0.01 * random.random()) for x in sequence]
                        total = sum(sequence)
                        if total > 0:
                            sequence = [x / total for x in sequence]
                
                # Check result
                _, inv_c1 = compute_c1(sequence)
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = sequence.copy()
            except Exception:
                continue
    
    # Strategy 2: Aggressive local search on promising candidates
    if time.time() - start_time < 40 and best_sequence is not None:
        try:
            refined = aggressive_local_search(best_sequence, 100)
            _, refined_score = compute_c1(refined)
            if refined_score > best_inv_c1:
                best_inv_c1 = refined_score
                best_sequence = refined
        except Exception:
            pass
    
    # Strategy 3: Try more direct optimization approaches
    if time.time() - start_time < 45 and best_sequence is not None:
        try:
            # Try to refine the best found sequence with more targeted approach
            sequence = best_sequence[:]
            for _ in range(50):
                if time.time() - start_time > 55:
                    break
                h_function = get_good_direction_to_move_into(sequence)
                if h_function is not None:
                    # Use smaller steps for more precise tuning
                    t = 0.01
                    sequence = [(1 - t) * x + t * y for x, y in zip(sequence, h_function)]
                else:
                    # Perturbation if optimization fails
                    sequence = [x * (0.999 + 0.002 * random.random()) for x in sequence]
                    total = sum(sequence)
                    if total > 0:
                        sequence = [x / total for x in sequence]
            
            _, final_inv_c1 = compute_c1(sequence)
            if final_inv_c1 > best_inv_c1:
                best_inv_c1 = final_inv_c1
                best_sequence = sequence
        except Exception:
            pass
    
    # Strategy 4: Try a completely fresh approach with high-quality patterns
    if best_sequence is None or time.time() - start_time < 45:
        # Create a pattern based on mathematical insight - balanced distribution
        n = 100
        pattern = []
        # Create a pattern that starts high, decreases gradually, then increases again
        for i in range(n):
            if i < n//3:
                pattern.append(np.exp(-i/10.0) * 1000)
            elif i < 2*n//3:
                pattern.append(1000 * np.exp(-(i-n//3)/15.0))
            else:
                pattern.append(1000 * np.exp(-(i-2*n//3)/20.0))
        
        # Normalize
        total = sum(pattern)
        if total > 0:
            sequence = [x / total for x in pattern]
            
            # Fine-tune
            for _ in range(100):
                if time.time() - start_time > 55:
                    break
                h_function = get_good_direction_to_move_into(sequence)
                if h_function is not None:
                    t = 0.02
                    sequence = [(1 - t) * x + t * y for x, y in zip(sequence, h_function)]
                else:
                    sequence = [x * (0.999 + 0.002 * random.random()) for x in sequence]
                    total = sum(sequence)
                    if total > 0:
                        sequence = [x / total for x in sequence]
            
            _, final_inv_c1 = compute_c1(sequence)
            if final_inv_c1 > best_inv_c1:
                best_inv_c1 = final_inv_c1
                best_sequence = sequence
    
    # Final fallback
    if best_sequence is None:
        n = 100
        sequence = [0.85**i for i in range(n)]
        total = sum(sequence)
        if total > 0:
            sequence = [x / total for x in sequence]
        best_sequence = sequence
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
