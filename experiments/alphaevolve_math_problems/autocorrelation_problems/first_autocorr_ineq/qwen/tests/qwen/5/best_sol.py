# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import random
from typing import List, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    
    Returns:
        tuple: (C₁, 1/C₁) where C₁ = 2n * max(convolution) / (sum(sequence))²
    """
    if not sequence:
        return float('inf'), 0.0
    
    n = len(sequence)
    total_sum = sum(sequence)
    
    # Avoid division by zero
    if total_sum < 1e-10:
        return float('inf'), 0.0
    
    # Use efficient FFT-based convolution for better performance
    arr = np.array(sequence)
    
    # Compute FFT-based autoconvolution
    autoconv = fftconvolve(arr, arr, mode='full')
    
    # The maximum correlation occurs at the center, but we want the maximum over all positions
    max_conv = np.max(autoconv)
    
    # Calculate C₁ = 2n * max(b) / (sum(a))²
    c1 = (2 * n * max_conv) / (total_sum ** 2)
    
    # Return both C₁ and its reciprocal 1/C₁
    return c1, 1.0 / c1 if c1 > 0 else 0.0

def solve_convolution_lp(f_sequence, rhs):
    """
    Solves the convolution LP for a given sequence and RHS.
    This implements the mathematical approach from inspiration programs.
    """
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    
    # Build constraint matrix for convolution constraints
    # The constraint is that for all k: sum_{i+j=k} f[i]*f[j] <= rhs
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
    a_ub_nonneg = -np.eye(n)  # Negative identity matrix for g_i >= 0
    b_ub_nonneg = np.zeros(n)  # Zero vector

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    try:
        from scipy.optimize import linprog
        result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'disp': False, 'presolve': True, 'maxiter': 2000})
        
        if result.success:
            g_sequence = result.x
            return g_sequence
    except Exception as e:
        # Debugging output for LP failures
        # print(f"LP solver failed: {e}")
        pass
    
    return None

def get_good_direction_to_move_into(sequence):
    """
    Returns the direction to move into the sequence based on LP optimization.
    This is inspired by the successful approach from INSPIRATION PROGRAM 2.
    """
    n = len(sequence)
    if n == 0:
        return None
        
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None
        
    # Normalize the sequence to have unit norm
    normalized_sequence = [x / sum_sequence for x in sequence]
    
    # Compute the convolution of the normalized sequence with itself
    conv_result = np.convolve(normalized_sequence, normalized_sequence)
    rhs = np.max(conv_result)
    
    # Solve the LP to find a better direction
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    
    if g_fun is None:
        # If LP fails, try a simple heuristic approach
        # Use a simple gradient approximation or return the original
        return None
        
    # Normalize the resulting sequence to same scale as input
    sum_g = np.sum(g_fun)
    if sum_g < 0.01:
        return None
        
    # We want to return a sequence in the same "direction" as the original
    # Let's make a better sequence that improves the objective
    # We want to keep the same overall structure but make it better
    normalized_g_fun = [x / sum_g for x in g_fun]
    
    # Mix with original sequence to create new candidate - even more precise mixing
    t = 0.005
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, [x * sum_sequence for x in normalized_g_fun])]
    
    return new_sequence

def mathematical_optimization_approach(max_time_seconds: float = 300) -> List[float]:
    """
    Mathematical optimization approach inspired by Program 2's LP-based approach.
    """
    start_time = time.time()
    
    # Start with multiple good mathematical patterns to increase chances of finding better ones
    n = 100
    phi = (1 + np.sqrt(5)) / 2
    
    # Try several different initial patterns
    initial_patterns = [
        # Golden ratio pattern
        [1000.0 * (phi ** i) / (phi ** 20) for i in range(n)],
        # Modified golden ratio with different scaling
        [1000.0 * (phi ** i) / (phi ** 18) for i in range(n)],
        # Exponential decay
        [1000.0 * np.exp(-i/10.0) for i in range(n)],
        # Gaussian-like
        [1000.0 * np.exp(-(i-50)**2/(2*15**2)) for i in range(n)],
        # Sine wave pattern
        [1000.0 * (1 + np.sin(i/3.0)) / 2 for i in range(n)],
        # Power law decay
        [1000.0 * (i+1)**(-0.7) for i in range(n)],
        # Alternating pattern
        [1000.0 if i % 3 == 0 else 500.0 if i % 3 == 1 else 250.0 for i in range(n)],
    ]
    
    # Initialize with the best of these patterns
    best_sequence = None
    best_inv_c1 = 0.0
    
    for pattern in initial_patterns:
        pattern = [max(0, val) for val in pattern]
        _, inv_c1 = compute_autocorrelation_constant(pattern)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = pattern[:]
    
    if best_sequence is None:
        # Fallback to golden ratio if all fail
        best_sequence = [1000.0 * (phi ** i) / (phi ** 20) for i in range(n)]
        best_sequence = [max(0, val) for val in best_sequence]
        best_inv_c1 = compute_autocorrelation_constant(best_sequence)[1]
    
    # Apply iterative improvement using the LP-based approach
    max_iterations = 3000  # Reduced iterations to stay within time limits
    improvement_count = 0
    
    # Track the best sequence found so far for early stopping
    best_so_far_sequence = best_sequence[:]
    best_so_far_inv_c1 = best_inv_c1
    
    # Add additional diversity to help escape local optima
    diversity_attempts = 0
    
    for iteration in range(max_iterations):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Try to improve with the LP-based direction
        improved_sequence = get_good_direction_to_move_into(best_sequence)
        
        if improved_sequence is not None:
            _, inv_c1 = compute_autocorrelation_constant(improved_sequence)
            if inv_c1 > best_inv_c1:
                best_sequence = improved_sequence
                best_inv_c1 = inv_c1
                improvement_count += 1
                # Early stopping if we're making good progress
                if best_inv_c1 > 0.75:  # Early exit if we get very good results
                    break
            else:
                # If no improvement, try a different approach
                if improvement_count == 0 and iteration > 50:
                    # Try to diversify by adding random noise to prevent getting stuck
                    diversified = []
                    for val in best_sequence:
                        if val > 0:
                            noise = random.gauss(0, val * 0.015)  # Even smaller noise for more precise control
                            diversified.append(max(0, val + noise))
                        else:
                            diversified.append(random.uniform(0, 100))
                    _, inv_c1_div = compute_autocorrelation_constant(diversified)
                    if inv_c1_div > best_inv_c1:
                        best_sequence = diversified
                        best_inv_c1 = inv_c1_div
        else:
            # If LP fails, try alternative approaches
            if iteration % 3 == 0:  # Even more frequent fallback attempts
                # Try a random perturbation to escape local optima
                diversified = []
                for val in best_sequence:
                    if val > 0:
                        noise = random.gauss(0, val * 0.1)  # Even wider noise range
                        diversified.append(max(0, val + noise))
                    else:
                        diversified.append(random.uniform(0, 100))
                _, inv_c1_div = compute_autocorrelation_constant(diversified)
                if inv_c1_div > best_inv_c1:
                    best_sequence = diversified
                    best_inv_c1 = inv_c1_div
            else:
                diversity_attempts += 1
                # Periodically try a completely different approach
                if diversity_attempts > 80 and iteration > 100:
                    # Try a completely different pattern
                    new_seq = [np.random.uniform(0.5, 2.0) for _ in range(n)]
                    _, inv_c1 = compute_autocorrelation_constant(new_seq)
                    if inv_c1 > best_inv_c1:
                        best_sequence = new_seq
                        best_inv_c1 = inv_c1
                    diversity_attempts = 0
        
        # Occasionally try a different approach to escape local optima
        if iteration % 30 == 0 and iteration > 0:
            # Try a more diverse sequence
            new_seq = [np.random.uniform(0.5, 2.0) for _ in range(n)]
            _, inv_c1 = compute_autocorrelation_constant(new_seq)
            if inv_c1 > best_inv_c1:
                best_sequence = new_seq
                best_inv_c1 = inv_c1
        
        # Update best so far
        if best_inv_c1 > best_so_far_inv_c1:
            best_so_far_inv_c1 = best_inv_c1
            best_so_far_sequence = best_sequence[:]
    
    # Final refinement with local optimization
    for _ in range(200):  # Fewer iterations for final refinement to save time
        if time.time() - start_time > max_time_seconds:
            break
        improved_sequence = get_good_direction_to_move_into(best_sequence)
        if improved_sequence is not None:
            _, inv_c1 = compute_autocorrelation_constant(improved_sequence)
            if inv_c1 > best_inv_c1:
                best_sequence = improved_sequence
                best_inv_c1 = inv_c1
    
    # Return the best sequence found
    return best_sequence

def create_advanced_mathematical_patterns():
    """Create advanced mathematical patterns based on proven constructions."""
    patterns = []
    
    # Golden ratio - extremely effective
    phi = (1 + np.sqrt(5)) / 2
    golden_pattern = [1000.0 * (phi ** i) / (phi ** 20) for i in range(100)]
    golden_pattern = [max(0, val) for val in golden_pattern]
    patterns.append(("golden", golden_pattern))
    
    # Modified golden ratio with different scaling factors
    special_golden = [1000.0 * (phi ** i) / (phi ** 18) for i in range(100)]
    special_golden = [max(0, val) for val in special_golden]
    patterns.append(("special_golden", special_golden))
    
    # Fibonacci - well documented
    fib = [1, 1]
    for i in range(98):
        fib.append(fib[-1] + fib[-2])
    fib_pattern = [1000.0 * val / max(fib) for val in fib]
    patterns.append(("fibonacci", fib_pattern))
    
    # Exponential decay with optimal rate
    exp_pattern = [1000.0 * np.exp(-i/12.0) for i in range(100)]
    patterns.append(("exponential", exp_pattern))
    
    # Gaussian with peak at center
    gaussian_pattern = [1000.0 * np.exp(-(i-50)**2/(2*12**2)) for i in range(100)]
    patterns.append(("gaussian", gaussian_pattern))
    
    # Smooth hyperbolic tangent transition
    tanh_pattern = [1000.0 * (1 - np.tanh((i-50)/15.0)) for i in range(100)]
    patterns.append(("tanh", tanh_pattern))
    
    # Logarithmic decay
    log_pattern = [1000.0 * np.log(i+2) / np.log(102) for i in range(100)]
    patterns.append(("logarithmic", log_pattern))
    
    # Power law decay with different exponents
    power_pattern = [1000.0 * (i+1)**(-0.6) for i in range(100)]
    patterns.append(("power_law", power_pattern))
    
    # Power law decay with slightly different exponent
    power_pattern2 = [1000.0 * (i+1)**(-0.7) for i in range(100)]
    patterns.append(("power_law_07", power_pattern2))
    
    # Sine wave pattern
    sine_pattern = [1000.0 * (1 + np.sin(i/4.0)) / 2 for i in range(100)]
    patterns.append(("sine", sine_pattern))
    
    # Triangular pattern
    tri_pattern = [1000.0 * (1 - abs(i - 50) / 50.0) for i in range(100)]
    patterns.append(("triangular", tri_pattern))
    
    # Inverse quadratic peak
    inv_quad_pattern = [1000.0 / (1 + (i - 50)**2 / 100.0) for i in range(100)]
    patterns.append(("inverse_quadratic", inv_quad_pattern))
    
    # Mixed structure with multiple peaks
    multi_peak_pattern = []
    for i in range(100):
        if i % 10 == 0:
            multi_peak_pattern.append(1000.0)
        elif i % 5 == 0:
            multi_peak_pattern.append(500.0)
        else:
            multi_peak_pattern.append(100.0)
    patterns.append(("multi_peak", multi_peak_pattern))
    
    # Additional mathematical pattern - highly concentrated peak
    peak_pattern = [0.0] * 100
    peak_pattern[45:55] = [1000.0] * 10  # Concentrated peak
    patterns.append(("peak", peak_pattern))
    
    # Additional pattern - alternating high/low with good spacing
    alt_pattern = [1000.0 if i % 3 == 0 else 500.0 if i % 3 == 1 else 250.0 for i in range(100)]
    patterns.append(("alternating_3", alt_pattern))
    
    # Very sharp peak pattern
    sharp_peak = [0.0] * 100
    sharp_peak[48:52] = [1000.0] * 4  # Very narrow peak
    patterns.append(("sharp_peak", sharp_peak))
    
    # Very narrow peak pattern - even more concentrated
    very_narrow = [0.0] * 100
    very_narrow[49:51] = [1000.0] * 2  # Extremely narrow peak
    patterns.append(("very_narrow", very_narrow))
    
    # Asymmetric decay pattern
    asym_decay = [1000.0 * np.exp(-i/5.0) if i < 50 else 1000.0 * np.exp(-(i-50)/8.0) for i in range(100)]
    patterns.append(("asym_decay", asym_decay))
    
    # Hyperbolic secant pattern
    sech_pattern = [1000.0 * (1 / np.cosh(i/10.0)) for i in range(100)]
    patterns.append(("sech", sech_pattern))
    
    # Oscillating power law
    oscillating_power = [1000.0 * (i+1)**(-0.7) * (1 + 0.1 * np.sin(i/10)) for i in range(100)]
    patterns.append(("oscillating_power", oscillating_power))
    
    # Double exponential pattern
    double_exp = [1000.0 * (np.exp(-i/10.0) + 0.5 * np.exp(-i/20.0)) for i in range(100)]
    patterns.append(("double_exp", double_exp))
    
    # Chebyshev polynomial pattern
    chebyshev_pattern = [1000.0 * np.cos(np.pi * i / 50) * np.exp(-i/25.0) for i in range(100)]
    patterns.append(("chebyshev", chebyshev_pattern))
    
    # Polynomial decay with oscillation
    poly_osc = [1000.0 * (1/(1+i/10)) * np.sin(i/5) for i in range(100)]
    patterns.append(("poly_osc", poly_osc))
    
    # Constant pattern for comparison
    constant_pattern = [1000.0] * 100
    patterns.append(("constant", constant_pattern))
    
    # Linearly decreasing pattern
    linear_decrease = [1000.0 * (1 - i/100.0) for i in range(100)]
    patterns.append(("linear_decrease", linear_decrease))
    
    # Quadratic decay
    quad_decay = [1000.0 * (1 - (i/100.0)**2) for i in range(100)]
    patterns.append(("quad_decay", quad_decay))
    
    return patterns

def advanced_pattern_search_approach(max_time_seconds: float = 300) -> List[float]:
    """
    Advanced pattern search approach that tries many mathematical constructions.
    """
    start_time = time.time()
    
    # Get all mathematical patterns
    patterns = create_advanced_mathematical_patterns()
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Try all patterns
    for name, pattern in patterns:
        if time.time() - start_time > max_time_seconds:
            break
            
        try:
            _, inv_c1 = compute_autocorrelation_constant(pattern)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = pattern[:]
        except Exception:
            continue
    
    # If we have a good pattern, try to refine it
    if best_sequence and time.time() - start_time < max_time_seconds * 0.8:
        refined = local_refinement(best_sequence, max_time_seconds - (time.time() - start_time))
        if refined is not None:
            _, final_inv_c1 = compute_autocorrelation_constant(refined)
            if final_inv_c1 > best_inv_c1:
                return refined
    
    return best_sequence if best_sequence is not None else [1000.0] * 100

def local_refinement(sequence: List[float], max_time: float) -> List[float]:
    """
    Enhanced local refinement with multiple starting points and better optimization.
    """
    def objective(x):
        # Ensure non-negativity
        seq = [max(0.0, val) for val in x]
        c1, inv_c1 = compute_autocorrelation_constant(seq)
        return -inv_c1 if inv_c1 > 0 else 1e10  # Negative because we want to maximize
    
    try:
        # Use L-BFGS-B with bounds
        bounds = [(0.0, 1000.0) for _ in range(len(sequence))]
        
        # Try multiple starting points to avoid local minima
        candidates = [sequence]
        
        # Add multiple perturbed versions with more controlled diversity
        for _ in range(15):  # Fewer perturbations for speed
            perturbed = []
            for val in sequence:
                if val > 0:
                    # Add Gaussian noise with adaptive intensity
                    intensity = random.uniform(0.001, 0.1)  # Smaller range for more careful changes
                    noise = random.gauss(0, intensity * val)
                    perturbed.append(max(0.0, val + noise))
                else:
                    perturbed.append(random.uniform(0, 100))
            candidates.append(perturbed)
        
        best_result = None
        best_score = float('-inf')
        
        for candidate in candidates:
            try:
                # Use a more robust optimization approach with better settings
                result = minimize(objective, candidate, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8})
                
                if result.success:
                    refined = [max(0.0, val) for val in result.x]
                    _, inv_c1 = compute_autocorrelation_constant(refined)
                    if inv_c1 > best_score:
                        best_score = inv_c1
                        best_result = refined
                        
            except Exception:
                continue
                
        return best_result
        
    except Exception:
        return None

def tensor_network_insight_approach() -> List[float]:
    """
    Approach inspired by Program 3's tensor network insights.
    Creates sequences that have mathematical properties related to tensor contractions.
    """
    # Create a sequence based on multiple sinusoidal components with coprime periods
    # This mimics how tensor networks might naturally form optimal structures
    try:
        n = 100
        sequence = []
        
        # Use several prime periods to avoid periodic correlation structures
        periods = [7, 11, 13, 17, 19]  # Prime numbers for good mixing
        
        for i in range(n):
            # Sum multiple sinusoids with different frequencies and amplitudes
            val = 0.0
            for j, period in enumerate(periods):
                amplitude = 1000.0 * (0.8 - 0.1 * j)  # Decreasing amplitudes
                phase = 2 * np.pi * i / period
                val += amplitude * np.sin(phase)
            
            sequence.append(max(0, val))
        
        # Add a small constant to ensure positivity and some structure
        sequence = [val + 100 for val in sequence]
        
        return sequence
    except Exception:
        return [1000.0] * 100

def hybrid_mathematical_search(max_time_seconds: float = 300) -> List[float]:
    """
    Comprehensive hybrid approach combining multiple mathematical strategies.
    """
    start_time = time.time()
    
    # Strategy 1: Mathematical optimization with LP approach (most promising)
    print("Strategy 1: Mathematical optimization with LP approach")
    lp_result = mathematical_optimization_approach(max_time_seconds * 0.5)
    
    # Strategy 2: Advanced pattern search
    print("Strategy 2: Advanced pattern search")
    pattern_result = advanced_pattern_search_approach(max_time_seconds * 0.3)
    
    # Strategy 3: Tensor network inspired approach
    print("Strategy 3: Tensor network insight approach")
    tensor_result = tensor_network_insight_approach()
    
    # Compare results and return the best
    results = [lp_result, pattern_result, tensor_result]
    best_sequence = None
    best_inv_c1 = 0.0
    
    for result in results:
        if result is not None:
            try:
                _, inv_c1 = compute_autocorrelation_constant(result)
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = result
            except Exception:
                continue
    
    # Final refinement if time permits
    if best_sequence and time.time() - start_time < max_time_seconds * 0.9:
        refined = local_refinement(best_sequence, max_time_seconds - (time.time() - start_time))
        if refined is not None:
            _, final_inv_c1 = compute_autocorrelation_constant(refined)
            if final_inv_c1 > best_inv_c1:
                return refined
    
    return best_sequence if best_sequence is not None else [1000.0] * 100

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence using mathematical optimization."""
    try:
        # Set seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        print("Starting comprehensive mathematical optimization approach...")
        # Try the most promising hybrid approach
        best_sequence = hybrid_mathematical_search(max_time_seconds=340)
        
        # Validate and clean up the result
        if best_sequence is None:
            best_sequence = [1000.0] * 100
            
        # Ensure non-empty and positive sum
        if sum(best_sequence) < 0.01:
            best_sequence = [1.0] + best_sequence[1:] if len(best_sequence) > 1 else [1.0]
        
        return best_sequence
        
    except Exception as e:
        # Fallback to simple approach
        print(f"Search failed: {e}")
        return [1000.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
