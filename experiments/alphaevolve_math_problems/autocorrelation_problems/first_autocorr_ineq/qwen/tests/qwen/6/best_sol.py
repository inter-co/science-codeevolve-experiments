# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
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
    
    # Use efficient convolution with proper handling
    arr = np.array(sequence)
    
    # Compute linear autoconvolution properly
    autoconv = convolve(arr, arr, mode='full')
    
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
        result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'disp': False, 'presolve': True, 'maxiter': 3000})
        
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
    
    # Mix with original sequence to create new candidate
    t = 0.003  # Even smaller mixing factor for more precise adjustments
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, [x * sum_sequence for x in normalized_g_fun])]
    
    return new_sequence

def mathematical_optimization_approach(max_time_seconds: float = 300) -> List[float]:
    """
    Mathematical optimization approach inspired by Program 2's LP-based approach.
    """
    start_time = time.time()
    
    # Try multiple starting patterns to avoid local optima
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Pattern 1: Golden ratio with different exponents
    n = 100
    phi = (1 + np.sqrt(5)) / 2
    sequence1 = [1000.0 * (phi ** i) / (phi ** 18) for i in range(n)]
    sequence1 = [max(0, val) for val in sequence1]
    
    # Pattern 2: Modified exponential decay with more aggressive decay
    sequence2 = [1000.0 * np.exp(-i/6.0) for i in range(n)]
    sequence2 = [max(0, val) for val in sequence2]
    
    # Pattern 3: Gaussian with sharper peak
    sequence3 = [1000.0 * np.exp(-(i-50)**2/(2*8**2)) for i in range(n)]
    sequence3 = [max(0, val) for val in sequence3]
    
    # Pattern 4: Hybrid approach - mix of geometric and exponential
    sequence4 = []
    for i in range(n):
        if i < n//2:
            sequence4.append(1000.0 * (0.9 ** i))
        else:
            sequence4.append(1000.0 * np.exp(-(i-n//2)/5.0))
    
    # Pattern 5: Modified golden ratio with even more concentrated peak
    sequence5 = [1000.0 * (phi ** i) / (phi ** 15) for i in range(n)]
    sequence5 = [max(0, val) for val in sequence5]
    
    # Pattern 6: Special optimized version with more concentrated peak
    sequence6 = [0.0] * 100
    sequence6[40:60] = [1000.0] * 20  # Even more concentrated peak
    sequence6 = [max(0, val) for val in sequence6]
    
    sequences = [sequence1, sequence2, sequence3, sequence4, sequence5, sequence6]
    
    for i, sequence in enumerate(sequences):
        # Apply iterative improvement using the LP-based approach
        max_iterations = 3500  # More iterations for better convergence
        improvement_count = 0
        
        current_sequence = sequence[:]
        current_inv_c1 = compute_autocorrelation_constant(current_sequence)[1]
        
        for iteration in range(max_iterations):
            if time.time() - start_time > max_time_seconds:
                break
                
            # Try to improve with the LP-based direction
            improved_sequence = get_good_direction_to_move_into(current_sequence)
            
            if improved_sequence is not None:
                _, inv_c1 = compute_autocorrelation_constant(improved_sequence)
                if inv_c1 > current_inv_c1:
                    current_sequence = improved_sequence
                    current_inv_c1 = inv_c1
                    improvement_count += 1
                    # Early stopping if we're making good progress
                    if current_inv_c1 > 0.75:  # Early exit if we get very good results
                        break
                else:
                    # If no improvement, try a different approach
                    if improvement_count == 0 and iteration > 30:
                        # Try to diversify by adding random noise to prevent getting stuck
                        diversified = []
                        for val in current_sequence:
                            if val > 0:
                                noise = random.gauss(0, val * 0.015)  # Even smaller noise
                                diversified.append(max(0, val + noise))
                            else:
                                diversified.append(random.uniform(0, 100))
                        _, inv_c1_div = compute_autocorrelation_constant(diversified)
                        if inv_c1_div > current_inv_c1:
                            current_sequence = diversified
                            current_inv_c1 = inv_c1_div
            else:
                # If LP fails, try alternative approaches
                if iteration % 6 == 0:  # Even more frequent fallback attempts
                    # Try a random perturbation to escape local optima
                    diversified = []
                    for val in current_sequence:
                        if val > 0:
                            noise = random.gauss(0, val * 0.12)  # Even wider noise range
                            diversified.append(max(0, val + noise))
                        else:
                            diversified.append(random.uniform(0, 100))
                    _, inv_c1_div = compute_autocorrelation_constant(diversified)
                    if inv_c1_div > current_inv_c1:
                        current_sequence = diversified
                        current_inv_c1 = inv_c1_div
            
            # Occasionally try a different approach to escape local optima
            if iteration % 30 == 0 and iteration > 0:
                # Try a more diverse sequence
                new_seq = [np.random.uniform(0.5, 2.0) for _ in range(n)]
                _, inv_c1 = compute_autocorrelation_constant(new_seq)
                if inv_c1 > current_inv_c1:
                    current_sequence = new_seq
                    current_inv_c1 = inv_c1
        
        # Final refinement with local optimization
        for _ in range(350):  # Even more aggressive final refinement
            if time.time() - start_time > max_time_seconds:
                break
            improved_sequence = get_good_direction_to_move_into(current_sequence)
            if improved_sequence is not None:
                _, inv_c1 = compute_autocorrelation_constant(improved_sequence)
                if inv_c1 > current_inv_c1:
                    current_sequence = improved_sequence
                    current_inv_c1 = inv_c1
        
        if current_inv_c1 > best_inv_c1:
            best_inv_c1 = current_inv_c1
            best_sequence = current_sequence[:]
    
    # If no good result found, fallback to simple pattern
    if best_sequence is None:
        # Try a simple geometric pattern
        sequence = [1000.0 * (0.9 ** i) for i in range(n)]
        sequence = [max(0, val) for val in sequence]
        best_sequence = sequence
    
    return best_sequence

def create_advanced_mathematical_patterns():
    """Create advanced mathematical patterns based on proven constructions."""
    patterns = []
    
    # Golden ratio - extremely effective
    phi = (1 + np.sqrt(5)) / 2
    golden_pattern = [1000.0 * (phi ** i) / (phi ** 20) for i in range(100)]
    golden_pattern = [max(0, val) for val in golden_pattern]
    patterns.append(("golden", golden_pattern))
    
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
    
    # Power law decay
    power_pattern = [1000.0 * (i+1)**(-0.6) for i in range(100)]
    patterns.append(("power_law", power_pattern))
    
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
    
    # Special high-performing pattern - modified golden ratio
    special_golden = [1000.0 * (phi ** i) / (phi ** 18) for i in range(100)]
    special_golden = [max(0, val) for val in special_golden]
    patterns.append(("special_golden", special_golden))
    
    # Additional mathematical pattern - highly concentrated peak
    peak_pattern = [0.0] * 100
    peak_pattern[45:55] = [1000.0] * 10  # Concentrated peak
    patterns.append(("peak", peak_pattern))
    
    # Additional pattern - alternating high/low with good spacing
    alt_pattern = [1000.0 if i % 3 == 0 else 500.0 if i % 3 == 1 else 250.0 for i in range(100)]
    patterns.append(("alternating_3", alt_pattern))
    
    # New pattern - smooth step with better transition
    step_pattern = []
    for i in range(100):
        if i < 25:
            step_pattern.append(1000.0)
        elif i < 75:
            step_pattern.append(500.0)
        else:
            step_pattern.append(100.0)
    patterns.append(("smooth_step", step_pattern))
    
    # New pattern - double Gaussian with better separation
    double_gaussian = []
    for i in range(100):
        val1 = 500.0 * np.exp(-(i-30)**2/(2*10**2))
        val2 = 500.0 * np.exp(-(i-70)**2/(2*10**2))
        double_gaussian.append(max(0, val1 + val2))
    patterns.append(("double_gaussian", double_gaussian))
    
    # New pattern - cosine with decay and modulation
    cosine_modulated = []
    for i in range(100):
        val = 1000.0 * np.cos(2 * np.pi * i / 15.0) * np.exp(-i/25.0)
        cosine_modulated.append(max(0, val))
    patterns.append(("cosine_modulated", cosine_modulated))
    
    # New pattern - very sharp peak
    sharp_peak = [0.0] * 100
    sharp_peak[48:52] = [1000.0] * 4  # Very narrow peak
    patterns.append(("sharp_peak", sharp_peak))
    
    # New pattern - piecewise linear with sharp transitions
    piecewise = [1000.0 if i < 30 else 500.0 if i < 70 else 100.0 for i in range(100)]
    patterns.append(("piecewise", piecewise))
    
    # New pattern - asymmetric decay
    asym_decay = [1000.0 * np.exp(-i/5.0) if i < 50 else 1000.0 * np.exp(-(i-50)/10.0) for i in range(100)]
    patterns.append(("asym_decay", asym_decay))
    
    # New pattern - oscillating with decay
    osc_decay = [1000.0 * np.exp(-i/12.0) * np.sin(i/3.0) for i in range(100)]
    patterns.append(("osc_decay", osc_decay))
    
    # New pattern - super-concentrated peak
    super_concentrated = [0.0] * 100
    super_concentrated[49:51] = [1000.0] * 2  # Extremely narrow peak
    patterns.append(("super_concentrated", super_concentrated))
    
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
        
        # Add multiple perturbed versions with even more aggressive diversity
        for _ in range(25):  # Even more perturbations for better exploration
            perturbed = []
            for val in sequence:
                if val > 0:
                    # Add Gaussian noise with adaptive intensity
                    intensity = random.uniform(0.003, 0.4)  # Even wider range for more aggressive changes
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
                                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9})
                
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
