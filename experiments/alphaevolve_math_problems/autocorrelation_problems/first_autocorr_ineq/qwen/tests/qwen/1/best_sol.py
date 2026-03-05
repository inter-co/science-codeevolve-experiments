# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import fftconvolve
from scipy.optimize import linprog
import time
from typing import List
import math
from itertools import product

# Global constants for optimization
MAX_STEPS = 1000
MIN_STEPS = 20
BOUND_UPPER = 1000.0
BOUND_LOWER = 0.0

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
    # The center element corresponds to index n-1 in the full convolution
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
    if n == 0:
        return None
    
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
        # Use more robust solver settings with tighter tolerances
        result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', 
                        options={'disp': False, 'presolve': True, 'maxiter': 2000, 'tol': 1e-10})
        if result.success:
            g_sequence = result.x
            return g_sequence
    except Exception as e:
        # Fallback to simplex method with better parameters
        try:
            result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='simplex',
                            options={'disp': False, 'maxiter': 2000, 'tol': 1e-10})
            if result.success:
                g_sequence = result.x
                return g_sequence
        except Exception:
            pass
    
    # Try a more direct approach for small problems
    try:
        if n <= 50:  # Only attempt for smaller problems
            # Try with different solver parameters
            result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs-ds', 
                           options={'disp': False, 'presolve': True, 'maxiter': 1000})
            if result.success:
                g_sequence = result.x
                return g_sequence
    except Exception:
        pass
    
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
    try:
        conv_result = np.convolve(normalized_sequence, normalized_sequence)
        rhs = np.max(conv_result)
    except Exception:
        # Fallback to simple estimate if convolution fails
        rhs = 1.0
    
    # Solve the LP to find a better direction
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    
    if g_fun is None:
        return None
        
    # Normalize the resulting sequence
    sum_g = np.sum(g_fun)
    if sum_g < 0.01:
        return None
        
    # Scale back to original magnitude using sqrt(2*n) normalization
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]
    
    # Mix with original sequence to create new candidate
    # Using a slightly more aggressive mixing factor for better exploration
    t = 0.08
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
    
    # Ensure non-negativity
    new_sequence = [max(0, x) for x in new_sequence]
    
    return new_sequence

def generate_advanced_mathematical_pattern(n: int) -> List[float]:
    """
    Generate an advanced mathematical pattern optimized for minimizing convolution peaks
    """
    sequence = [0.0] * n
    
    # Create a sophisticated pattern that combines:
    # 1. Fast initial decay to concentrate energy early
    # 2. Oscillatory components to distribute energy evenly
    # 3. Careful tapering to prevent large final convolution contributions
    
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        
        # Base exponential decay with controlled rate
        base_decay = math.exp(-1.8 * t)
        
        # Multiple oscillatory components with different frequencies and amplitudes
        osc1 = 0.3 * math.sin(8 * math.pi * t)      # High frequency
        osc2 = 0.25 * math.cos(10 * math.pi * t)     # Medium-high frequency
        osc3 = 0.2 * math.sin(12 * math.pi * t)    # Medium frequency
        osc4 = 0.15 * math.cos(16 * math.pi * t)     # Low-medium frequency
        
        # Additional component for fine-tuning
        extra = 0.1 * math.sin(20 * math.pi * t) * math.cos(6 * math.pi * t)
        
        # Combined amplitude with careful weighting to avoid convolution spikes
        amplitude = 1000 * (base_decay + 0.38 * osc1 + 0.32 * osc2 + 0.28 * osc3 + 0.22 * osc4 + 0.12 * extra)
        
        # Apply smoothing to avoid sharp transitions
        if i > 0 and i < n - 1:
            # Simple averaging with neighbors for smoothing
            smooth_factor = 0.18
            prev_val = sequence[i-1]
            next_val = sequence[i+1] if i+1 < n else 0
            amplitude = (1 - smooth_factor) * amplitude + smooth_factor * (prev_val + next_val) / 2
        
        sequence[i] = max(0, amplitude)
    
    # Normalize to ensure reasonable sum
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_fibonacci_like_pattern(n: int) -> List[float]:
    """
    Generate a Fibonacci-like pattern that spreads energy well
    """
    sequence = [0.0] * n
    
    # Create a Fibonacci-like pattern with exponential decay
    fib_prev, fib_curr = 1.0, 1.0
    for i in range(n):
        if i == 0:
            sequence[i] = 1000.0
        elif i == 1:
            sequence[i] = 1000.0
        else:
            # Fibonacci-like growth but with decay
            fib_next = fib_prev + fib_curr
            fib_prev, fib_curr = fib_curr, fib_next
            sequence[i] = max(0, 1000 * fib_curr / (fib_curr + 1000) * math.exp(-0.01 * i))
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_balanced_sequence(n: int) -> List[float]:
    """
    Generate a balanced sequence that spreads energy evenly
    """
    sequence = [0.0] * n
    
    # Create a pattern with gradual decay and oscillation
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        
        # Gradual exponential decay with oscillation to distribute energy
        base = math.exp(-1.8 * t)
        oscillation = 0.22 * math.sin(6 * math.pi * t) + 0.18 * math.cos(8 * math.pi * t)
        amplitude = 1000 * (base + 0.32 * oscillation)
        
        sequence[i] = max(0, amplitude)
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_peak_and_trough_sequence(n: int) -> List[float]:
    """
    Generate a sequence with strategic peak and trough placement
    """
    sequence = [0.0] * n
    
    # Place a strong peak in the middle with controlled decay
    peak_pos = n // 2
    peak_height = 1000.0
    
    for i in range(n):
        distance_from_peak = abs(i - peak_pos)
        
        # Exponential decay with oscillation around peak
        if distance_from_peak == 0:
            amplitude = peak_height
        else:
            # Exponential decay with oscillation to avoid convolution spikes
            base_decay = math.exp(-distance_from_peak / (n / 6))
            oscillation = 0.12 * math.sin(5 * math.pi * distance_from_peak / n)
            amplitude = peak_height * base_decay * (0.85 + 0.15 * oscillation)
        
        sequence[i] = max(0, amplitude)
    
    # Normalize
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
        freq1 = math.sin(0.6 * math.pi * i / n)
        freq2 = math.cos(0.9 * math.pi * i / n)
        freq3 = math.sin(1.3 * math.pi * i / n) * math.cos(0.7 * math.pi * i / n)
        
        # Combine with exponential decay to focus energy early
        decay = math.exp(-0.015 * i)
        amplitude = 1000 * (0.75 * decay + 0.22 * freq1 + 0.18 * freq2 + 0.08 * freq3)
        sequence.append(max(0, amplitude))
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_bell_curve_sequence(n: int) -> List[float]:
    """
    Generate a bell curve pattern that concentrates energy in the center
    """
    sequence = [0.0] * n
    
    # Create a bell curve centered in the middle
    center = n // 2
    spread = n / 4  # Controls width of the bell curve
    
    for i in range(n):
        # Gaussian-like shape
        exponent = -((i - center) ** 2) / (2 * spread ** 2)
        amplitude = 1000 * math.exp(exponent)
        sequence[i] = max(0, amplitude)
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_sine_wave_sequence(n: int) -> List[float]:
    """
    Generate a sine wave pattern that can help reduce convolution peaks
    """
    sequence = []
    for i in range(n):
        # Create a pattern with multiple sine waves
        sine1 = 0.55 * math.sin(0.6 * math.pi * i / n)
        sine2 = 0.35 * math.sin(1.3 * math.pi * i / n)
        sine3 = 0.25 * math.sin(2.2 * math.pi * i / n)
        
        # Combine with exponential decay to focus energy early
        decay = math.exp(-0.012 * i)
        amplitude = 1000 * (0.72 * decay + 0.24 * sine1 + 0.19 * sine2 + 0.08 * sine3)
        sequence.append(max(0, amplitude))
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_piecewise_linear_sequence(n: int) -> List[float]:
    """
    Generate a piecewise linear sequence that can help reduce convolution peaks
    """
    sequence = [0.0] * n
    
    # Create a piecewise linear pattern
    mid = n // 2
    for i in range(n):
        if i <= mid:
            # Increasing phase
            sequence[i] = 1000 * (i / mid)
        else:
            # Decreasing phase
            sequence[i] = 1000 * ((n - i) / (n - mid))
    
    # Add some oscillation to avoid convolution spikes
    for i in range(n):
        oscillation = 0.06 * math.sin(2.5 * math.pi * i / n)
        sequence[i] = max(0, sequence[i] * (1 + oscillation))
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_optimized_sidon_sequence(n: int) -> List[float]:
    """
    Generate a sequence inspired by Sidon set constructions
    These typically have good properties for minimizing convolution peaks
    """
    sequence = [0.0] * n
    
    # Use a pattern that mimics known good constructions
    # This uses a combination of geometric decay and periodic components
    for i in range(n):
        # Geometric decay component
        geometric = 1000 * math.exp(-0.025 * i)
        
        # Periodic component to avoid convolution spikes
        periodic = 120 * math.sin(0.6 * math.pi * i / (n/4)) * math.cos(0.4 * math.pi * i / (n/4))
        
        # Add a small constant to ensure all values are positive
        amplitude = geometric + periodic + 60
        
        sequence[i] = max(0, amplitude)
    
    # Normalize
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_alternating_pattern(n: int) -> List[float]:
    """
    Generate an alternating pattern with strategic values
    """
    sequence = []
    for i in range(n):
        if i % 4 == 0:
            sequence.append(1000.0)
        elif i % 4 == 1:
            sequence.append(800.0)
        elif i % 4 == 2:
            sequence.append(600.0)
        else:
            sequence.append(400.0)
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    return sequence

def generate_exponential_pattern(n: int) -> List[float]:
    """
    Generate an exponential decay pattern with optimized rate
    """
    sequence = []
    for i in range(n):
        value = 1000 * math.exp(-0.025 * i)
        sequence.append(max(0, value))
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    return sequence

def generate_highly_optimized_pattern(n: int) -> List[float]:
    """
    Generate a highly optimized pattern that incorporates mathematical insights
    from multiple inspirations to reduce convolution peaks effectively
    """
    # Pattern based on mathematical analysis that works well for this problem
    sequence = []
    
    # Create a pattern that combines:
    # 1. Initial high values to concentrate early energy
    # 2. Controlled decay to reduce later convolution contributions
    # 3. Oscillations to spread energy evenly without creating spikes
    # 4. Strategic peaks to balance energy distribution
    
    for i in range(n):
        # Early part: rapid decay with oscillation
        if i < n // 3:
            # Quick initial decay with oscillation
            value = 1000 * (0.92 + 0.08 * math.sin(12 * math.pi * i / (n/3)))
        # Middle part: slower decay with oscillation and strategic peaks
        elif i < 2 * n // 3:
            t = (i - n//3) / (n//3)
            base = math.exp(-1.6 * t)
            oscillation = 0.2 * math.sin(14 * math.pi * t) + 0.15 * math.cos(16 * math.pi * t)
            # Add a strategic peak around middle
            if abs(i - n//2) < n//10:
                oscillation += 0.3 * math.sin(28 * math.pi * (i - n//2) / (n//10))
            value = 1000 * (base + 0.28 * oscillation)
        # Late part: tapering with oscillation
        else:
            t = (i - 2*n//3) / (n//3)
            value = 1000 * math.exp(-3.2 * t) * (0.2 + 0.15 * math.sin(10 * math.pi * t))
        
        sequence.append(max(0, value))
    
    # Normalize to ensure good scale
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    # Add some additional smoothing to reduce potential convolution spikes
    smoothed = sequence[:]
    for i in range(1, len(sequence)-1):
        # Weighted average with more emphasis on center
        smoothed[i] = 0.25 * sequence[i-1] + 0.5 * sequence[i] + 0.25 * sequence[i+1]
    
    # Re-normalize after smoothing
    total_smoothed = sum(smoothed)
    if total_smoothed > 0:
        smoothed = [x * 1000 / total_smoothed for x in smoothed]
    
    return smoothed

def advanced_local_search(initial_sequence: List[float], max_iter: int = 2000) -> List[float]:
    """
    Enhanced local search with better neighborhood exploration and convergence
    """
    current_seq = initial_sequence.copy()
    current_score = compute_autocorrelation_constant(current_seq)
    
    # Track improvement for early stopping
    last_improvement = 0
    consecutive_no_improvement = 0
    max_consecutive_no_improvement = 250
    
    # More diverse neighborhood operators with better weights
    operators = [
        ("small_changes", 0.15),      # Small random changes for fine tuning
        ("moderate_changes", 0.2),    # Moderate changes for larger adjustments
        ("scaling", 0.12),            # Global scaling
        ("strategic_adjustments", 0.13), # Strategic position adjustments
        ("segment_shift", 0.12),      # Segment shifts
        ("adaptive_changes", 0.15),   # Adaptive changes
        ("random_segment", 0.12),     # Random segments
        ("pattern_based", 0.15),      # Pattern-based mutations
        ("gaussian_noise", 0.1),      # Gaussian noise for fine tuning
        ("neighborhood_shift", 0.1),  # Shift neighboring elements
        ("peak_reduction", 0.08),     # Targeted peak reduction
        ("symmetry_preservation", 0.08), # Preserve symmetry properties
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
                    num_changes = min(12, len(neighbor) // 6)
                    for _ in range(num_changes):
                        idx = random.randint(0, len(neighbor) - 1)
                        change = random.gauss(0, 10)
                        neighbor[idx] = max(0, neighbor[idx] + change)
                        
                elif name == "moderate_changes":
                    # Moderate changes for larger adjustments
                    num_changes = min(6, len(neighbor) // 10)
                    for _ in range(num_changes):
                        idx = random.randint(0, len(neighbor) - 1)
                        change = random.gauss(0, 35)
                        neighbor[idx] = max(0, neighbor[idx] + change)
                        
                elif name == "scaling":
                    # Global scaling
                    scale_factor = random.uniform(0.95, 1.05)
                    neighbor = [max(0, x * scale_factor) for x in neighbor]
                    
                elif name == "strategic_adjustments":
                    # Adjust some strategic positions
                    positions_to_adjust = random.sample(range(len(neighbor)), min(5, len(neighbor) // 10))
                    for idx in positions_to_adjust:
                        neighbor[idx] = max(0, neighbor[idx] + random.gauss(0, 22))
                        
                elif name == "segment_shift":
                    # Apply a shift to a segment
                    start_idx = random.randint(0, len(neighbor) - 5)
                    end_idx = min(start_idx + 5, len(neighbor))
                    shift_amount = random.gauss(0, 18)
                    for i in range(start_idx, end_idx):
                        neighbor[i] = max(0, neighbor[i] + shift_amount)
                        
                elif name == "adaptive_changes":
                    # Adaptive changes based on current values
                    for i in range(len(neighbor)):
                        if neighbor[i] > 500:
                            change = random.gauss(0, 10)
                        else:
                            change = random.gauss(0, 22)
                        neighbor[i] = max(0, neighbor[i] + change)
                        
                elif name == "random_segment":
                    # Randomized segment adjustments
                    segment_size = max(1, len(neighbor) // 15)
                    start_idx = random.randint(0, len(neighbor) - segment_size)
                    for i in range(start_idx, min(start_idx + segment_size, len(neighbor))):
                        change = random.gauss(0, 18)
                        neighbor[i] = max(0, neighbor[i] + change)
                        
                elif name == "pattern_based":
                    # Pattern-based mutations with enhanced structure awareness
                    for i in range(len(neighbor)):
                        # If at beginning or end, make more significant changes
                        if i < 3 or i > len(neighbor) - 3:
                            change = random.gauss(0, 32)
                        else:
                            change = random.gauss(0, 10)
                        neighbor[i] = max(0, neighbor[i] + change)
                        
                elif name == "gaussian_noise":
                    # Add Gaussian noise to all elements
                    for i in range(len(neighbor)):
                        noise = random.gauss(0, 12)
                        neighbor[i] = max(0, neighbor[i] + noise)
                        
                elif name == "neighborhood_shift":
                    # Shift elements in neighborhood to preserve structure
                    idx = random.randint(0, len(neighbor) - 1)
                    shift_amount = random.gauss(0, 18)
                    # Shift the element and its neighbors
                    for i in range(max(0, idx-1), min(len(neighbor), idx+2)):
                        neighbor[i] = max(0, neighbor[i] + shift_amount * (1.0 - abs(i-idx)/2.0))
                
                elif name == "peak_reduction":
                    # Specifically target to reduce peak values
                    if len(neighbor) > 10:
                        # Reduce the maximum value in the sequence
                        max_idx = np.argmax(neighbor)
                        neighbor[max_idx] = max(0, neighbor[max_idx] * 0.85)
                        # Also adjust nearby values to smooth out
                        for j in range(max(max_idx-2, 0), min(max_idx+3, len(neighbor))):
                            if j != max_idx:
                                neighbor[j] = max(0, neighbor[j] * 0.92)
                
                elif name == "symmetry_preservation":
                    # Preserve some symmetry properties
                    if len(neighbor) >= 10:
                        # Mirror some values to preserve symmetry
                        mid = len(neighbor) // 2
                        for i in range(mid):
                            if i < len(neighbor) - i - 1:
                                avg_val = (neighbor[i] + neighbor[len(neighbor) - 1 - i]) / 2
                                neighbor[i] = avg_val
                                neighbor[len(neighbor) - 1 - i] = avg_val
                
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
            if consecutive_no_improvement > 15 and random.random() < 0.04:
                current_seq = best_neighbor
                current_score = best_score
                last_improvement = iteration
                consecutive_no_improvement = 0
        
        # Early stopping if no improvement for a while
        if iteration - last_improvement > max_consecutive_no_improvement:
            break
    
    return current_seq

def iterative_improvement_with_enhanced_restart(max_iterations: int = 1500) -> List[float]:
    """
    Enhanced iterative improvement with smarter restart strategies and better convergence
    """
    # Start with the best pattern from our analysis
    n = 200  # Larger size for better optimization potential
    sequence = generate_highly_optimized_pattern(n)
    
    best_sequence = sequence[:]
    best_inv_c1 = compute_autocorrelation_constant(best_sequence)
    
    print(f"Initial score: {best_inv_c1:.6f}")
    
    # Track convergence history for early stopping
    convergence_history = []
    patience = 0
    max_patience = 80
    
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
        if len(convergence_history) >= 15:
            recent_scores = convergence_history[-15:]
            # Check if scores are essentially the same (within small tolerance)
            if len(set([round(s, 6) for s in recent_scores])) == 1:
                print(f"Converged after {iteration} iterations")
                break
                
        # Occasionally try a different approach to escape local optima
        if iteration % 40 == 0 and iteration > 0:
            # Try a pattern with more oscillation for diversity
            oscillation_pattern = []
            for i in range(n):
                # More aggressive oscillation
                value = 1000 * (0.8 + 0.2 * math.sin(15 * math.pi * i / n))
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
        if iteration % 150 == 0 and iteration > 0:
            # Try different pattern types with better variety
            restart_patterns = [
                generate_advanced_mathematical_pattern(n),
                generate_fibonacci_like_pattern(n),
                generate_balanced_sequence(n),
                generate_peak_and_trough_sequence(n),
                generate_oscillating_pattern(n),
                generate_bell_curve_sequence(n),
                generate_sine_wave_sequence(n),
                generate_piecewise_linear_sequence(n),
                generate_optimized_sidon_sequence(n),
                generate_alternating_pattern(n),
                generate_exponential_pattern(n),
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

def multi_strategy_search() -> List[float]:
    """
    Multi-strategy approach that tries several different methods and picks the best
    """
    print("Multi-strategy search initiated")
    
    # Strategy 1: Highly optimized pattern + iterative improvement
    print("Strategy 1: Highly optimized pattern with iterative improvement")
    try:
        result1 = iterative_improvement_with_enhanced_restart(1000)
        score1 = compute_autocorrelation_constant(result1)
        print(f"Strategy 1 score: {score1:.6f}")
    except Exception as e:
        print(f"Strategy 1 failed: {e}")
        score1 = 0
        result1 = None
    
    # Strategy 2: Direct local search from multiple starting points
    print("Strategy 2: Direct local search from multiple starting points")
    try:
        candidates = [
            generate_advanced_mathematical_pattern(200),
            generate_fibonacci_like_pattern(200),
            generate_balanced_sequence(200),
            generate_peak_and_trough_sequence(200),
            generate_oscillating_pattern(200),
            generate_bell_curve_sequence(200),
            generate_sine_wave_sequence(200),
            generate_piecewise_linear_sequence(200),
            generate_optimized_sidon_sequence(200),
            generate_alternating_pattern(200),
            generate_exponential_pattern(200),
            [1000 * math.exp(-0.02 * i) for i in range(200)],
            [1000.0] * 200
        ]
        
        best_candidate = None
        best_score = 0
        
        for i, candidate in enumerate(candidates):
            refined = advanced_local_search(candidate, 1000)
            score = compute_autocorrelation_constant(refined)
            print(f"Direct search candidate {i+1} score: {score:.6f}")
            if score > best_score:
                best_score = score
                best_candidate = refined
                
        score2 = best_score
        result2 = best_candidate
        print(f"Strategy 2 best score: {score2:.6f}")
    except Exception as e:
        print(f"Strategy 2 failed: {e}")
        score2 = 0
        result2 = None
    
    # Strategy 3: Enhanced pattern-based approach
    print("Strategy 3: Enhanced pattern-based approach")
    try:
        # Try the best patterns individually with local search
        patterns_to_try = [
            generate_highly_optimized_pattern(200),
            generate_advanced_mathematical_pattern(200),
            generate_fibonacci_like_pattern(200),
            generate_balanced_sequence(200),
            generate_peak_and_trough_sequence(200),
        ]
        
        best_pattern = None
        best_pattern_score = 0
        
        for pattern in patterns_to_try:
            # Apply advanced local search
            refined = advanced_local_search(pattern, 800)
            score = compute_autocorrelation_constant(refined)
            print(f"Pattern-based refinement score: {score:.6f}")
            if score > best_pattern_score:
                best_pattern_score = score
                best_pattern = refined
                
        score3 = best_pattern_score
        result3 = best_pattern
        print(f"Strategy 3 best score: {score3:.6f}")
    except Exception as e:
        print(f"Strategy 3 failed: {e}")
        score3 = 0
        result3 = None
    
    # Strategy 4: Systematic search through many patterns
    print("Strategy 4: Systematic search through many patterns")
    try:
        # Try a broader range of patterns with more focused local search
        systematic_patterns = [
            generate_highly_optimized_pattern(150),
            generate_advanced_mathematical_pattern(150),
            generate_fibonacci_like_pattern(150),
            generate_balanced_sequence(150),
            generate_peak_and_trough_sequence(150),
            generate_oscillating_pattern(150),
            generate_bell_curve_sequence(150),
            generate_sine_wave_sequence(150),
            generate_piecewise_linear_sequence(150),
            generate_optimized_sidon_sequence(150),
        ]
        
        best_systematic = None
        best_systematic_score = 0
        
        for i, pattern in enumerate(systematic_patterns):
            refined = advanced_local_search(pattern, 600)
            score = compute_autocorrelation_constant(refined)
            print(f"Systematic pattern {i+1} score: {score:.6f}")
            if score > best_systematic_score:
                best_systematic_score = score
                best_systematic = refined
                
        score4 = best_systematic_score
        result4 = best_systematic
        print(f"Strategy 4 best score: {score4:.6f}")
    except Exception as e:
        print(f"Strategy 4 failed: {e}")
        score4 = 0
        result4 = None
    
    # Select the best result from all strategies
    results = [(result1, score1), (result2, score2), (result3, score3), (result4, score4)]
    valid_results = [(r, s) for r, s in results if r is not None and s > 0]
    
    if valid_results:
        best_result = max(valid_results, key=lambda x: x[1])
        print(f"Best overall result score: {best_result[1]:.6f}")
        return best_result[0]
    else:
        # Fallback to highly optimized pattern
        print("Falling back to highly optimized pattern")
        return generate_highly_optimized_pattern(200)

def search_for_best_sequence() -> List[float]:
    """
    Main search function implementing the enhanced approach that combines best techniques
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("Starting enhanced hybrid optimization search...")
    
    try:
        # Use the most effective multi-strategy approach
        result = multi_strategy_search()
        final_score = compute_autocorrelation_constant(result)
        print(f"Final result score: {final_score:.6f}")
        return result
    except Exception as e:
        print(f"Search failed: {e}")
        # Fallback to a robust mathematical pattern
        return generate_highly_optimized_pattern(200)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
