# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from typing import List, Tuple
from scipy.signal import fftconvolve
import time
import math
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    Uses FFT for efficiency.
    
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
    
    # Use FFT-based convolution for efficiency
    a = np.array(sequence)
    conv_result = fftconvolve(a, a, mode='full')
    
    # The maximum correlation occurs at the center
    # But we want the maximum over all positions (the true convolution)
    max_conv = np.max(conv_result)
    
    # Calculate C₁ = 2n * max(b) / (sum(a))²
    c1 = (2 * n * max_conv) / (total_sum ** 2)
    
    # Return both C₁ and its reciprocal 1/C₁
    return c1, 1.0 / c1 if c1 > 0 else 0.0

def solve_convolution_lp(f_sequence, rhs):
    """
    Solves the convolution LP for a given sequence and RHS.
    """
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    
    # Build constraint matrix for convolution constraints
    # For each k, we want sum_{i+j=k} f[i]*f[j] <= rhs
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
        # Use highs solver with better settings
        result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', 
                        options={'disp': False, 'presolve': True, 'maxiter': 1000, 'tol': 1e-9})
        if result.success:
            g_sequence = result.x
            return g_sequence
    except Exception:
        # Fallback to simplex method with better parameters
        try:
            result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='simplex',
                            options={'disp': False, 'maxiter': 1000, 'tol': 1e-9})
            if result.success:
                g_sequence = result.x
                return g_sequence
        except Exception:
            pass
    
    return None

def get_good_direction_to_move_into(sequence):
    """
    Returns the direction to move into the sequence based on LP optimization.
    """
    n = len(sequence)
    if n == 0:
        return None
        
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None
        
    # Normalize the sequence appropriately (as done in inspirations)
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
    t = 0.08  # Increased from 0.05 to allow more aggressive moves
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
    
    # Ensure non-negativity
    new_sequence = [max(0, x) for x in new_sequence]
    
    return new_sequence

def generate_advanced_mathematical_pattern(n: int) -> List[float]:
    """
    Generate an advanced mathematical pattern optimized for minimizing convolution peaks.
    """
    sequence = [0.0] * n
    
    # Create a sophisticated pattern that combines:
    # 1. Fast initial decay to concentrate energy early
    # 2. Oscillatory components to distribute energy evenly
    # 3. Careful tapering to prevent large final convolution contributions
    
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        
        # Base exponential decay with controlled rate
        base_decay = math.exp(-2.0 * t)
        
        # Multiple oscillatory components with different frequencies and amplitudes
        osc1 = 0.28 * math.sin(8 * math.pi * t)      # High frequency
        osc2 = 0.22 * math.cos(10 * math.pi * t)     # Medium-high frequency
        osc3 = 0.18 * math.sin(12 * math.pi * t)    # Medium frequency
        osc4 = 0.14 * math.cos(16 * math.pi * t)     # Low-medium frequency
        
        # Additional component for fine-tuning
        extra = 0.08 * math.sin(20 * math.pi * t) * math.cos(6 * math.pi * t)
        
        # Combined amplitude with careful weighting to avoid convolution spikes
        amplitude = 1000 * (base_decay + 0.35 * osc1 + 0.3 * osc2 + 0.25 * osc3 + 0.2 * osc4 + 0.1 * extra)
        
        # Apply smoothing to avoid sharp transitions
        if i > 0 and i < n - 1:
            # Simple averaging with neighbors for smoothing
            smooth_factor = 0.15
            prev_val = sequence[i-1]
            next_val = sequence[i+1] if i+1 < n else 0
            amplitude = (1 - smooth_factor) * amplitude + smooth_factor * (prev_val + next_val) / 2
        
        sequence[i] = max(0, amplitude)
    
    # Normalize to ensure reasonable sum
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    return sequence

def generate_highly_optimized_pattern(n: int) -> List[float]:
    """
    Generate a highly optimized pattern that incorporates mathematical insights
    from multiple inspirations to reduce convolution peaks effectively.
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
            value = 1000 * (0.95 + 0.05 * math.sin(10 * math.pi * i / (n/3)))
        # Middle part: slower decay with oscillation and strategic peaks
        elif i < 2 * n // 3:
            t = (i - n//3) / (n//3)
            base = math.exp(-1.5 * t)
            oscillation = 0.18 * math.sin(12 * math.pi * t) + 0.12 * math.cos(14 * math.pi * t)
            # Add a strategic peak around middle
            if abs(i - n//2) < n//12:
                oscillation += 0.25 * math.sin(24 * math.pi * (i - n//2) / (n//12))
            value = 1000 * (base + 0.25 * oscillation)
        # Late part: tapering with oscillation
        else:
            t = (i - 2*n//3) / (n//3)
            value = 1000 * math.exp(-3.0 * t) * (0.18 + 0.12 * math.sin(8 * math.pi * t))
        
        sequence.append(max(0, value))
    
    # Normalize to ensure good scale
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000 / total for x in sequence]
    
    # Add some additional smoothing to reduce potential convolution spikes
    smoothed = sequence[:]
    for i in range(1, len(sequence)-1):
        smoothed[i] = 0.3 * sequence[i-1] + 0.4 * sequence[i] + 0.3 * sequence[i+1]
    
    # Re-normalize after smoothing
    total_smoothed = sum(smoothed)
    if total_smoothed > 0:
        smoothed = [x * 1000 / total_smoothed for x in smoothed]
    
    return smoothed

def iterative_improvement_with_enhanced_restart(max_iterations: int = 1000) -> List[float]:
    """
    Enhanced iterative improvement with smarter restart strategies and better convergence.
    """
    # Start with the best pattern from our analysis
    n = 200  # Larger size for better optimization potential
    sequence = generate_highly_optimized_pattern(n)
    
    best_sequence = sequence[:]
    best_inv_c1 = compute_autocorrelation_constant(best_sequence)[1]
    
    print(f"Initial score: {best_inv_c1:.6f}")
    
    # Track convergence history for early stopping
    convergence_history = []
    patience = 0
    max_patience = 100  # Increased patience for better convergence
    
    # Apply iterative improvement using the LP-based approach
    for iteration in range(max_iterations):
        # Try to improve with the LP-based direction
        improved_sequence = get_good_direction_to_move_into(best_sequence)
        
        if improved_sequence is not None:
            inv_c1 = compute_autocorrelation_constant(improved_sequence)[1]
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
                
            inv_c1 = compute_autocorrelation_constant(oscillation_pattern)[1]
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
                generate_highly_optimized_pattern(n),
                [1000 * math.exp(-0.02 * i) for i in range(n)],  # Exponential decay
                [1000.0] * n,  # Constant
            ]
            
            for pattern in restart_patterns:
                inv_c1 = compute_autocorrelation_constant(pattern)[1]
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

def search_for_best_sequence() -> List[float]:
    """
    Main search function implementing the enhanced approach that combines best techniques.
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("Starting enhanced hybrid optimization search...")
    
    try:
        # Use the most effective iterative approach
        result = iterative_improvement_with_enhanced_restart(1000)
        final_score = compute_autocorrelation_constant(result)[1]
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
