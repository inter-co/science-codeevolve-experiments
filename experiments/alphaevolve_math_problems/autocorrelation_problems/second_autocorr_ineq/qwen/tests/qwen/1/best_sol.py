# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
from scipy.optimize import minimize, differential_evolution
import warnings
import random
import time
from typing import List, Tuple
import numba
from numba import jit

@jit(nopython=True)
def fast_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Fast computation of norms using numba JIT compilation.
    """
    if len(f_values) == 0:
        return 0.0, 0.0, 0.0
    
    n = len(f_values)
    # Precompute autoconvolution manually for speed
    g = np.zeros(2 * n - 1)
    
    # Manual autoconvolution computation (f * f)
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    # Compute norms using evaluator's approach
    g_abs = np.abs(g)
    
    # ||g||₂² using evaluator's approach: sum of squares directly
    g_l2_sq = np.sum(g_abs**2)
    
    # ||g||₁ using evaluator's approach: sum of absolute values divided by (len(g)+1)
    g_l1 = np.sum(g_abs) / (len(g) + 1)
    
    # ||g||∞ 
    g_linf = np.max(g_abs)
    
    return g_l2_sq, g_l1, g_linf

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation using the correct integration method.
    """
    if len(f_values) == 0:
        return 0.0, 0.0, 0.0
    
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using numpy's convolve
    g_full = convolve(f, f, mode='full')
    
    # For domain [-1/4, 1/4], we want the central portion of the convolution
    # Result of convolution of length n arrays is length 2*n-1
    center_idx = len(g_full) // 2
    g_start = center_idx - len(f) + 1
    g_end = center_idx + len(f)
    g = g_full[g_start:g_end]
    
    # Compute ||g||₂² using the correct trapezoidal-like integration method:
    # For interval with heights y₁, y₂ and width h, contribution is (h/3)(y₁² + y₁y₂ + y₂²)
    n_g = len(g)
    dx = 0.5 / (len(f) - 1) if len(f) > 1 else 0.5
    
    if n_g <= 1:
        norm_g2_squared = 0.0
    else:
        norm_g2_squared = 0.0
        for i in range(n_g - 1):
            y1 = g[i]
            y2 = g[i+1]
            norm_g2_squared += (dx/3) * (y1**2 + y1*y2 + y2**2)
    
    # ||g||₁ = sum(|g|) / (len(g) + 1) as specified in the prompt
    norm_g1 = np.sum(np.abs(g)) / (len(g) + 1)
    
    # ||g||∞ = max(|g|)
    norm_ginf = np.max(np.abs(g))
    
    return norm_g2_squared, norm_g1, norm_ginf

def compute_c2(f_values: List[float]) -> float:
    """
    Compute C2 = ||g||₂² / (||g||₁ · ||g||∞) where g = f*f
    """
    try:
        g_l2_sq, g_l1, g_linf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_l1 <= 1e-15 or g_linf <= 1e-15:
            return 0.0
            
        c2 = g_l2_sq / (g_l1 * g_linf)
        return c2
    except Exception as e:
        warnings.warn(f"Error computing C2: {e}")
        return 0.0

def create_advanced_patterns(n: int) -> List[Tuple[str, List[float]]]:
    """
    Create mathematically-inspired patterns with proven effectiveness.
    Based on best patterns from inspirations with improvements.
    """
    patterns = []
    
    # Pattern 1: Optimized sharp tent with better normalization
    tent = []
    for i in range(n):
        dist_from_center = abs(i - (n-1)/2) / ((n-1)/2)
        # Sharper edges with better control
        val = max(0, 1.0 - 6.0 * dist_from_center)
        tent.append(val)
    total = sum(tent)
    if total > 0:
        tent = [x * 2.0 / total for x in tent]
    patterns.append(("sharp_tent", tent))
    
    # Pattern 2: Multi-frequency pattern inspired by signal processing
    multi_freq = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Mix of multiple frequencies for rich convolution properties
        val = 0.5 + 0.3 * np.cos(6 * np.pi * x) + 0.2 * np.sin(10 * np.pi * x) \
              + 0.15 * np.cos(14 * np.pi * x) + 0.05 * np.sin(18 * np.pi * x)
        multi_freq.append(max(0.0, val))
    total = sum(multi_freq)
    if total > 0:
        multi_freq = [x * 2.0 / total for x in multi_freq]
    patterns.append(("multi_freq", multi_freq))
    
    # Pattern 3: Double peak with careful spacing
    double_peak = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Two peaks with strategic placement and width
        val = (0.6 * np.exp(-((x - 0.25)**2) * 10) + 
               0.6 * np.exp(-((x + 0.25)**2) * 10)) * (1 + 0.1 * np.cos(12 * np.pi * x))
        double_peak.append(max(0.0, val))
    total = sum(double_peak)
    if total > 0:
        double_peak = [x * 2.0 / total for x in double_peak]
    patterns.append(("double_peak", double_peak))
    
    # Pattern 4: Oscillating pattern with high frequency
    high_osc = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # High frequency oscillation with amplitude modulation
        val = 0.7 + 0.2 * np.sin(25 * np.pi * x) + 0.15 * np.cos(35 * np.pi * x)
        high_osc.append(max(0.0, val))
    total = sum(high_osc)
    if total > 0:
        high_osc = [x * 2.0 / total for x in high_osc]
    patterns.append(("high_osc", high_osc))
    
    # Pattern 5: Asymmetric pattern with strategic asymmetry
    asymmetric = []
    for i in range(n):
        pos = i / (n - 1) if n > 1 else 0.5
        # Create asymmetric shape with steeper rise on left side
        if pos < 0.25:
            val = 1.0 - 5.0 * pos  # Steeper drop
        elif pos > 0.75:
            val = 1.0 - 2.0 * (1 - pos)  # Gentler rise
        else:
            # Smooth transition in center with slight oscillation
            val = 0.3 + 0.4 * np.cos(10 * np.pi * pos) + 0.1 * np.sin(20 * np.pi * pos)
        asymmetric.append(max(0.0, val))
    
    total = sum(asymmetric)
    if total > 0:
        asymmetric = [x * 2.0 / total for x in asymmetric]
    patterns.append(("asymmetric", asymmetric))
    
    # Pattern 6: Exponential decay with oscillation
    exp_osc = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Center peak with exponential decay and oscillation
        val = 1.2 * np.exp(-x**2 * 15) * (1 + 0.15 * np.cos(12 * np.pi * x))
        exp_osc.append(max(0.0, val))
    total = sum(exp_osc)
    if total > 0:
        exp_osc = [x * 2.0 / total for x in exp_osc]
    patterns.append(("exp_osc", exp_osc))
    
    return patterns

def enhanced_local_search(initial_solution: List[float], max_iter: int = 150) -> List[float]:
    """
    Enhanced local search with better exploration and adaptive strategies.
    More aggressive for better results while respecting time limits.
    """
    current = np.array(initial_solution, dtype=float)
    best_solution = current.copy()
    best_c2 = compute_c2(current.tolist())
    
    # Track improvement for adaptive behavior
    last_improvement = 0
    
    # Step sizes with better distribution for exploration
    step_sizes = [0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try adjustments with current step sizes
        current_step_sizes = step_sizes[:max(1, len(step_sizes) - iteration // 10)]
        
        # Try adjustments with current step sizes
        for i in range(len(current)):
            for adjustment in current_step_sizes + [-x for x in current_step_sizes]:
                if abs(adjustment) > 0:
                    test_current = current.copy()
                    test_current[i] = max(0.0, current[i] + adjustment)
                    
                    new_c2 = compute_c2(test_current.tolist())
                    if new_c2 > best_c2:
                        best_c2 = new_c2
                        best_solution = test_current.copy()
                        current = test_current.copy()
                        improved = True
                        last_improvement = iteration
        
        # Occasionally make larger random adjustments for escape from local optima
        if not improved and iteration % 3 == 0:
            for _ in range(len(current) // 20):
                i = random.randint(0, len(current) - 1)
                adjustment = random.uniform(-0.1, 0.1)
                test_current = current.copy()
                test_current[i] = max(0.0, current[i] + adjustment)
                
                new_c2 = compute_c2(test_current.tolist())
                if new_c2 > best_c2:
                    best_c2 = new_c2
                    best_solution = test_current.copy()
                    current = test_current.copy()
                    improved = True
                    last_improvement = iteration
        
        # Early stopping if no improvement for too long
        if iteration - last_improvement > 15:
            break
    
    return best_solution.tolist()

def gradient_based_refinement(initial_solution: List[float], max_iter: int = 50) -> List[float]:
    """
    Use gradient-based optimization for rapid refinement.
    """
    x0 = np.array(initial_solution)
    
    def objective(x):
        f_list = x.tolist()
        c2_val = compute_c2(f_list)
        return -c2_val  # Negative because we want to maximize
    
    try:
        # Use L-BFGS-B with bounds for fast optimization
        bounds = [(0, None) for _ in range(len(x0))]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': max_iter, 'ftol': 1e-10})
        
        refined = np.maximum(0, result.x).tolist()
        return refined
    except Exception:
        return np.maximum(0, x0).tolist()

def hybrid_optimization_approach(n_steps: int = 500) -> List[float]:
    """
    Hybrid optimization approach combining multiple strategies for better results.
    """
    # Create mathematical patterns that are likely to perform well
    initial_patterns = create_advanced_patterns(n_steps)
    
    best_solution = None
    best_c2 = -np.inf
    
    # Evaluate all initial patterns thoroughly
    for name, pattern in initial_patterns:
        try:
            c2_val = compute_c2(pattern)
            if c2_val > best_c2:
                best_c2 = c2_val
                best_solution = pattern[:]
        except Exception as e:
            warnings.warn(f"Failed evaluating {name}: {e}")
            continue
    
    # Apply enhanced local search to the best pattern with more iterations
    if best_solution is not None:
        refined_solution = enhanced_local_search(best_solution, 120)
        final_c2 = compute_c2(refined_solution)
        
        # Use refined version if it's better
        if final_c2 > best_c2:
            best_solution = refined_solution[:]
            best_c2 = final_c2
    
    # Apply differential evolution for global search with more iterations
    if best_solution is not None:
        try:
            # Use bounds that make sense for our problem
            bounds = [(0.0, 10.0) for _ in range(n_steps)]
            
            # Run DE with better configuration
            result = differential_evolution(
                lambda x: -compute_c2(x.tolist()),
                bounds,
                seed=42,
                maxiter=30,
                popsize=25,
                mutation=(0.9, 1.0),
                recombination=0.8,
                disp=False
            )
            if result.success:
                c2_val = compute_c2(result.x.tolist())
                if c2_val > best_c2:
                    best_c2 = c2_val
                    best_solution = result.x.tolist()
        except Exception:
            pass
    
    # Additional fine-tuning with gradient-based optimization
    if best_solution is not None:
        try:
            # Try gradient-based refinement with more iterations
            refined_solution = gradient_based_refinement(best_solution, 50)
            final_c2 = compute_c2(refined_solution)
            if final_c2 > best_c2:
                best_c2 = final_c2
                best_solution = refined_solution
        except Exception:
            pass
    
    # Final validation and return
    if best_solution is not None and best_c2 > 0.0:
        return [max(0.0, x) for x in best_solution]
    else:
        # Fallback to the best pattern from initial set
        pattern = create_advanced_patterns(n_steps)[0][1]
        return pattern

def construct_function() -> List[float]:
    """
    Main function to construct the optimal step function.
    Uses enhanced hybrid approach with better patterns and optimization strategies.
    """
    try:
        # Use more steps to increase resolution and potential for better results
        n_steps = 500
        
        # Run hybrid optimization approach
        best_solution = hybrid_optimization_approach(n_steps)
        
        # Final validation
        final_c2 = compute_c2(best_solution)
        
        if final_c2 > 0.0:
            return best_solution
        else:
            # Fallback to a proven good pattern
            return create_advanced_patterns(n_steps)[0][1]
            
    except Exception as e:
        warnings.warn(f"Main optimization failed: {e}")
        # Return a simple fallback pattern
        return [1.0] * 500

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
