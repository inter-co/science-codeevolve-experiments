# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
from scipy.optimize import differential_evolution, minimize
import warnings
import time
import random
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
    Compute the three norms needed for C2 calculation using direct convolution.
    """
    if len(f_values) == 0:
        return 0.0, 0.0, 0.0
    
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using numpy's convolve
    g = convolve(f, f, mode='full')
    
    # For the evaluator's approach:
    # ||g||₂² = sum of squares of g values (direct sum approach)
    # ||g||₁ = sum of absolute values of g / (len(g) + 1) 
    # ||g||∞ = max absolute value of g
    
    g_abs = np.abs(g)
    
    # ||g||₂² using evaluator's approach: sum of squares directly
    g_l2_sq = np.sum(g_abs**2)
    
    # ||g||₁ using evaluator's approach: sum of absolute values divided by (len(g)+1)
    g_l1 = np.sum(g_abs) / (len(g) + 1)
    
    # ||g||∞ 
    g_linf = np.max(g_abs)
    
    return g_l2_sq, g_l1, g_linf

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

def create_mathematically_optimized_patterns(n: int) -> List[Tuple[str, List[float]]]:
    """
    Create mathematically optimized patterns with proven effectiveness for maximizing C2.
    """
    patterns = []
    
    # Pattern 1: Ultra-sharp tent pattern - critical for maximizing convolution concentration
    ultra_sharp_tent = []
    for i in range(n):
        dist_from_center = abs(i - (n-1)/2) / ((n-1)/2)
        # Extremely sharp edges to maximize convolution concentration effect
        val = max(0, 1.0 - 10.0 * dist_from_center)
        ultra_sharp_tent.append(val)
    total = sum(ultra_sharp_tent)
    if total > 0:
        ultra_sharp_tent = [x * 2.0 / total for x in ultra_sharp_tent]
    patterns.append(("ultra_sharp_tent", ultra_sharp_tent))
    
    # Pattern 2: Multi-peak with wide spacing for maximal convolution spread
    wide_multi_peak = [0.0] * n
    peak_positions = [n//12, 2*n//12, 3*n//12, 9*n//12, 10*n//12, 11*n//12]  
    peak_height = 4.0
    peak_width = min(20, n // 6)
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                wide_multi_peak[i] += peak_height * (1 - dist / peak_width)
    
    # Normalize
    total = sum(wide_multi_peak)
    if total > 0:
        wide_multi_peak = [x * 2.0 / total for x in wide_multi_peak]
    patterns.append(("wide_multi_peak", wide_multi_peak))
    
    # Pattern 3: High-frequency oscillation pattern with maximum spread
    high_freq_osc = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Very high frequency oscillations to maximize convolution spread
        val = 0.5 + 0.3 * np.sin(40 * np.pi * x) + 0.2 * np.cos(60 * np.pi * x) + 0.1 * np.sin(80 * np.pi * x)
        high_freq_osc.append(max(0, val))
    
    total = sum(high_freq_osc)
    if total > 0:
        high_freq_osc = [x * 2.0 / total for x in high_freq_osc]
    patterns.append(("high_freq_osc", high_freq_osc))
    
    # Pattern 4: Extremely asymmetric pattern with strategic slope ratios
    extreme_asymmetric = []
    for i in range(n):
        pos = i / (n - 1) if n > 1 else 0.5
        # Very steep rise, very gentle fall to maximize convolution properties
        if pos < 0.1:
            val = 1.0 - 15 * pos
        elif pos > 0.9:
            val = 1.0 - 5 * (1 - pos)
        else:
            val = 0.1 + 0.6 * np.cos(20 * np.pi * pos)
        extreme_asymmetric.append(max(0.0, val))
    
    total = sum(extreme_asymmetric)
    if total > 0:
        extreme_asymmetric = [x * 2.0 / total for x in extreme_asymmetric]
    patterns.append(("extreme_asymmetric", extreme_asymmetric))
    
    # Pattern 5: Double Gaussian with strategic peak placement
    double_gaussian = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Two peaks with different characteristics for optimal convolution
        val = (1.0 * np.exp(-((x - 0.25)**2) * 30) + 
               1.0 * np.exp(-((x + 0.25)**2) * 30)) * (1 + 0.2 * np.cos(15 * np.pi * x))
        double_gaussian.append(max(0.0, val))
    total = sum(double_gaussian)
    if total > 0:
        double_gaussian = [x * 2.0 / total for x in double_gaussian]
    patterns.append(("double_gaussian", double_gaussian))
    
    # Pattern 6: Central peak with very strong exponential decay
    strong_central = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Very strong central peak with rapid exponential decay
        val = 2.5 * np.exp(-x**2 * 40) * (1 + 0.3 * np.cos(25 * np.pi * x))
        strong_central.append(max(0.0, val))
    total = sum(strong_central)
    if total > 0:
        strong_central = [x * 2.0 / total for x in strong_central]
    patterns.append(("strong_central", strong_central))
    
    # Pattern 7: Sparse high-amplitude peaks with minimal overlap
    sparse_high_amp = [0.0] * n
    peak_positions = [n//10, 3*n//10, 7*n//10, 9*n//10]
    peak_height = 5.0
    peak_width = min(18, n // 8)
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                sparse_high_amp[i] += peak_height * (1 - dist / peak_width)
    
    # Normalize
    total = sum(sparse_high_amp)
    if total > 0:
        sparse_high_amp = [x * 2.0 / total for x in sparse_high_amp]
    patterns.append(("sparse_high_amp", sparse_high_amp))
    
    # Pattern 8: Multi-frequency oscillation with strategic phase alignment
    multi_freq_osc = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Multiple frequencies with specific phase relationships
        val = 0.4 + 0.3 * np.cos(30 * np.pi * x) + 0.2 * np.sin(45 * np.pi * x) + 0.1 * np.cos(60 * np.pi * x)
        multi_freq_osc.append(max(0.0, val))
    total = sum(multi_freq_osc)
    if total > 0:
        multi_freq_osc = [x * 2.0 / total for x in multi_freq_osc]
    patterns.append(("multi_freq_osc", multi_freq_osc))
    
    # Pattern 9: Very narrow sharp peaks for extreme convolution
    very_narrow_peaks = [0.0] * n
    peak_positions = [n//8, 3*n//8, 5*n//8, 7*n//8]
    peak_height = 6.0
    peak_width = min(12, n // 10)
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                very_narrow_peaks[i] += peak_height * (1 - dist / peak_width)
    
    # Normalize
    total = sum(very_narrow_peaks)
    if total > 0:
        very_narrow_peaks = [x * 2.0 / total for x in very_narrow_peaks]
    patterns.append(("very_narrow_peaks", very_narrow_peaks))
    
    # Pattern 10: Optimized bell curve with oscillation
    optimized_bell = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Optimized bell shape with oscillation for better autoconvolution properties
        val = 1.8 * np.exp(-x**2 * 25) * (1 + 0.25 * np.cos(18 * np.pi * x))
        optimized_bell.append(max(0.0, val))
    total = sum(optimized_bell)
    if total > 0:
        optimized_bell = [x * 2.0 / total for x in optimized_bell]
    patterns.append(("optimized_bell", optimized_bell))
    
    return patterns

def advanced_local_search(initial_solution: List[float], max_iter: int = 200) -> List[float]:
    """
    Advanced local search with multiple strategies for better exploration.
    """
    current = np.array(initial_solution, dtype=float)
    best_solution = current.copy()
    best_c2 = compute_c2(current.tolist())
    
    # Track improvement for adaptive behavior
    last_improvement = 0
    improvement_count = 0
    
    # Step sizes with progressive reduction and more thorough exploration
    step_sizes = [0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.002, 0.001]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try adjustments with current step sizes
        current_step_sizes = step_sizes[:max(1, len(step_sizes) - iteration // 12)]
        
        # Try both positive and negative adjustments
        adjustments = []
        for s in current_step_sizes:
            adjustments.extend([s, -s])
        
        # Try adjustments to each element
        for i in range(len(current)):
            for adjustment in adjustments:
                if abs(adjustment) > 0:
                    test_current = current.copy()
                    test_current[i] = max(0.0, current[i] + adjustment)
                    
                    new_c2 = compute_c2(test_current.tolist())
                    if new_c2 > best_c2:
                        best_c2 = new_c2
                        best_solution = test_current.copy()
                        current = test_current.copy()
                        improved = True
                        improvement_count += 1
                        last_improvement = iteration
        
        # Occasionally make larger random adjustments for escape from local optima
        if not improved and iteration % 8 == 0:
            for _ in range(len(current) // 12):
                i = random.randint(0, len(current) - 1)
                adjustment = random.uniform(-0.15, 0.15)
                test_current = current.copy()
                test_current[i] = max(0.0, current[i] + adjustment)
                
                new_c2 = compute_c2(test_current.tolist())
                if new_c2 > best_c2:
                    best_c2 = new_c2
                    best_solution = test_current.copy()
                    current = test_current.copy()
                    improved = True
                    improvement_count += 1
                    last_improvement = iteration
        
        # Early stopping if no improvement for a while
        if iteration - last_improvement > 35:
            break
    
    return best_solution.tolist()

def enhanced_global_optimization_approach(n_steps: int = 600) -> List[float]:
    """
    Enhanced global optimization approach that leverages the best patterns from inspirations.
    """
    # Create specialized patterns (from inspiration 3)
    initial_patterns = create_mathematically_optimized_patterns(n_steps)
    
    best_solution = None
    best_c2 = -np.inf
    
    # Evaluate all patterns thoroughly
    for name, pattern in initial_patterns:
        try:
            c2_val = compute_c2(pattern)
            if c2_val > best_c2:
                best_c2 = c2_val
                best_solution = pattern[:]
        except Exception as e:
            warnings.warn(f"Failed evaluating {name}: {e}")
            continue
    
    # Apply advanced local search to the best pattern with more iterations
    if best_solution is not None:
        refined_solution = advanced_local_search(best_solution, 200)
        final_c2 = compute_c2(refined_solution)
        
        if final_c2 > best_c2:
            best_solution = refined_solution[:]
            best_c2 = final_c2
    
    # Apply differential evolution with aggressive settings for further improvement
    if best_solution is not None:
        try:
            # Use bounds for our problem
            bounds = [(0.0, 5.0) for _ in range(n_steps)]
            
            # Run differential evolution with multiple configurations
            de_configs = [
                {'maxiter': 50, 'popsize': 30, 'mutation': (0.7, 1.0), 'recombination': 0.9},
                {'maxiter': 45, 'popsize': 35, 'mutation': (0.6, 1.0), 'recombination': 0.85},
                {'maxiter': 40, 'popsize': 25, 'mutation': (0.8, 1.0), 'recombination': 0.95},
            ]
            
            for i, config in enumerate(de_configs):
                seed_val = 42 + i * 17
                result = differential_evolution(
                    lambda x: -compute_c2(x.tolist()),
                    bounds,
                    seed=seed_val,
                    **config,
                    disp=False
                )
                if result.success:
                    c2_val = compute_c2(result.x.tolist())
                    if c2_val > best_c2:
                        best_c2 = c2_val
                        best_solution = result.x.tolist()
        except Exception:
            pass
    
    # Apply gradient-based optimization for fine-tuning with limited iterations
    if best_solution is not None:
        try:
            x0 = np.array(best_solution)
            
            def objective(x):
                f_list = x.tolist()
                c2_val = compute_c2(f_list)
                return -c2_val  # Negative because we want to maximize
            
            # Use L-BFGS-B with bounds for fast optimization
            bounds = [(0, None) for _ in range(len(x0))]
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 60, 'ftol': 1e-10})
            
            if result.success:
                refined = np.maximum(0, result.x).tolist()
                final_c2 = compute_c2(refined)
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_solution = refined
        except Exception:
            pass
    
    # Final validation and return
    if best_solution is not None and best_c2 > 0.0:
        return [max(0.0, x) for x in best_solution]
    else:
        # Fallback to a proven good pattern
        pattern = create_mathematically_optimized_patterns(n_steps)[0][1]
        return pattern

def construct_function() -> List[float]:
    """
    Main function to construct the optimal step function.
    Uses enhanced global optimization approach to maximize C2.
    """
    start_time = time.time()
    
    try:
        # Use more steps to allow for better resolution and optimization (as in inspiration 3)
        n_steps = 600
        
        # Run enhanced global optimization approach
        best_solution = enhanced_global_optimization_approach(n_steps)
        
        # Final validation
        final_c2 = compute_c2(best_solution)
        
        if final_c2 > 0.0:
            benchmark_ratio = final_c2 / 0.962
            eval_time = time.time() - start_time
            print(f"C2: {final_c2}, Benchmark Ratio: {benchmark_ratio}, Time: {eval_time:.4f}s")
            return best_solution
        else:
            # Fallback to a proven good pattern
            pattern = create_mathematically_optimized_patterns(n_steps)[0][1]
            benchmark_ratio = compute_c2(pattern) / 0.962
            eval_time = time.time() - start_time
            print(f"Fallback C2: {compute_c2(pattern)}, Benchmark Ratio: {benchmark_ratio}, Time: {eval_time:.4f}s")
            return pattern
            
    except Exception as e:
        warnings.warn(f"Main optimization failed: {e}")
        # Return a simple fallback pattern
        return [1.0] * 600

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
