# EVOLVE-BLOCK-START

import numpy as np
import time
import random
from typing import List, Tuple
from numba import jit
import warnings
from scipy.signal import convolve
from scipy.optimize import differential_evolution, minimize

# Use Numba for faster computation
@jit(nopython=True)
def compute_autoconvolution_fast(f_values: np.ndarray) -> np.ndarray:
    """
    Fast computation of autoconvolution using direct convolution.
    """
    n = len(f_values)
    g = np.zeros(2 * n - 1)  # Result size for linear convolution
    
    # Direct convolution computation for better performance
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < len(g):
                g[idx] += f_values[i] * f_values[j]
    
    return g

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    """
    if len(f_values) == 0:
        return 0.0, 0.0, 0.0
    
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using numpy's convolve
    g = convolve(f, f, mode='full')
    
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

def create_aggressive_patterns(n: int) -> List[Tuple[str, List[float]]]:
    """
    Create highly aggressive mathematical patterns specifically designed to push C2 toward maximum.
    Based on inspiration program insights.
    """
    patterns = []
    
    # Pattern 1: Ultra-sharp tent with maximum edge concentration
    ultra_sharp = []
    for i in range(n):
        dist_from_center = abs(i - (n-1)/2) / ((n-1)/2)
        # Extremely sharp edges for maximum convolution concentration
        val = max(0, 1.0 - 12.0 * dist_from_center)
        ultra_sharp.append(val)
    total = sum(ultra_sharp)
    if total > 0:
        ultra_sharp = [x * 3.5 / total for x in ultra_sharp]
    patterns.append(("ultra_sharp", ultra_sharp))
    
    # Pattern 2: Multi-peak with maximum constructive interference
    multi_peak = [0.0] * n
    # Dense peak placement to maximize constructive interference
    peak_positions = [n//16, 2*n//16, 3*n//16, 4*n//16, 5*n//16, 6*n//16, 7*n//16, 
                      9*n//16, 10*n//16, 11*n//16, 12*n//16, 13*n//16, 14*n//16, 15*n//16]
    peak_height = 4.5
    peak_width = min(15, n // 8)
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                multi_peak[i] += peak_height * (1 - dist / peak_width)
    
    # Normalize
    total = sum(multi_peak)
    if total > 0:
        multi_peak = [x * 3.0 / total for x in multi_peak]
    patterns.append(("dense_multi_peak", multi_peak))
    
    # Pattern 3: Highly oscillatory pattern with precise phase control
    oscillatory = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Very high frequency oscillations with strong amplitude
        val = 0.7 + 0.3 * np.cos(35 * np.pi * x) + 0.2 * np.sin(50 * np.pi * x) + 0.1 * np.cos(70 * np.pi * x)
        oscillatory.append(max(0.0, val))
    
    total = sum(oscillatory)
    if total > 0:
        oscillatory = [x * 3.0 / total for x in oscillatory]
    patterns.append(("high_freq_osc", oscillatory))
    
    # Pattern 4: Extreme asymmetric with very steep gradients
    extreme_asym = []
    for i in range(n):
        pos = i / (n - 1) if n > 1 else 0.5
        # Very steep rise, very gentle fall
        if pos < 0.05:
            val = 1.0 - 20 * pos
        elif pos > 0.95:
            val = 1.0 - 10 * (1 - pos)
        else:
            val = 0.1 + 0.8 * np.cos(15 * np.pi * pos) + 0.05 * np.sin(30 * np.pi * pos)
        extreme_asym.append(max(0.0, val))
    
    total = sum(extreme_asym)
    if total > 0:
        extreme_asym = [x * 2.5 / total for x in extreme_asym]
    patterns.append(("extreme_asym", extreme_asym))
    
    # Pattern 5: Double Gaussian with peak separation optimized for convolution
    double_gaussian = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Two strongly peaked Gaussians with strategic separation
        val = (1.5 * np.exp(-((x - 0.3)**2) * 40) + 
               1.5 * np.exp(-((x + 0.3)**2) * 40)) * (1 + 0.3 * np.cos(20 * np.pi * x))
        double_gaussian.append(max(0.0, val))
    total = sum(double_gaussian)
    if total > 0:
        double_gaussian = [x * 3.0 / total for x in double_gaussian]
    patterns.append(("double_gaussian", double_gaussian))
    
    # Pattern 6: Concentrated with ultra-fast decay
    concentrated = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Very strong central peak with ultra-fast decay
        val = 3.0 * np.exp(-x**2 * 50) * (1 + 0.4 * np.cos(25 * np.pi * x))
        concentrated.append(max(0.0, val))
    total = sum(concentrated)
    if total > 0:
        concentrated = [x * 3.5 / total for x in concentrated]
    patterns.append(("ultra_concentrated", concentrated))
    
    # Pattern 7: Sparse but extremely high amplitude peaks
    sparse_high = [0.0] * n
    # Strategically placed very high peaks
    peak_positions = [n//12, 3*n//12, 5*n//12, 7*n//12, 9*n//12, 11*n//12]
    peak_height = 6.0
    peak_width = min(10, n // 12)
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                sparse_high[i] += peak_height * (1 - dist / peak_width)
    
    # Normalize
    total = sum(sparse_high)
    if total > 0:
        sparse_high = [x * 3.0 / total for x in sparse_high]
    patterns.append(("sparse_high", sparse_high))
    
    # Pattern 8: Multi-frequency with phase optimization
    multi_freq = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Multiple frequencies with optimized amplitudes and phases
        val = 0.5 + 0.3 * np.cos(40 * np.pi * x) + 0.2 * np.sin(60 * np.pi * x) + 0.15 * np.cos(80 * np.pi * x) + 0.05 * np.sin(100 * np.pi * x)
        multi_freq.append(max(0.0, val))
    total = sum(multi_freq)
    if total > 0:
        multi_freq = [x * 3.0 / total for x in multi_freq]
    patterns.append(("multi_freq", multi_freq))
    
    # Pattern 9: Very narrow, very high peaks
    narrow_peaks = [0.0] * n
    peak_positions = [n//10, 2*n//10, 3*n//10, 4*n//10, 5*n//10, 6*n//10, 7*n//10, 8*n//10, 9*n//10]
    peak_height = 7.0
    peak_width = min(8, n // 15)
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                narrow_peaks[i] += peak_height * (1 - dist / peak_width)
    
    # Normalize
    total = sum(narrow_peaks)
    if total > 0:
        narrow_peaks = [x * 3.0 / total for x in narrow_peaks]
    patterns.append(("narrow_peaks", narrow_peaks))
    
    # Pattern 10: Optimized bell curve with superimposed oscillation
    optimized_bell = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Supercharged bell with strong oscillation
        val = 2.0 * np.exp(-x**2 * 35) * (1 + 0.4 * np.cos(20 * np.pi * x) + 0.1 * np.sin(40 * np.pi * x))
        optimized_bell.append(max(0.0, val))
    total = sum(optimized_bell)
    if total > 0:
        optimized_bell = [x * 3.0 / total for x in optimized_bell]
    patterns.append(("optimized_bell", optimized_bell))
    
    return patterns

def aggressive_local_search(initial_solution: List[float], max_iter: int = 150) -> List[float]:
    """
    Aggressive local search with extensive exploration and adaptive strategies.
    """
    current = np.array(initial_solution, dtype=float)
    best_solution = current.copy()
    best_c2 = compute_c2(current.tolist())
    
    # Track improvement for adaptive behavior
    last_improvement = 0
    improvement_count = 0
    
    # Very aggressive step sizes for quick exploration
    step_sizes = [0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.003, 0.002, 0.001]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try adjustments with current step sizes
        # Decrease step size as iterations progress for better refinement
        current_step_sizes = step_sizes[:len(step_sizes) - iteration // 12]
        if not current_step_sizes:
            current_step_sizes = [0.001]
            
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
                        improvement_count += 1
                        last_improvement = iteration
        
        # Occasionally make larger random adjustments for escape from local optima
        if not improved and iteration % 4 == 0:
            # Make more substantial random adjustments occasionally
            for _ in range(len(current) // 10):
                i = random.randint(0, len(current) - 1)
                # Use larger adjustments occasionally but with mathematical guidance
                adjustment = random.uniform(-0.2, 0.2)
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
        
        # Early stopping if no improvement for too long
        if iteration - last_improvement > 8:
            break
    
    return best_solution.tolist()

def aggressive_hybrid_approach(n_steps: int = 700) -> List[float]:
    """
    Aggressive hybrid approach that combines multiple strategies for maximum C2.
    """
    # Create aggressive patterns
    initial_patterns = create_aggressive_patterns(n_steps)
    
    best_solution = None
    best_c2 = -np.inf
    
    # Evaluate all initial patterns thoroughly with more careful selection
    for name, pattern in initial_patterns:
        try:
            c2_val = compute_c2(pattern)
            if c2_val > best_c2:
                best_c2 = c2_val
                best_solution = pattern[:]
        except Exception as e:
            warnings.warn(f"Failed evaluating {name}: {e}")
            continue
    
    # Apply aggressive local search to the best pattern with many iterations
    if best_solution is not None:
        refined_solution = aggressive_local_search(best_solution, 180)
        final_c2 = compute_c2(refined_solution)
        
        # Use refined version if it's better
        if final_c2 > best_c2:
            best_solution = refined_solution[:]
            best_c2 = final_c2
    
    # Apply gradient-based optimization for fine-tuning with more iterations
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
                             options={'maxiter': 70, 'ftol': 1e-14, 'gtol': 1e-14})
            
            if result.success:
                refined = np.maximum(0, result.x).tolist()
                final_c2 = compute_c2(refined)
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_solution = refined
        except Exception:
            pass
    
    # Apply differential evolution with even more aggressive configurations
    if best_solution is not None:
        try:
            # Use bounds that make sense for our problem
            bounds = [(0.0, 20.0) for _ in range(n_steps)]
            
            # Run DE with very aggressive parameters for maximum exploration
            de_configs = [
                {'maxiter': 60, 'popsize': 40, 'mutation': (0.75, 1.0), 'recombination': 0.95},
                {'maxiter': 50, 'popsize': 45, 'mutation': (0.7, 1.0), 'recombination': 0.9},
                {'maxiter': 40, 'popsize': 35, 'mutation': (0.8, 1.0), 'recombination': 0.98},
            ]
            
            for i, config in enumerate(de_configs):
                # Use different seeds for diversity
                seed_val = 100 + i * 23
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
    
    # Final validation and return
    if best_solution is not None and best_c2 > 0.0:
        return [max(0.0, x) for x in best_solution]
    else:
        # Fallback to the best pattern from initial set
        pattern = create_aggressive_patterns(n_steps)[0][1]
        return pattern

def construct_function() -> List[float]:
    """
    Main function to construct the optimal step function.
    Uses an aggressive hybrid approach to maximize C2.
    """
    start_time = time.time()
    
    try:
        # Use more steps to allow for better resolution and optimization
        n_steps = 700
        
        # Run aggressive hybrid optimization
        best_solution = aggressive_hybrid_approach(n_steps)
        
        # Final validation
        final_c2 = compute_c2(best_solution)
        
        if final_c2 > 0.0:
            benchmark_ratio = final_c2 / 0.962
            eval_time = time.time() - start_time
            print(f"C2: {final_c2}, Benchmark Ratio: {benchmark_ratio}, Time: {eval_time:.4f}s")
            return best_solution
        else:
            # Fallback to a proven good pattern
            pattern = create_aggressive_patterns(n_steps)[0][1]
            benchmark_ratio = compute_c2(pattern) / 0.962
            eval_time = time.time() - start_time
            print(f"Fallback C2: {compute_c2(pattern)}, Benchmark Ratio: {benchmark_ratio}, Time: {eval_time:.4f}s")
            return pattern
            
    except Exception as e:
        warnings.warn(f"Main optimization failed: {e}")
        # Return a simple fallback pattern
        return [1.0] * 700

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
