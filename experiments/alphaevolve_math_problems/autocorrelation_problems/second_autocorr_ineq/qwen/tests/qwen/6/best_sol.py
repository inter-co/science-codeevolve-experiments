# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
import warnings
import time
import random
from typing import List, Tuple

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    f_values: step heights (non-negative)
    Returns: (||g||₂², ||g||₁, ||g||∞) where g = f*f
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

def create_aggressive_patterns(n: int) -> List[Tuple[str, List[float]]]:
    """
    Create aggressive patterns inspired by best performers from inspirations.
    These patterns are designed to produce high C2 values through sharp edges and concentrated peaks.
    """
    patterns = []
    
    # Pattern 1: Ultra-sharp tent with maximum edge emphasis (like INSPIRATION 2)
    ultra_sharp_tent = []
    for i in range(n):
        dist_from_center = abs(i - (n-1)/2) / ((n-1)/2)
        # Very aggressive sharpening to maximize convolution impact
        val = max(0, 1.0 - 15.0 * dist_from_center)
        ultra_sharp_tent.append(val)
    total = sum(ultra_sharp_tent)
    if total > 0:
        ultra_sharp_tent = [x * 4.0 / total for x in ultra_sharp_tent]
    patterns.append(("ultra_sharp_tent", ultra_sharp_tent))
    
    # Pattern 2: Multi-peak with ultra-dense spacing and high amplitude (like INSPIRATION 2)
    ultra_multi_peak = [0.0] * n
    # Even more dense peak spacing for maximum constructive interference
    peak_positions = [n//12, 2*n//12, 3*n//12, 4*n//12, 5*n//12, 6*n//12, 7*n//12, 8*n//12, 9*n//12, 10*n//12, 11*n//12]
    peak_height = 4.0
    peak_width = min(4, n // 15)
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                ultra_multi_peak[i] += peak_height * (1 - dist / peak_width)
    
    # Normalize
    total = sum(ultra_multi_peak)
    if total > 0:
        ultra_multi_peak = [x * 3.8 / total for x in ultra_multi_peak]
    patterns.append(("ultra_multi_peak", ultra_multi_peak))
    
    # Pattern 3: Extremely asymmetric with very steep edges (like INSPIRATION 2)
    ultra_asymmetric = []
    for i in range(n):
        pos = i / (n - 1) if n > 1 else 0.5
        # Extremely steep rise and fall for maximum convolution impact
        if pos < 0.1:
            val = 1.0 - 25 * pos
        elif pos > 0.9:
            val = 1.0 - 25 * (1 - pos)
        else:
            val = 0.1 + 0.8 * np.cos(20 * np.pi * pos)
        ultra_asymmetric.append(max(0.0, val))
    
    total = sum(ultra_asymmetric)
    if total > 0:
        ultra_asymmetric = [x * 4.2 / total for x in ultra_asymmetric]
    patterns.append(("ultra_asymmetric", ultra_asymmetric))
    
    # Pattern 4: Ultra-high frequency oscillation with peak concentration (like INSPIRATION 2)
    ultra_high_freq = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Very high frequency with stronger central concentration
        val = 0.75 + 0.45 * np.sin(70 * np.pi * x) + 0.25 * np.cos(100 * np.pi * x) + 0.15 * np.sin(130 * np.pi * x)
        ultra_high_freq.append(max(0.0, val))
    
    total = sum(ultra_high_freq)
    if total > 0:
        ultra_high_freq = [x * 4.0 / total for x in ultra_high_freq]
    patterns.append(("ultra_high_freq", ultra_high_freq))
    
    # Pattern 5: Double peak with very narrow width and highest amplitude (like INSPIRATION 2)
    ultra_double_peak = [0.0] * n
    mid1 = n // 6
    mid2 = 5 * n // 6
    width = min(3, n // 20)
    for i in range(n):
        dist1 = abs(i - mid1)
        dist2 = abs(i - mid2)
        if dist1 < width:
            ultra_double_peak[i] += 5.0 * (1 - dist1 / width)
        if dist2 < width:
            ultra_double_peak[i] += 5.0 * (1 - dist2 / width)
    total = sum(ultra_double_peak)
    if total > 0:
        ultra_double_peak = [x * 5.0 / total for x in ultra_double_peak]
    patterns.append(("ultra_double_peak", ultra_double_peak))
    
    # Pattern 6: Spike pattern with very high peaks and ultra-narrow width (like INSPIRATION 2)
    ultra_spike_pattern = [0.0] * n
    peak_positions = [n//10, 2*n//10, 3*n//10, 4*n//10, 5*n//10, 6*n//10, 7*n//10, 8*n//10, 9*n//10]
    peak_height = 6.0
    peak_width = min(3, n // 25)
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                ultra_spike_pattern[i] += peak_height * (1 - dist / peak_width)
    
    total = sum(ultra_spike_pattern)
    if total > 0:
        ultra_spike_pattern = [x * 5.5 / total for x in ultra_spike_pattern]
    patterns.append(("ultra_spike_pattern", ultra_spike_pattern))
    
    # Pattern 7: Concentrated central peak with ultra-rapid decay (like INSPIRATION 3)
    ultra_concentrated = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Very strong central peak with ultra-rapid decay
        val = 4.5 * np.exp(-x**2 * 60) * (1 + 0.4 * np.cos(30 * np.pi * x))
        ultra_concentrated.append(max(0.0, val))
    
    total = sum(ultra_concentrated)
    if total > 0:
        ultra_concentrated = [x * 5.0 / total for x in ultra_concentrated]
    patterns.append(("ultra_concentrated", ultra_concentrated))
    
    # Pattern 8: Complex multi-peak with varying frequencies and amplitudes (like INSPIRATION 3)
    ultra_complex_multi = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Multiple overlapping oscillations with different frequencies and amplitudes
        val = 0.7 + 0.5 * np.cos(20 * np.pi * x) + 0.35 * np.sin(40 * np.pi * x) + 0.25 * np.cos(60 * np.pi * x) + 0.15 * np.sin(80 * np.pi * x)
        ultra_complex_multi.append(max(0.0, val))
    
    total = sum(ultra_complex_multi)
    if total > 0:
        ultra_complex_multi = [x * 4.0 / total for x in ultra_complex_multi]
    patterns.append(("ultra_complex_multi", ultra_complex_multi))
    
    return patterns

def aggressive_local_search(initial_solution: List[float], max_iter: int = 250) -> List[float]:
    """
    Aggressive local search with very thorough exploration and better adaptation.
    """
    current = np.array(initial_solution, dtype=float)
    best_solution = current.copy()
    best_c2 = compute_c2(current.tolist())
    
    # Track improvement for adaptive behavior
    last_improvement = 0
    improvement_count = 0
    
    # Much more aggressive step sizes for faster convergence
    step_sizes = [0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.003, 0.002, 0.001]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try many adjustments with current step sizes
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
                        improvement_count += 1
                        last_improvement = iteration
        
        # Occasionally make larger random adjustments for escape from local optima
        if not improved and iteration % 4 == 0:
            for _ in range(len(current) // 8):
                i = random.randint(0, len(current) - 1)
                adjustment = random.uniform(-0.3, 0.3)
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
        if iteration - last_improvement > 12:
            break
    
    return best_solution.tolist()

def aggressive_hybrid_approach(n_steps: int = 500) -> List[float]:
    """
    Aggressive hybrid approach that combines the most effective strategies from inspirations.
    """
    # Create aggressive initial patterns
    initial_patterns = create_aggressive_patterns(n_steps)
    
    best_solution = None
    best_c2 = -np.inf
    
    # Evaluate all initial patterns thoroughly with early stopping
    for name, pattern in initial_patterns:
        try:
            c2_val = compute_c2(pattern)
            if c2_val > best_c2:
                best_c2 = c2_val
                best_solution = pattern[:]
        except Exception as e:
            warnings.warn(f"Failed evaluating {name}: {e}")
            continue
    
    # Apply aggressive local search to the best pattern with more iterations
    if best_solution is not None:
        refined_solution = aggressive_local_search(best_solution, 250)
        final_c2 = compute_c2(refined_solution)
        
        # Use refined version if it's better
        if final_c2 > best_c2:
            best_solution = refined_solution[:]
            best_c2 = final_c2
    
    # Apply gradient-based optimization for fine-tuning with aggressive settings
    if best_solution is not None:
        try:
            x0 = np.array(best_solution)
            
            def objective(x):
                f_list = x.tolist()
                c2_val = compute_c2(f_list)
                return -c2_val  # Negative because we want to maximize
            
            # Use L-BFGS-B with very tight tolerances for maximum precision
            bounds = [(0, None) for _ in range(len(x0))]
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-16})
            
            if result.success:
                refined = np.maximum(0, result.x).tolist()
                final_c2 = compute_c2(refined)
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_solution = refined
        except Exception:
            pass
    
    # Apply differential evolution with aggressive parameters
    if best_solution is not None:
        try:
            bounds = [(0.0, 10.0) for _ in range(n_steps)]
            
            # Run DE with very aggressive configurations
            result = differential_evolution(
                lambda x: -compute_c2(x.tolist()),
                bounds,
                seed=42,
                maxiter=200,  # More iterations for better convergence
                popsize=80,   # Larger population
                mutation=(0.95, 1.0),  # Most aggressive mutation
                recombination=0.99,    # Very high recombination rate
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
        # Use fewer steps to maintain performance within time constraints
        n_steps = 450  # Slightly reduced for better time management
        
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
        return [1.0] * 500

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
