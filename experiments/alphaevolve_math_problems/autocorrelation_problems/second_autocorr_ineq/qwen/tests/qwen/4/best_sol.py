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

def create_mathematically_informed_pattern(n: int) -> List[float]:
    """
    Create a pattern informed by mathematical analysis of optimal functions.
    Based on the principle that optimal functions often have specific structures.
    """
    # Create a pattern that is high at edges and low in center (like a "tent" shape)
    pattern = [0.0] * n
    for i in range(n):
        # Distance from center
        dist_from_center = abs(i - n//2) / (n//2)
        # Create a tent-like shape with high values at edges
        pattern[i] = max(0, 1.0 - 2 * dist_from_center)
    
    # Normalize to avoid very extreme values
    total = sum(pattern)
    if total > 0:
        pattern = [x * 2.0 / total for x in pattern]
    
    return pattern

def create_advanced_patterns(n: int) -> List[Tuple[str, List[float]]]:
    """
    Create mathematically-informed patterns optimized for high C2 values.
    Leveraging insights from harmonic analysis and extremal function theory.
    """
    patterns = []
    
    # Pattern 1: Ultra-sharp tent pattern (inspired by highest performing patterns)
    ultra_sharp = []
    for i in range(n):
        dist_from_center = abs(i - (n-1)/2) / ((n-1)/2)
        # Even sharper edges for maximum convolution impact
        val = max(0, 1.0 - 12.0 * dist_from_center)
        ultra_sharp.append(val)
    total = sum(ultra_sharp)
    if total > 0:
        ultra_sharp = [x * 3.0 / total for x in ultra_sharp]
    patterns.append(("ultra_sharp", ultra_sharp))
    
    # Pattern 2: Multi-peak with optimized spacing and height
    multi_peak = [0.0] * n
    peak_positions = [n//8, 3*n//8, 5*n//8, 7*n//8]  # Tighter spacing
    peak_height = 3.5
    peak_width = min(10, n // 6)  # Smaller width for sharper peaks
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                multi_peak[i] += peak_height * (1 - dist / peak_width)
    
    # Normalize
    total = sum(multi_peak)
    if total > 0:
        multi_peak = [x * 2.5 / total for x in multi_peak]
    patterns.append(("multi_peak", multi_peak))
    
    # Pattern 3: Asymmetric with very steep edges
    asymmetric = []
    for i in range(n):
        pos = i / (n - 1) if n > 1 else 0.5
        # Extremely steep rise and fall for better convolution
        if pos < 0.12:
            val = 1.0 - 12 * pos
        elif pos > 0.88:
            val = 1.0 - 10 * (1 - pos)
        else:
            # Mix of sine/cosine for mathematical elegance
            val = 0.1 + 0.7 * np.cos(12 * np.pi * pos) + 0.1 * np.sin(24 * np.pi * pos)
        asymmetric.append(max(0.0, val))
    
    total = sum(asymmetric)
    if total > 0:
        asymmetric = [x * 2.5 / total for x in asymmetric]
    patterns.append(("asymmetric", asymmetric))
    
    # Pattern 4: High-frequency oscillation with strong central peak
    high_freq = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # High frequency components with strong central emphasis
        val = 0.7 + 0.3 * np.sin(30 * np.pi * x) + 0.2 * np.cos(50 * np.pi * x) + 0.1 * np.sin(70 * np.pi * x)
        high_freq.append(max(0.0, val))
    
    total = sum(high_freq)
    if total > 0:
        high_freq = [x * 2.5 / total for x in high_freq]
    patterns.append(("high_freq", high_freq))
    
    # Pattern 5: Double peak with very narrow width
    double_peak = [0.0] * n
    mid1 = n // 6
    mid2 = 5 * n // 6
    width = min(8, n // 10)  # Very narrow peaks
    for i in range(n):
        dist1 = abs(i - mid1)
        dist2 = abs(i - mid2)
        if dist1 < width:
            double_peak[i] += 3.0 * (1 - dist1 / width)
        if dist2 < width:
            double_peak[i] += 3.0 * (1 - dist2 / width)
    total = sum(double_peak)
    if total > 0:
        double_peak = [x * 2.5 / total for x in double_peak]
    patterns.append(("double_peak", double_peak))
    
    # Pattern 6: Concentrated peak with rapid decay
    concentrated = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Very strong central peak with rapid decay
        val = 2.5 * np.exp(-x**2 * 30) * (1 + 0.3 * np.cos(12 * np.pi * x))
        concentrated.append(max(0.0, val))
    
    total = sum(concentrated)
    if total > 0:
        concentrated = [x * 2.5 / total for x in concentrated]
    patterns.append(("concentrated", concentrated))
    
    # Pattern 7: Spike pattern with strategic placement
    spike_pattern = [0.0] * n
    peak_positions = [n//7, 2*n//7, 3*n//7, 4*n//7, 5*n//7, 6*n//7]
    peak_height = 4.0
    peak_width = min(6, n // 12)
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                spike_pattern[i] += peak_height * (1 - dist / peak_width)
    
    total = sum(spike_pattern)
    if total > 0:
        spike_pattern = [x * 2.5 / total for x in spike_pattern]
    patterns.append(("spike_pattern", spike_pattern))
    
    # Pattern 8: Mathematical combination with stronger emphasis
    combined = []
    for i in range(n):
        x = (i / (n - 1) if n > 1 else 0.5) * 2 - 1
        # Stronger mixing of components for optimal convolution
        val = 0.4 + 0.5 * np.cos(15 * np.pi * x) + 0.25 * np.sin(30 * np.pi * x) + 0.15 * np.exp(-x**2 * 20)
        combined.append(max(0.0, val))
    
    total = sum(combined)
    if total > 0:
        combined = [x * 2.5 / total for x in combined]
    patterns.append(("combined", combined))
    
    # Pattern 9: Extreme sharpness with peak emphasis
    extreme_sharp = []
    for i in range(n):
        dist_from_center = abs(i - (n-1)/2) / ((n-1)/2)
        # Very sharp edges to maximize convolution impact
        val = max(0, 1.0 - 15.0 * dist_from_center)
        extreme_sharp.append(val)
    total = sum(extreme_sharp)
    if total > 0:
        extreme_sharp = [x * 3.5 / total for x in extreme_sharp]
    patterns.append(("extreme_sharp", extreme_sharp))
    
    # Pattern 10: Multi-peak with mathematical spacing
    math_multi_peak = [0.0] * n
    # Golden ratio spacing with optimized positions
    golden_ratio = (1 + np.sqrt(5)) / 2
    peak_positions = [int(n * i / (golden_ratio + 1)) for i in range(1, 7)] 
    peak_positions = [p for p in peak_positions if p < n]
    peak_height = 3.0
    peak_width = min(12, n // 8)
    
    for peak_pos in peak_positions:
        for i in range(n):
            dist = abs(i - peak_pos)
            if dist < peak_width:
                math_multi_peak[i] += peak_height * (1 - dist / peak_width)
    
    total = sum(math_multi_peak)
    if total > 0:
        math_multi_peak = [x * 2.5 / total for x in math_multi_peak]
    patterns.append(("math_multi_peak", math_multi_peak))
    
    return patterns

def adaptive_local_search(initial_solution: List[float], max_iter: int = 80) -> List[float]:
    """
    Enhanced aggressive local search with mathematical precision.
    """
    current = np.array(initial_solution, dtype=float)
    best_solution = current.copy()
    best_c2 = compute_c2(current.tolist())
    
    # Track improvement for adaptive behavior
    last_improvement = 0
    
    # More aggressive step sizes for faster convergence
    step_sizes = [0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.002, 0.001]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try adjustments with current step sizes
        # Use more aggressive steps initially, then refine
        current_step_sizes = step_sizes[:len(step_sizes) - iteration // 8]
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
                        last_improvement = iteration
        
        # Occasionally make larger random adjustments for escape from local optima
        if not improved and iteration % 4 == 0:
            for _ in range(len(current) // 12):
                i = random.randint(0, len(current) - 1)
                # Use larger adjustments occasionally to escape local minima
                adjustment = random.uniform(-0.2, 0.2)
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
        if iteration - last_improvement > 10:
            break
    
    return best_solution.tolist()

def enhanced_hybrid_approach(n_steps: int = 500) -> List[float]:
    """
    Enhanced hybrid approach that combines multiple strategies for maximum C2.
    Focuses on mathematical optimization and efficient search strategies.
    """
    # Create advanced initial patterns
    initial_patterns = create_advanced_patterns(n_steps)
    
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
    
    # Apply aggressive local search to the best pattern with fewer iterations to save time
    if best_solution is not None:
        refined_solution = adaptive_local_search(best_solution, 60)
        final_c2 = compute_c2(refined_solution)
        
        # Use refined version if it's better
        if final_c2 > best_c2:
            best_solution = refined_solution[:]
            best_c2 = final_c2
    
    # Apply gradient-based optimization for fine-tuning with fewer iterations
    if best_solution is not None:
        try:
            x0 = np.array(best_solution)
            
            def objective(x):
                f_list = x.tolist()
                c2_val = compute_c2(f_list)
                return -c2_val  # Negative because we want to maximize
            
            # Use L-BFGS-B with bounds for fast optimization with fewer iterations
            bounds = [(0, None) for _ in range(len(x0))]
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if result.success:
                refined = np.maximum(0, result.x).tolist()
                final_c2 = compute_c2(refined)
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_solution = refined
        except Exception:
            pass
    
    # Apply differential evolution with more focused parameters for better results
    if best_solution is not None:
        try:
            # Use bounds that make sense for our problem
            bounds = [(0.0, 15.0) for _ in range(n_steps)]
            
            # Run DE with fewer iterations but more aggressive parameters to save time
            result = differential_evolution(
                lambda x: -compute_c2(x.tolist()),
                bounds,
                seed=42,
                maxiter=25,  # Fewer iterations to save time
                popsize=25,   # Slightly smaller population size
                mutation=(0.9, 1.0),  # More aggressive mutation
                recombination=0.95,    # Even higher recombination rate
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
        pattern = create_advanced_patterns(n_steps)[0][1]
        return pattern

def construct_function() -> List[float]:
    """
    Main function to construct the optimal step function.
    Uses an enhanced hybrid approach to maximize C2.
    """
    start_time = time.time()
    
    try:
        # Use a moderate number of steps for good balance of resolution and time
        n_steps = 500
        
        # Run enhanced hybrid optimization
        best_solution = enhanced_hybrid_approach(n_steps)
        
        # Final validation
        final_c2 = compute_c2(best_solution)
        
        if final_c2 > 0.0:
            benchmark_ratio = final_c2 / 0.962
            eval_time = time.time() - start_time
            print(f"C2: {final_c2}, Benchmark Ratio: {benchmark_ratio}, Time: {eval_time:.4f}s")
            return best_solution
        else:
            # Fallback to a proven good pattern
            pattern = create_advanced_patterns(n_steps)[0][1]
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
