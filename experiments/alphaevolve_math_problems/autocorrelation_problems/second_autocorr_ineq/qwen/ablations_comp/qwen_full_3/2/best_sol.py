# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import minimize
import random
from typing import List, Tuple
import time
import numba
from numba import jit

@jit(nopython=True)
def compute_autoconvolution_fast(f_values: List[float]) -> np.ndarray:
    """Fast numba-optimized autoconvolution computation."""
    n = len(f_values)
    g = np.zeros(2 * n - 1)
    
    # Manual convolution loop for better performance
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    return g

def compute_autoconvolution_norms_fast(f_values: List[float]) -> tuple:
    """
    Fast computation of norms using numba-optimized autoconvolution
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution using fast numba version
    g = compute_autoconvolution_fast(f_values)
    
    # For a function of length n, the convolution produces 2*n-1 elements
    # We want the central portion that represents the main convolution
    n = len(f)
    center = len(g) // 2
    
    # Extract the central portion that represents the main convolution
    # Using a more conservative approach to avoid edge effects
    half_range = 2 * n - 1  # Full convolution
    start_idx = max(0, center - half_range // 2)
    end_idx = min(len(g), center + half_range // 2)
    
    g_center = g[start_idx:end_idx]
    
    # Compute the three norms
    g_abs = np.abs(g_center)
    
    # ||g||₂² = sum(g²) 
    g_l2_squared = np.sum(g_abs**2)
    
    # ||g||₁ = sum(|g|)
    g_l1 = np.sum(g_abs)
    
    # ||g||∞ = max(|g|)
    g_linf = np.max(g_abs)
    
    return g_l2_squared, g_l1, g_linf

def compute_c2_fast(f_values: List[float]) -> float:
    """Fast C2 computation with error handling"""
    try:
        g_l2_squared, g_l1, g_linf = compute_autoconvolution_norms_fast(f_values)
        
        # Avoid division by zero
        if g_l1 <= 1e-15 or g_linf <= 1e-15:
            return 0.0
            
        c2 = g_l2_squared / (g_l1 * g_linf)
        return c2
    except Exception:
        return 0.0

def create_mathematical_patterns(n: int) -> List[Tuple[str, List[float]]]:
    """
    Create mathematically-inspired patterns based on harmonic analysis and optimal function theory
    """
    patterns = []
    
    # Pattern 1: Optimized double Gaussian with strategic placement
    x = np.linspace(-0.25, 0.25, n)
    # Focus on central region for better autoconvolution properties
    gaussian1 = np.exp(-0.5 * x**2 * 10)
    gaussian2 = 0.8 * np.exp(-0.5 * ((x - 0.15)**2 * 10))
    gaussian3 = 0.8 * np.exp(-0.5 * ((x + 0.15)**2 * 10))
    double_gaussian = gaussian1 + gaussian2 + gaussian3
    patterns.append(('double_gaussian', double_gaussian.tolist()))
    
    # Pattern 2: High-frequency oscillation with central peak
    x = np.linspace(-0.25, 0.25, n)
    central_peak = np.exp(-x**2 * 8) * 1.8
    oscillation = 0.3 * np.sin(8 * np.pi * x)
    oscillatory = np.maximum(0, central_peak + oscillation)
    patterns.append(('oscillatory', oscillatory.tolist()))
    
    # Pattern 3: Multi-peak with strategic spacing
    multi_peak = np.zeros(n)
    centers = [n//4, n//2, 3*n//4]
    for center in centers:
        sigma = n // 15
        x = np.arange(n) - center
        multi_peak += np.exp(-0.5 * (x / sigma)**2)
    patterns.append(('multi_peak', multi_peak.tolist()))
    
    # Pattern 4: Symmetric exponential with sharp peaks
    x = np.linspace(-0.25, 0.25, n)
    exp_pattern = np.exp(-5 * np.abs(x)) * 1.5
    patterns.append(('exp_symmetric', exp_pattern.tolist()))
    
    # Pattern 5: Trigonometric pattern with constructive interference
    x = np.linspace(-0.25, 0.25, n)
    trig_pattern = 1.0 + 0.6 * np.sin(4 * np.pi * x) + 0.4 * np.sin(8 * np.pi * x)
    patterns.append(('trig_pattern', trig_pattern.tolist()))
    
    # Pattern 6: Optimized for uniform convolution (flat g)
    x = np.linspace(-0.25, 0.25, n)
    flat_conv = 1.0 + 0.7 * np.exp(-x**2 * 4)
    patterns.append(('flat_conv', flat_conv.tolist()))
    
    return patterns

def create_analytical_patterns(n: int) -> List[Tuple[str, List[float]]]:
    """
    Create analytical patterns with specific mathematical properties
    """
    patterns = []
    
    # Pattern 1: Sinc-based pattern (often good for convolution properties)
    x = np.linspace(-0.25, 0.25, n)
    sinc_pattern = np.sinc(4 * x) * 1.8
    sinc_pattern = np.maximum(0, sinc_pattern)
    patterns.append(('sinc', sinc_pattern.tolist()))
    
    # Pattern 2: Rectangular with controlled peaks
    rect_pattern = np.ones(n)
    rect_pattern[n//4:3*n//4] = 2.0
    patterns.append(('rectangular', rect_pattern.tolist()))
    
    # Pattern 3: Smooth bump function
    x = np.linspace(-0.25, 0.25, n)
    bump = np.exp(-1 / (1 - (x/0.25)**2)) * (np.abs(x) < 0.25)
    bump = bump / np.max(bump) * 2.0 if np.max(bump) > 0 else bump
    patterns.append(('bump', bump.tolist()))
    
    # Pattern 4: Gaussian with multiple peaks
    x = np.linspace(-0.25, 0.25, n)
    gaussian_multi = np.exp(-0.5 * x**2) * 1.5
    gaussian_multi += 0.6 * np.exp(-0.5 * ((x - 0.15)**2)) 
    gaussian_multi += 0.6 * np.exp(-0.5 * ((x + 0.15)**2))
    patterns.append(('gaussian_multi', gaussian_multi.tolist()))
    
    # Pattern 5: Optimized for maximum C2 - narrow central peak with side lobes
    x = np.linspace(-0.25, 0.25, n)
    peak = np.exp(-x**2 * 20) * 3.0
    side_lobe1 = 0.5 * np.exp(-0.5 * ((x - 0.1)**2 * 8))
    side_lobe2 = 0.5 * np.exp(-0.5 * ((x + 0.1)**2 * 8))
    optimized = peak + side_lobe1 + side_lobe2
    patterns.append(('optimized', optimized.tolist()))
    
    return patterns

def create_special_patterns(n: int) -> List[Tuple[str, List[float]]]:
    """
    Create specialized patterns designed to push C2 toward theoretical maximum
    """
    patterns = []
    
    # Pattern 1: Very sharp central peak with minimal support
    x = np.linspace(-0.25, 0.25, n)
    sharp_peak = np.exp(-x**2 * 50) * 5.0
    patterns.append(('sharp_peak', sharp_peak.tolist()))
    
    # Pattern 2: Double peak with wide separation
    x = np.linspace(-0.25, 0.25, n)
    double_peak = 0.7 * np.exp(-0.5 * ((x - 0.1)**2 * 10)) + \
                  0.7 * np.exp(-0.5 * ((x + 0.1)**2 * 10))
    patterns.append(('double_wide', double_peak.tolist()))
    
    # Pattern 3: Optimized for flat autoconvolution
    x = np.linspace(-0.25, 0.25, n)
    # Create a pattern that produces a flatter convolution
    flat_pattern = 1.0 + 0.5 * np.cos(4 * np.pi * x)
    flat_pattern = np.maximum(0, flat_pattern)
    patterns.append(('flat_cosine', flat_pattern.tolist()))
    
    return patterns

def hybrid_optimization_approach() -> Tuple[List[float], float, float]:
    """
    Hybrid optimization approach combining multiple strategies for maximum efficiency
    """
    start_time = time.time()
    max_time = 55.0  # Leave buffer for cleanup
    
    best_c2 = 0.0
    best_heights = None
    eval_time = 0.0
    
    # Strategy 1: Try mathematical patterns with quick optimization
    n_steps_list = [400, 450, 500]  # Larger resolution for better optimization
    
    for n_steps in n_steps_list:
        if time.time() - start_time > max_time:
            break
            
        # Try mathematical patterns
        patterns = create_mathematical_patterns(n_steps)
        patterns.extend(create_analytical_patterns(n_steps))
        patterns.extend(create_special_patterns(n_steps))
        
        for pattern_name, pattern in patterns:
            if time.time() - start_time > max_time:
                break
                
            # Normalize pattern
            if np.max(pattern) > 0:
                normalized_pattern = [x / np.max(pattern) * 2.5 for x in pattern]
            else:
                normalized_pattern = pattern
                
            # Take a reasonable subset for optimization
            heights = normalized_pattern[:min(len(normalized_pattern), n_steps//3)]
            if len(heights) < 15:
                continue
                
            # Quick local optimization with fewer iterations
            try:
                def objective(h):
                    return -compute_c2_fast(h)
                
                # Use L-BFGS-B for local refinement with tight tolerances
                bounds = [(0, 15.0)] * len(heights)
                result = minimize(
                    objective, 
                    heights, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    final_heights = result.x.tolist()
                    final_c2 = -objective(final_heights)
                    
                    if final_c2 > best_c2:
                        best_c2 = final_c2
                        best_heights = final_heights
                        
            except Exception:
                continue
    
    # Strategy 2: If we haven't found something good yet, try a more systematic approach
    if best_heights is None or best_c2 <= 0.90:
        # Try a more direct approach with carefully chosen parameters
        n_steps = 500
        # Create a pattern that's known to work well for this type of problem
        x = np.linspace(-0.25, 0.25, n_steps)
        # Use a symmetric pattern that emphasizes the center with strategic oscillation
        pattern = np.exp(-x**2 * 12) * 2.0
        # Add oscillation to encourage good convolution properties
        pattern += 0.3 * np.sin(10 * np.pi * x)
        heights = pattern.tolist()
        
        # Refine with optimization
        try:
            def objective(h):
                return -compute_c2_fast(h)
            
            bounds = [(0, 15.0)] * len(heights)
            result = minimize(
                objective, 
                heights, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 25, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                final_heights = result.x.tolist()
                final_c2 = -objective(final_heights)
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_heights = final_heights
                    
        except Exception:
            pass
    
    # Strategy 3: Final aggressive optimization attempt with best known pattern
    if best_heights is None or best_c2 <= 0.92:
        try:
            # Use a more sophisticated pattern based on mathematical analysis
            n_steps = 500
            x = np.linspace(-0.25, 0.25, n_steps)
            
            # Create a pattern with multiple components that tend to maximize C2
            # Based on insights from optimal function theory and convolution properties
            pattern = np.exp(-x**2 * 10) * 2.5
            pattern += 0.4 * np.exp(-0.5 * ((x - 0.1)**2 * 15))
            pattern += 0.4 * np.exp(-0.5 * ((x + 0.1)**2 * 15))
            pattern += 0.2 * np.sin(12 * np.pi * x)
            
            heights = pattern.tolist()
            
            # Aggressive optimization
            def objective(h):
                return -compute_c2_fast(h)
            
            bounds = [(0, 15.0)] * len(heights)
            result = minimize(
                objective, 
                heights, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success:
                final_heights = result.x.tolist()
                final_c2 = -objective(final_heights)
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_heights = final_heights
                    
        except Exception:
            pass
    
    # Final fallback
    if best_heights is None:
        # Create a very robust pattern
        n_steps = 400
        heights = [1.5] * n_steps
        best_c2 = compute_c2_fast(heights)
        best_heights = heights
    
    eval_time = time.time() - start_time
    return best_heights, best_c2, eval_time

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses hybrid approach with mathematical patterns and optimization.
    """
    try:
        heights, c2, eval_time = hybrid_optimization_approach()
        # print(f"Optimized C2: {c2}, Benchmark Ratio: {c2/0.962:.4f}, Time: {eval_time:.4f}s")
        return heights
    except Exception as e:
        # Fallback to simple construction if optimization fails
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
