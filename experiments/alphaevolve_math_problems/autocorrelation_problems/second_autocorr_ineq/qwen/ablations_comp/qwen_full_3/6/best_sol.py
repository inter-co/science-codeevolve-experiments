# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution, minimize
import time
import random
from typing import List, Tuple

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = signal.convolve(f, f, mode='full')
    
    # The convolution result has length 2*n - 1
    # We want to extract the central portion that captures the main convolution effect
    n = len(f)
    center_idx = len(g) // 2
    
    # Take a reasonable central portion - this should be enough for most practical purposes
    # We'll take a bit more than n to get good convolution coverage
    half_width = min(2 * n - 1, len(g))
    start_idx = max(0, center_idx - half_width // 2)
    end_idx = min(len(g), center_idx + half_width // 2)
    
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

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given step function values"""
    try:
        g_l2_squared, g_l1, g_linf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_l1 <= 1e-15 or g_linf <= 1e-15:
            return 0.0
            
        c2 = g_l2_squared / (g_l1 * g_linf)
        return c2
    except Exception:
        return 0.0

def create_high_performance_patterns(n: int) -> List[List[float]]:
    """
    Create high-performance initial patterns specifically designed for maximizing C2
    Based on mathematical analysis of what leads to optimal convolution properties
    """
    patterns = []
    
    # Pattern 1: Optimized central peak with oscillatory tail
    x = np.linspace(-1, 1, n)
    # Strong central peak with controlled oscillation for better convolution
    central_peak = np.exp(-0.5 * x**2) * 2.5
    oscillation = 0.4 * np.sin(4 * np.pi * x) * np.exp(-0.5 * x**2)
    pattern1 = np.maximum(0, central_peak + oscillation)
    patterns.append(pattern1.tolist())
    
    # Pattern 2: Multi-peak with strategic spacing for constructive interference
    pattern2 = np.zeros(n)
    centers = [n//4, n//2, 3*n//4]
    for center in centers:
        sigma = n // 12
        x = np.arange(n) - center
        pattern2 += np.exp(-0.5 * (x / sigma)**2) * 1.8
    patterns.append(pattern2.tolist())
    
    # Pattern 3: High-contrast with sharp transitions (better for convolution energy spread)
    pattern3 = np.ones(n)
    center_start = n//3
    center_end = 2*n//3
    pattern3[center_start:center_end] = 3.5
    patterns.append(pattern3.tolist())
    
    # Pattern 4: Double peak with controlled amplitude difference
    pattern4 = np.zeros(n)
    center1 = n//3
    center2 = 2*n//3
    sigma = n // 10
    x = np.arange(n)
    pattern4 += np.exp(-0.5 * ((x - center1) / sigma)**2) * 2.8
    pattern4 += np.exp(-0.5 * ((x - center2) / sigma)**2) * 2.0
    patterns.append(pattern4.tolist())
    
    # Pattern 5: Smooth bell curve with enhanced edges
    x = np.linspace(-1, 1, n)
    gaussian = np.exp(-0.5 * x**2) * 2.0
    # Add edge enhancement
    edge_enhancement = 0.5 * (1 + np.tanh(3 * x))
    pattern5 = gaussian * edge_enhancement
    patterns.append(pattern5.tolist())
    
    return patterns

def advanced_local_optimization(initial_heights: List[float], n_steps: int) -> Tuple[List[float], float]:
    """
    Advanced local optimization with multiple refinement strategies
    """
    def local_objective(h):
        return -compute_c2(h)
    
    try:
        # First attempt: L-BFGS-B with tight tolerances
        bounds = [(0, 10.0)] * len(initial_heights)
        result = minimize(
            local_objective, 
            initial_heights, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if result.success:
            refined_heights = result.x.tolist()
            final_c2 = -local_objective(refined_heights)
            return refined_heights, final_c2
    except:
        pass
    
    # Second attempt: Simple gradient-free local search if optimization fails
    try:
        best_h = initial_heights.copy()
        best_c2 = compute_c2(best_h)
        
        # Gradient-free local search with small perturbations
        for _ in range(20):
            new_h = best_h.copy()
            for i in range(len(new_h)):
                if np.random.rand() < 0.5:  # 50% chance to perturb
                    # Add small random perturbation
                    perturbation = np.random.normal(0, 0.1)
                    new_h[i] = max(0, new_h[i] + perturbation)
            
            new_c2 = compute_c2(new_h)
            if new_c2 > best_c2:
                best_h = new_h
                best_c2 = new_c2
                
        return best_h, best_c2
    except:
        pass
    
    # Fallback to basic computation if everything fails
    final_c2 = compute_c2(initial_heights)
    return initial_heights, final_c2

def enhanced_hybrid_approach() -> Tuple[List[float], float]:
    """
    Enhanced hybrid approach that leverages multiple optimization strategies
    """
    start_time = time.time()
    max_time = 55  # Leave buffer for cleanup
    
    best_c2 = 0.0
    best_heights = None
    
    # Strategy 1: Comprehensive pattern testing with aggressive local refinement
    n_steps_list = [200, 250, 300]  # Larger sizes for better resolution
    
    for n_steps in n_steps_list:
        if time.time() - start_time > max_time:
            break
            
        patterns = create_high_performance_patterns(n_steps)
        
        for i, pattern in enumerate(patterns):
            if time.time() - start_time > max_time:
                break
                
            # Normalize pattern to reasonable scale
            if np.max(pattern) > 0:
                normalized_pattern = [x / np.max(pattern) * 4.0 for x in pattern]
            else:
                normalized_pattern = pattern
                
            # Use a representative subset of the pattern
            heights = normalized_pattern[:min(len(normalized_pattern), n_steps//2)]
            if len(heights) < 10:  # Minimum size requirement
                continue
                
            # Aggressive local optimization
            refined_heights, c2 = advanced_local_optimization(heights, n_steps)
            
            if c2 > best_c2:
                best_c2 = c2
                best_heights = refined_heights.copy()
    
    # Strategy 2: Global optimization approach for final refinement
    if best_heights is None or best_c2 <= 0:
        if time.time() - start_time < max_time:
            # Use a more sophisticated optimization approach
            n_steps = 250
            # Start with a more informed initial guess
            initial_heights = [2.0] * (n_steps // 4)
            
            # Try different optimization approaches
            try:
                def objective(x):
                    return -compute_c2(x)
                
                bounds = [(0, 10.0)] * len(initial_heights)
                # Try with different settings for better convergence
                result = minimize(
                    objective, 
                    initial_heights, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    final_heights = result.x.tolist()
                    final_c2 = -objective(final_heights)
                    if final_c2 > best_c2:
                        best_c2 = final_c2
                        best_heights = final_heights
            except:
                pass
    
    # Strategy 3: Final fallback with proven patterns
    if best_heights is None:
        # Create a highly optimized pattern based on known good characteristics
        n_final = 200
        x = np.linspace(-1, 1, n_final)
        # Very concentrated peak with oscillatory component
        final_pattern = np.exp(-0.5 * x**2) * 3.0
        oscillation = 0.5 * np.sin(5 * np.pi * x) * np.exp(-0.5 * x**2)
        final_pattern = np.maximum(0, final_pattern + oscillation)
        
        # Convert to heights
        heights = final_pattern[:n_final//2].tolist()
        best_c2 = compute_c2(heights)
        best_heights = heights
    
    return best_heights, best_c2

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value.
    Enhanced approach with better patterns and optimization strategies.
    """
    try:
        heights, c2 = enhanced_hybrid_approach()
        return heights
    except Exception as e:
        # Fallback to simple construction if optimization fails
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
