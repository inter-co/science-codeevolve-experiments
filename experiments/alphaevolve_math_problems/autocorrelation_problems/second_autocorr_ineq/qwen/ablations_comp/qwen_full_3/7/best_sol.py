# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution, minimize
import time
from typing import List, Tuple

def compute_autoconvolution_norms(f_values: List[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = signal.convolve(f, f, mode='full')
    
    # For a function of length n, the convolution produces 2*n-1 elements
    # We want the central portion that represents the main convolution
    n = len(f)
    center = len(g) // 2
    
    # Extract the central portion that corresponds to meaningful convolution
    # Take the central part that captures the essential convolution behavior
    half_range = n  # This should be sufficient for most purposes
    start_idx = max(0, center - half_range + 1)
    end_idx = min(len(g), center + half_range)
    
    g_center = g[start_idx:end_idx]
    
    # Compute the three norms
    g_abs = np.abs(g_center)
    
    # ||g||₂² = sum(g²) * dx (using trapezoidal rule approximation)
    # We'll approximate the integral with the sum of squares
    if len(g_abs) >= 2:
        # Trapezoidal rule weights
        weights = np.ones_like(g_abs)
        weights[0] = 0.5
        weights[-1] = 0.5
        norm_g2_squared = np.sum(weights * (g_abs**2)) / len(g_abs)
    else:
        norm_g2_squared = np.sum(g_abs**2) / len(g_abs) if len(g_abs) > 0 else 0.0
    
    # ||g||₁ = sum(|g|) * dx  
    if len(g_abs) >= 2:
        weights = np.ones_like(g_abs)
        weights[0] = 0.5
        weights[-1] = 0.5
        norm_g1 = np.sum(weights * g_abs) / len(g_abs)
    else:
        norm_g1 = np.sum(g_abs) / len(g_abs) if len(g_abs) > 0 else 0.0
    
    # ||g||∞ = max(|g|)
    norm_ginf = np.max(g_abs)
    
    # Handle numerical edge cases
    if norm_g1 < 1e-15:
        norm_g1 = 1e-15
    if norm_ginf < 1e-15:
        norm_ginf = 1e-15
        
    return norm_g2_squared, norm_g1, norm_ginf

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

def create_advanced_patterns(n: int) -> List[Tuple[str, List[float]]]:
    """
    Create advanced initial patterns designed to maximize C2
    """
    patterns = []
    
    # Pattern 1: Optimized Gaussian with multiple peaks
    x = np.linspace(-1, 1, n)
    gaussian_pattern = np.exp(-0.5 * x**2)
    # Add a secondary peak to encourage broader convolution
    gaussian_pattern += 0.3 * np.exp(-0.5 * ((x - 0.4)**2)) 
    gaussian_pattern += 0.3 * np.exp(-0.5 * ((x + 0.4)**2))
    patterns.append(('multi_gaussian', gaussian_pattern.tolist()))
    
    # Pattern 2: Multi-peak pattern (inspired by good solutions)
    multi_peak = np.zeros(n)
    centers = [n//4, n//2, 3*n//4]
    for center in centers:
        sigma = n // 10
        x = np.arange(n) - center
        multi_peak += np.exp(-0.5 * (x / sigma)**2)
    patterns.append(('multi_peak', multi_peak.tolist()))
    
    # Pattern 3: Sine-based pattern with constructive interference
    sine_pattern = 1.0 + 0.5 * np.sin(2 * np.pi * np.linspace(0, 2, n))
    patterns.append(('sine', sine_pattern.tolist()))
    
    # Pattern 4: Exponential pattern with peak in center
    exp_pattern = np.exp(-2 * np.abs(np.linspace(-1, 1, n)))
    patterns.append(('exp', exp_pattern.tolist()))
    
    # Pattern 5: Piecewise constant with strategic peaks
    piecewise = np.ones(n)
    piecewise[n//3:n//2] = 2.0
    piecewise[n//2:2*n//3] = 2.0
    patterns.append(('piecewise', piecewise.tolist()))
    
    # Pattern 6: Central peak with symmetric tails
    x = np.linspace(-1, 1, n)
    central_peak = np.exp(-0.5 * x**2) * 2.0
    oscillation = 0.3 * np.sin(3 * np.pi * x)
    pattern6 = np.maximum(0, central_peak + oscillation)
    patterns.append(('central_peak', pattern6.tolist()))
    
    # Pattern 7: Modified version of AlphaEvolve's approach
    # Create a smoother pattern that avoids sharp transitions
    x = np.linspace(-0.5, 0.5, n)
    alpha_pattern = np.exp(-4 * x**2) * (1 + 0.5 * np.sin(8 * np.pi * x))
    patterns.append(('alpha_like', alpha_pattern.tolist()))
    
    return patterns

def adaptive_optimization_approach() -> Tuple[List[float], float]:
    """
    Adaptive optimization approach that tries multiple strategies
    """
    # Set up different optimization strategies
    best_c2 = 0.0
    best_heights = None
    
    # Strategy 1: Try multiple advanced patterns with local refinement
    n_steps_list = [200, 300, 400]  # Different resolutions
    
    for n_steps in n_steps_list:
        patterns = create_advanced_patterns(n_steps)
        
        for pattern_name, pattern in patterns:
            # Normalize pattern to reasonable values
            if np.max(pattern) > 0:
                normalized_pattern = [x / np.max(pattern) * 3.0 for x in pattern]
            else:
                normalized_pattern = pattern
                
            # Take subset to match reasonable number of heights
            heights = normalized_pattern[:min(len(normalized_pattern), n_steps//2)]
            if len(heights) == 0:
                continue
                
            # Try local optimization with L-BFGS-B for refinement
            try:
                def local_objective(h):
                    # Create function with specified heights
                    return -compute_c2(h)
                
                # Use L-BFGS-B for local refinement
                bounds = [(0, 10.0)] * len(heights)
                result = minimize(
                    local_objective, 
                    heights, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                if result.success:
                    final_heights = result.x.tolist()
                    final_c2 = -local_objective(final_heights)
                    
                    if final_c2 > best_c2:
                        best_c2 = final_c2
                        best_heights = final_heights
                        
            except Exception:
                continue
    
    # Strategy 2: If nothing worked well, try global optimization with reduced search space
    if best_heights is None or best_c2 <= 0:
        # Reduce search space for speed and focus on promising regions
        n_steps = 250
        initial_heights = [1.5] * (n_steps // 4)  # More concentrated initial heights
        
        # Direct approach with less complex optimization
        try:
            def objective(x):
                return -compute_c2(x)
            
            bounds = [(0, 10.0)] * len(initial_heights)
            result = minimize(
                objective, 
                initial_heights, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                final_heights = result.x.tolist()
                final_c2 = -objective(final_heights)
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_heights = final_heights
        except Exception:
            pass
    
    # Fallback to simple pattern if everything failed
    if best_heights is None:
        best_heights = [1.0] * 100
        best_c2 = compute_c2(best_heights)
    
    return best_heights, best_c2

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value.
    """
    try:
        heights, c2 = adaptive_optimization_approach()
        # print(f"Optimized C2: {c2}, Benchmark Ratio: {c2/0.962:.4f}")
        return heights
    except Exception as e:
        # Fallback to simple construction if optimization fails
        return [0.5] * 50

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
