# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal, optimize
import random
import warnings
warnings.filterwarnings('ignore')

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation using the exact method specified.
    f_values: step function heights on equally spaced points in [-1/4, 1/4]
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Create the step function on [-1/4, 1/4] with proper spacing
    n = len(f)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Compute autoconvolution using fast convolution
    g = signal.convolve(f, f, mode='full')
    
    # Compute norms according to exact specification:
    # ||g||_2^2 = sum of (dx/3)(y1^2 + y1*y2 + y2^2) over consecutive pairs
    # ||g||_1 = sum(|g|) / (len(g) + 1)
    # ||g||_inf = max(|g|)
    
    if len(g) <= 1:
        g_norm_2_sq = 0.0
        g_norm_1 = 0.0
        g_norm_inf = 0.0 if len(g) == 0 else abs(g[0])
    else:
        # Compute norms using the exact evaluator method
        g_norm_2_sq = 0.0
        g_norm_1 = 0.0
        g_norm_inf = 0.0
        
        # Estimate dx based on domain and number of points
        # Domain is [-1/4, 1/4] = 1/2 width  
        # For n points in original function, convolution gives 2n-1 points
        dx = 0.5 / (len(g) - 1) if len(g) > 1 else 0.5
        
        # Compute ||g||_1
        g_norm_1 = np.sum(np.abs(g)) / (len(g) + 1)
        
        # Compute ||g||_inf
        g_norm_inf = np.max(np.abs(g))
        
        # Compute ||g||_2^2 using trapezoidal-like piecewise integration
        # Each segment contributes (dx/3)(g[i]^2 + g[i]*g[i+1] + g[i+1]^2)
        for i in range(len(g) - 1):
            g_norm_2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)
        
        # Add the last term for completeness (only g[i]^2 term)
        g_norm_2_sq += (dx/3) * (g[-1]**2)
    
    return g_norm_2_sq, g_norm_1, g_norm_inf

def compute_c2(f_values: list[float]) -> float:
    """Compute C2 value for given step function."""
    g_norm_2_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
        return 0.0
    
    c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
    return c2

def construct_function() -> list[float]:
    """
    Continuous optimization approach using gradient-based methods.
    This approach treats the step function as a continuous function 
    and applies optimization techniques to maximize C2.
    """
    # Parameters for the optimization
    n = 200  # Number of discrete points (smaller for faster optimization)
    max_iter = 1000  # Maximum iterations for optimization
    
    # Start with a good initial guess based on theoretical considerations
    # A flat distribution often works well for maximizing C2
    initial_guess = np.ones(n) * 0.5
    
    # Use a smooth approximation of the step function
    # We'll optimize the parameters of a smooth function that approximates
    # the desired step function
    
    def smooth_step_function(params):
        """Create a smooth approximation of a step function with given parameters"""
        # params contains the heights at different positions
        # We'll use a sigmoid-based approach to create smooth transitions
        x = np.linspace(-0.25, 0.25, n)
        result = np.zeros_like(x)
        
        # Create a combination of smooth peaks
        for i in range(0, len(params), 2):
            if i + 1 < len(params):
                center = params[i]
                height = params[i+1]
                # Gaussian-like peak centered at 'center' with height 'height'
                result += height * np.exp(-((x - center) ** 2) * 10)
        
        # Ensure non-negative values and normalize
        result = np.maximum(result, 0)
        # Normalize to prevent overly large values
        if np.max(result) > 0:
            result = result / np.max(result) * 0.8
        
        return result
    
    def objective_function(params):
        """Objective function to maximize C2"""
        try:
            # Create step function from parameters
            f_vals = smooth_step_function(params)
            
            # Clip to valid range [0, 1]
            f_vals = np.clip(f_vals, 0, 1)
            
            # Compute C2
            c2_val = compute_c2(f_vals.tolist())
            
            # Return negative because we're minimizing in scipy
            return -c2_val
        except Exception:
            # Return large negative value for invalid configurations
            return 1e10
    
    # Optimize using L-BFGS-B which handles bounds well
    # Initialize with some reasonable parameters
    bounds = [(0, 1) for _ in range(n)]  # Bounds for each parameter
    
    # Create a smarter initial parameter set
    initial_params = []
    # Start with a few peaks
    num_peaks = min(10, n // 5)
    for i in range(num_peaks):
        center = (i / (num_peaks - 1) - 0.5) * 0.5 if num_peaks > 1 else 0
        height = 0.8 + 0.2 * random.random()
        initial_params.extend([center, height])
    
    # Pad to full length
    while len(initial_params) < n:
        initial_params.append(0.5)
    
    # Truncate to match expected length
    initial_params = initial_params[:n]
    
    try:
        # Use scipy's minimize with bounds
        result = optimize.minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(len(initial_params))],
            options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6},
            callback=None
        )
        
        # Extract the best solution
        if result.success:
            final_params = result.x
            final_f = smooth_step_function(final_params)
            final_f = np.clip(final_f, 0, 1)
            return final_f.tolist()
        else:
            # Fallback to simpler approach if optimization fails
            pass
    except Exception:
        # If optimization fails, fall back to a simpler approach
        pass
    
    # Fallback: Use a simple but effective approach with specific patterns
    # Based on known good patterns from previous work
    x = np.linspace(-0.25, 0.25, n)
    
    # Try a symmetric pattern that often works well
    # This is a modification of the best-known pattern from literature
    pattern = 0.6 * np.exp(-x**2 * 4) + 0.3 * np.exp(-((x - 0.15)**2) * 8) + 0.3 * np.exp(-((x + 0.15)**2) * 8)
    
    # Add some noise to avoid local minima issues
    noise = 0.02 * (np.random.random(n) - 0.5)
    pattern += noise
    
    # Clip to valid range
    pattern = np.clip(pattern, 0, 1)
    
    # Refinement using local search
    refined_pattern = pattern.copy()
    best_c2 = compute_c2(refined_pattern.tolist())
    
    # Local search around the current solution
    for _ in range(500):
        # Make small perturbations
        idx = random.randint(0, n - 1)
        delta = random.uniform(-0.05, 0.05)
        old_val = refined_pattern[idx]
        refined_pattern[idx] = max(0, min(1, old_val + delta))
        
        new_c2 = compute_c2(refined_pattern.tolist())
        if new_c2 > best_c2:
            best_c2 = new_c2
        else:
            # Revert if worse
            refined_pattern[idx] = old_val
    
    return refined_pattern.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
