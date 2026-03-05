# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import time

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f
    # Using 'full' mode to get complete convolution
    g = convolve(f, f, mode='full')
    
    # Since we're working with symmetric functions on [-1/4, 1/4], 
    # we only need the middle portion corresponding to the full support
    mid_idx = len(g) // 2
    half_len = len(f)
    g = g[mid_idx - half_len + 1:mid_idx + half_len]
    
    # Compute norms
    # ||g||₂² = sum(g²) * step_size
    # We approximate step size as 1/len(f) since domain is [-1/4, 1/4] 
    # and we have len(f) steps
    step_size = 0.5 / len(f)
    g_squared = g**2
    norm_g_2_squared = np.sum(g_squared) * step_size
    
    # ||g||₁ = sum(|g|) * step_size
    norm_g_1 = np.sum(np.abs(g)) * step_size
    
    # ||g||∞ = max(|g|)
    norm_g_infinity = np.max(np.abs(g))
    
    return norm_g_2_squared, norm_g_1, norm_g_infinity

def evaluate_c2(f_values: list[float]) -> float:
    """Evaluate C2 for given step function values"""
    try:
        norm_g_2_squared, norm_g_1, norm_g_infinity = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_g_1 <= 1e-15 or norm_g_infinity <= 1e-15:
            return 0.0
            
        c2 = norm_g_2_squared / (norm_g_1 * norm_g_infinity)
        return c2
    except Exception:
        return 0.0

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using evolutionary optimization."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Use evolutionary optimization to find better step functions
    # Start with a reasonable initial guess
    initial_guess = [0.5] * 100  # Uniform distribution as starting point
    
    # Define bounds for each parameter (step height)
    bounds = [(0.0, 1.0) for _ in range(len(initial_guess))]
    
    # Use differential evolution for global optimization
    result = differential_evolution(
        lambda x: -evaluate_c2(x.tolist()),  # Negative because we want to maximize
        bounds,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=False
    )
    
    # Return the optimized solution
    best_solution = result.x.tolist()
    
    # Ensure all values are non-negative
    best_solution = [max(0.0, val) for val in best_solution]
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
