# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time
from typing import List

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Create step function on [-1/4, 1/4] with equal spacing
    n = len(f)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Define the domain
    x = np.linspace(-0.25, 0.25, 2*n)  # Double resolution for better convolution
    
    # Create piecewise constant function
    step_width = 0.5 / n
    g = np.zeros(len(x))
    
    # Compute autoconvolution using discrete convolution
    # We use the fact that f*f is the convolution of f with itself
    g = np.convolve(f, f, mode='full')
    
    # Trim to proper size (we want the convolution on the same interval)
    g = g[len(f)-1 : 2*len(f)-1]
    
    # Compute norms
    # ||g||₂² (L2 norm squared)
    g_squared = g * g
    # Using trapezoidal rule approximation for integral of g²
    norm_g2_squared = np.sum((g_squared[:-1] + g_squared[1:]) * step_width / 2)
    
    # ||g||₁ (L1 norm)
    norm_g1 = np.sum(np.abs(g)) * step_width
    
    # ||g||∞ (Infinity norm)
    norm_ginf = np.max(np.abs(g))
    
    return norm_g2_squared, norm_g1, norm_ginf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C2 for given step function values.
    Returns negative C2 since we want to maximize it.
    """
    try:
        norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_g1 <= 1e-12 or norm_ginf <= 1e-12:
            return -1e10  # Very poor score
            
        c2 = norm_g2_sq / (norm_g1 * norm_ginf)
        return c2
    except Exception:
        return -1e10

def construct_function() -> List[float]:
    """
    Construct step function using evolutionary optimization to maximize C2.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Start with a simple configuration and evolve it
    initial_guess = [1.0] * 50  # Start with uniform distribution
    
    # Optimization parameters
    bounds = [(0.0, 10.0)] * 50  # Bounds for step heights
    
    # Use differential evolution for global optimization
    result = differential_evolution(
        lambda x: -evaluate_c2(x.tolist()),  # Minimize negative C2
        bounds,
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=False
    )
    
    # Return optimized step heights
    best_solution = result.x.tolist()
    
    # Ensure non-negativity
    best_solution = [max(0.0, val) for val in best_solution]
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
