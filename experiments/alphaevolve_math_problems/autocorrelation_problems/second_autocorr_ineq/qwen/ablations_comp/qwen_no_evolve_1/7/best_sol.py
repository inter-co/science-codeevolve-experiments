# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import time

def compute_c2(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute C2 value for given step function values.
    Returns (c2, eval_time)
    """
    start_time = time.time()
    
    # Convert to numpy array
    f = np.array(f_values)
    
    # Define domain [-1/4, 1/4] with n points
    n = len(f)
    if n == 0:
        return 0.0, 0.0
    
    # Create step function with equal spacing
    x_domain = np.linspace(-0.25, 0.25, n)
    dx = x_domain[1] - x_domain[0]
    
    # Compute autoconvolution g = f * f
    # Using discrete convolution
    g = convolve(f, f, mode='full')
    
    # The convolution result has 2*n - 1 elements
    # We need to map back to appropriate domain
    # The support of g is [-0.5, 0.5], so we take center portion
    g_center = g[len(g)//2 - n//2 : len(g)//2 + n//2]
    
    # Compute norms
    # ||g||₂² (L2 norm squared)
    g_squared = g_center * g_center
    norm_g2_squared = np.sum(g_squared) * dx
    
    # ||g||₁ (L1 norm)
    norm_g1 = np.sum(np.abs(g_center)) * dx
    
    # ||g||∞ (L-infinity norm)
    norm_ginf = np.max(np.abs(g_center))
    
    # Compute C2
    if norm_g1 > 1e-12 and norm_ginf > 1e-12:
        c2 = norm_g2_squared / (norm_g1 * norm_ginf)
    else:
        c2 = 0.0
    
    eval_time = time.time() - start_time
    return c2, eval_time

def objective_function(params):
    """Objective function to maximize C2"""
    # params contains step heights
    # Clip negative values to 0
    f_values = np.clip(params, 0, None)
    
    # Normalize so that sum doesn't explode
    if np.sum(f_values) > 0:
        f_values = f_values / np.sum(f_values) * 100
    
    c2, _ = compute_c2(f_values)
    # Return negative because we're minimizing
    return -c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using optimization."""
    
    # Try different sizes to find good starting point
    best_c2 = 0.0
    best_params = []
    
    # Try several different lengths
    test_lengths = [50, 100, 200, 500]
    
    for n in test_lengths:
        # Initialize with some heuristic approach
        initial_guess = np.ones(n) * 0.1  # Start with small uniform values
        
        # Set bounds for each parameter (non-negative)
        bounds = [(0, 100) for _ in range(n)]
        
        try:
            # Use differential evolution for global optimization
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            if -result.fun > best_c2:
                best_c2 = -result.fun
                best_params = result.x
                
        except Exception as e:
            continue
    
    # Final refinement with local optimization if needed
    if len(best_params) > 0:
        # Clip and normalize
        final_params = np.clip(best_params, 0, None)
        if np.sum(final_params) > 0:
            final_params = final_params / np.sum(final_params) * 100
        return final_params.tolist()
    else:
        # Fallback to simple construction
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
