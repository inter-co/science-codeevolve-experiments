# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Create step function on [-1/4, 1/4] with given heights
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Define the domain
    x_domain = np.linspace(-0.25, 0.25, 2*n+1)  # finer grid for better accuracy
    dx = x_domain[1] - x_domain[0]
    
    # Create step function (piecewise constant)
    f = np.zeros_like(x_domain)
    step_width = 0.5 / n  # width of each step
    for i in range(n):
        start_idx = int(i * 2 + 1)  # offset to center steps
        end_idx = int((i + 1) * 2 + 1)
        f[start_idx:end_idx] = f_values[i]
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = signal.convolve(f, f, mode='full')
    
    # Adjust for proper scaling - need to account for dx in integration
    g = g * dx
    
    # Compute norms
    g_abs = np.abs(g)
    
    # ||g||₂² = ∫|g|² dx ≈ sum(|g|² * dx)  
    norm_g2_squared = np.sum(g_abs**2) * dx
    
    # ||g||₁ = ∫|g| dx ≈ sum(|g| * dx)
    norm_g1 = np.sum(g_abs) * dx
    
    # ||g||∞ = max |g|
    norm_ginf = np.max(g_abs)
    
    return norm_g2_squared, norm_g1, norm_ginf

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given step function"""
    norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_g1 <= 1e-15 or norm_ginf <= 1e-15:
        return 0.0
    
    c2 = norm_g2_sq / (norm_g1 * norm_ginf)
    return c2

def construct_function() -> List[float]:
    """
    Evolved approach using evolutionary optimization to find optimal step function
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Use a hybrid approach: start with some structured patterns
    # and then optimize using evolutionary algorithm
    
    # Initialize with a few promising patterns
    initial_patterns = [
        # Simple symmetric pattern
        [1.0] * 20,
        # Gaussian-like pattern (more concentrated in center)
        [0.5, 0.8, 1.0, 1.0, 0.8, 0.5],
        # Uniform pattern
        [0.5] * 15,
        # Spike pattern
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]
    
    best_c2 = 0.0
    best_pattern = []
    
    # Try several initial patterns and optimize each
    for pattern in initial_patterns:
        # Add some noise to create variations
        varied_pattern = [max(0.0, val + np.random.normal(0, 0.1)) for val in pattern]
        
        # Optimization using differential evolution
        try:
            # Create bounds for each parameter (0.0 to 2.0)
            bounds = [(0.0, 2.0) for _ in range(len(varied_pattern))]
            
            # Define objective function (we want to maximize C2, so minimize negative C2)
            def objective(params):
                return -compute_c2(list(params))
            
            # Run optimization
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=10,
                seed=42,
                disp=False
            )
            
            if result.success:
                optimized_pattern = list(result.x)
                c2_val = compute_c2(optimized_pattern)
                if c2_val > best_c2:
                    best_c2 = c2_val
                    best_pattern = optimized_pattern
                    
        except Exception:
            # Fallback to simple pattern
            c2_val = compute_c2(varied_pattern)
            if c2_val > best_c2:
                best_c2 = c2_val
                best_pattern = varied_pattern
    
    # Final refinement with a simple hill climbing approach
    if len(best_pattern) > 0:
        # Perform local search around the best solution
        current_pattern = best_pattern.copy()
        current_c2 = compute_c2(current_pattern)
        
        # Try small perturbations
        for _ in range(20):
            # Make small random changes
            new_pattern = current_pattern.copy()
            idx = random.randint(0, len(new_pattern)-1)
            new_pattern[idx] = max(0.0, new_pattern[idx] + np.random.normal(0, 0.1))
            
            new_c2 = compute_c2(new_pattern)
            if new_c2 > current_c2:
                current_pattern = new_pattern
                current_c2 = new_c2
        
        return current_pattern
    
    # Fallback to simple symmetric pattern if nothing works
    return [1.0] * 25

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
