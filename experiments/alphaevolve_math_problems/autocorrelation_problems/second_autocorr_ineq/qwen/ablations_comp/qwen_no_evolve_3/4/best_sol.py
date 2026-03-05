# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time
from typing import List

def compute_autoconvolution(f_values: List[float]) -> np.ndarray:
    """Compute autoconvolution g = f * f"""
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute convolution using fft for efficiency
    # Using 'full' mode to get complete convolution
    g = signal.convolve(f, f, mode='full')
    
    # Since we're working with symmetric interval [-1/4, 1/4], 
    # we need to center the convolution properly
    # The convolution will be of length 2*n - 1
    return g

def compute_c2_norms(g: np.ndarray) -> tuple[float, float, float]:
    """Compute the three norms needed for C2 calculation"""
    # L2 norm squared (using trapezoidal approximation)
    # For piecewise linear integration, we use the formula:
    # integral of g^2 ≈ sum of (h/3)(g[i]^2 + g[i]*g[i+1] + g[i+1]^2) 
    # where h is the step size
    
    # We'll compute this numerically assuming unit step size for simplicity
    # In practice, for a proper implementation, we'd need to account for actual spacing
    
    # Compute L2 norm squared using trapezoidal rule
    g_squared = g**2
    # Trapezoidal rule approximation for integral of g^2
    # For a discrete sequence, this becomes sum of trapezoids
    # But we'll use a simpler approach: sum of squares weighted appropriately
    # For piecewise linear interpolation, we approximate with triangular elements
    # Simpler approach: sum of squares of values (this is a reasonable approximation)
    l2_sq = np.sum(g_squared)
    
    # L1 norm (sum of absolute values)
    l1 = np.sum(np.abs(g))
    
    # L-infinity norm (maximum absolute value)
    l_inf = np.max(np.abs(g))
    
    return l2_sq, l1, l_inf

def compute_c2_score(f_values: List[float]) -> float:
    """Compute C2 score for given step function values"""
    try:
        # Compute autoconvolution
        g = compute_autoconvolution(f_values)
        
        # Compute norms
        l2_sq, l1, l_inf = compute_c2_norms(g)
        
        # Avoid division by zero
        if l1 <= 1e-12 or l_inf <= 1e-12:
            return 0.0
            
        # Compute C2
        c2 = l2_sq / (l1 * l_inf)
        return c2
    except Exception as e:
        return 0.0

def evolve_step_function() -> List[float]:
    """
    Evolve a step function to maximize C2 using evolutionary algorithm
    """
    # Use a more sophisticated approach: start with a good initial guess
    # and optimize using differential evolution
    
    # First, let's create a reasonable initial population
    # Start with some known good patterns
    initial_patterns = [
        # Simple two-peak pattern
        [1.0, 0.0, 1.0] + [0.0] * 5,
        # Uniform distribution
        [0.5] * 10,
        # Gaussian-like shape
        [0.1, 0.3, 0.6, 0.8, 1.0, 0.8, 0.6, 0.3, 0.1] + [0.0] * 5,
        # Single spike
        [0.0] * 5 + [1.0] + [0.0] * 5,
    ]
    
    best_c2 = 0.0
    best_f = []
    
    # Try different patterns
    for pattern in initial_patterns:
        # Add noise to make it more diverse
        noisy_pattern = [max(0, x + np.random.normal(0, 0.1)) for x in pattern]
        # Ensure non-negative values
        noisy_pattern = [max(0, x) for x in noisy_pattern]
        
        # Evaluate this pattern
        c2 = compute_c2_score(noisy_pattern)
        if c2 > best_c2:
            best_c2 = c2
            best_f = noisy_pattern.copy()
    
    # Now optimize using differential evolution
    # Define bounds for each parameter
    n_steps = len(best_f)
    bounds = [(0, 2.0) for _ in range(n_steps)]
    
    def objective(x):
        # Minimize negative C2 (since we want to maximize C2)
        return -compute_c2_score(list(x))
    
    # Run optimization
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42
        )
        
        optimized_f = list(result.x)
        # Clip negative values
        optimized_f = [max(0, x) for x in optimized_f]
        
        final_c2 = compute_c2_score(optimized_f)
        if final_c2 > best_c2:
            best_c2 = final_c2
            best_f = optimized_f
    except:
        pass  # If optimization fails, keep the best found so far
    
    return best_f

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Start with a larger number of steps for better resolution
    n_steps = 200
    
    # Create initial function using our evolutionary approach
    f_values = evolve_step_function()
    
    # Make sure we have the right number of steps
    if len(f_values) < n_steps:
        # Pad with zeros
        f_values.extend([0.0] * (n_steps - len(f_values)))
    elif len(f_values) > n_steps:
        # Truncate
        f_values = f_values[:n_steps]
    
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
