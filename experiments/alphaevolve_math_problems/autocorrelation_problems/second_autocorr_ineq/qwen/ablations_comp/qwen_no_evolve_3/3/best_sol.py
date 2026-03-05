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
    
    # Create step function on [-1/4, 1/4] with equal spacing
    n_steps = len(f)
    if n_steps == 0:
        return 0.0, 0.0, 0.0
    
    # Define the domain [-1/4, 1/4]
    x_domain = np.linspace(-0.25, 0.25, n_steps)
    dx = x_domain[1] - x_domain[0] if len(x_domain) > 1 else 1.0
    
    # Compute autoconvolution g = f * f using discrete convolution
    # We'll use the full convolution and then extract the relevant part
    g = convolve(f, f, mode='full')
    
    # The convolution result has length 2*n_steps - 1
    # We want to evaluate on the same domain as original function
    # But since we're looking for optimal step functions, we'll work with 
    # the autoconvolution directly
    
    # Get the middle portion that corresponds to the autoconvolution
    mid = len(g) // 2
    g_centered = g[mid - n_steps + 1 : mid + n_steps]
    
    # Compute norms
    g_squared = g_centered ** 2
    g_abs = np.abs(g_centered)
    
    # ||g||₂² (L2 norm squared)
    # Using trapezoidal rule approximation for continuous case
    # For discrete points with equal spacing, we approximate integral
    # Since we have n_points, we have n_points-1 intervals
    # We use Simpson's rule-like approximation for better accuracy
    if len(g_squared) >= 2:
        # Trapezoidal rule for ||g||₂²
        norm_2_sq = np.sum((g_squared[:-1] + g_squared[1:]) * dx / 2)
    else:
        norm_2_sq = 0.0 if len(g_squared) == 0 else g_squared[0] * dx
    
    # ||g||₁ (L1 norm)  
    # Approximate as sum of absolute values divided by number of intervals
    if len(g_abs) >= 2:
        norm_1 = np.sum((g_abs[:-1] + g_abs[1:]) * dx / 2)
    else:
        norm_1 = 0.0 if len(g_abs) == 0 else g_abs[0] * dx
    
    # ||g||∞ (infinity norm)
    norm_inf = np.max(np.abs(g_centered)) if len(g_centered) > 0 else 0.0
    
    return norm_2_sq, norm_1, norm_inf

def compute_c2(f_values: list[float]) -> float:
    """Compute C2 value for given step function."""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    return norm_2_sq / (norm_1 * norm_inf)

def objective_function(f_values: list[float]) -> float:
    """
    Objective function to minimize (we negate because we want to maximize C2).
    """
    return -compute_c2(f_values)

def evolve_step_function() -> list[float]:
    """
    Evolve a step function using a global optimization approach.
    This is a more sophisticated strategy than random generation.
    """
    # Use a hybrid approach: start with some heuristic initialization
    # and then optimize using differential evolution
    
    # Initial guess: try some structured patterns that might work well
    initial_patterns = [
        # Simple symmetric pattern
        [1.0] * 20,
        # Gaussian-like pattern
        [np.exp(-i**2/100) for i in range(20)],
        # Uniform pattern
        [0.5] * 20,
        # Single peak
        [0.0] * 10 + [1.0] + [0.0] * 10,
        # Double peak
        [0.0] * 5 + [0.5] + [0.0] * 5 + [0.5] + [0.0] * 5,
    ]
    
    best_c2 = -float('inf')
    best_solution = []
    
    # Try different initial patterns
    for pattern in initial_patterns:
        # Use differential evolution with bounds
        bounds = [(0.0, 2.0) for _ in range(len(pattern))]
        
        try:
            # Run optimization
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                disp=False,
                tol=1e-6
            )
            
            # Check if this is better
            current_c2 = -result.fun
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_solution = result.x.tolist()
                
        except Exception:
            continue
    
    # If we didn't find anything good, fall back to a reasonable pattern
    if not best_solution:
        # Try a more systematic approach
        n_steps = 50
        # Create a smooth decreasing pattern that might yield good results
        f_values = [max(0, 1.0 - i/n_steps) for i in range(n_steps)]
        return f_values
    
    # Clip negative values and normalize
    best_solution = [max(0.0, val) for val in best_solution]
    return best_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Use evolutionary approach to find good step function
    return evolve_step_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
