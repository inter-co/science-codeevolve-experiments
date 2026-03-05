# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Create step function on [-1/4, 1/4] with appropriate spacing
    n = len(f)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Define domain for step function
    x_domain = np.linspace(-0.25, 0.25, n, endpoint=False)
    dx = x_domain[1] - x_domain[0] if n > 1 else 0.5
    
    # Compute autoconvolution using discrete convolution
    # This gives us g = f * f
    g = signal.convolve(f, f, mode='full')
    
    # Adjust indices for proper domain mapping
    # The convolution result has length 2*n - 1
    # We need to map back to [-0.5, 0.5] domain appropriately
    g_len = len(g)
    g_domain = np.linspace(-0.5, 0.5, g_len, endpoint=False)
    
    # For our purposes, we'll focus on the central region [-0.25, 0.25]
    # Find indices corresponding to this region
    center_idx = g_len // 2
    half_width = n - 1  # Number of points in the central region
    start_idx = center_idx - half_width
    end_idx = center_idx + half_width + 1
    
    # Extract central portion
    g_center = g[start_idx:end_idx]
    
    # Compute norms
    # ||g||₂² = sum(g[i]²) * dx (approximate integral)
    g_squared = g_center ** 2
    g_norm_2_sq = np.sum(g_squared) * dx
    
    # ||g||₁ = sum(|g[i]|) * dx 
    g_norm_1 = np.sum(np.abs(g_center)) * dx
    
    # ||g||∞ = max(|g[i]|)
    g_norm_inf = np.max(np.abs(g_center))
    
    return g_norm_2_sq, g_norm_1, g_norm_inf

def evaluate_c2(f_values: list[float]) -> float:
    """
    Evaluate C2 = ||g||₂² / (||g||₁ · ||g||∞)
    """
    try:
        g_norm_2_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
            return 0.0
            
        c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
        return c2
    except Exception:
        return 0.0

def evolve_step_function() -> list[float]:
    """
    Evolutionary approach to construct step function with high C2 value.
    Uses a hybrid strategy combining local search and global optimization.
    """
    # Start with a good baseline: a simple symmetric function
    n_initial = 100
    f_initial = [0.0] * n_initial
    
    # Create a more structured initial population
    # Try to create something that will give high C2
    for i in range(n_initial):
        # Simple bell-shaped pattern might work well
        x = (i - n_initial/2) / (n_initial/2)
        f_initial[i] = max(0, 1 - abs(x)**2)
    
    # Normalize so that the total area isn't too large
    total_area = sum(f_initial)
    if total_area > 0:
        f_initial = [val/total_area for val in f_initial]
    
    # Apply optimization using differential evolution
    # This is a global optimization method that works well for this kind of problem
    bounds = [(0.0, 1.0)] * n_initial  # All values between 0 and 1
    
    # Use a custom objective function (minimize negative C2)
    def objective(x):
        return -evaluate_c2(list(x))
    
    # Run optimization
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        # Return best found solution
        optimized_f = list(result.x)
        
        # Ensure non-negativity
        optimized_f = [max(0.0, val) for val in optimized_f]
        
        return optimized_f
    except:
        # Fallback to initial function if optimization fails
        return f_initial

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Time-limited optimization approach
    start_time = time.time()
    
    # Try several approaches and return the best one
    best_c2 = 0.0
    best_function = []
    
    # Approach 1: Optimized bell curve
    try:
        f1 = [max(0, 1 - (i - 50)**2/2500) for i in range(100)]
        f1 = [val/sum(f1) for val in f1]  # normalize
        c2_1 = evaluate_c2(f1)
        if c2_1 > best_c2:
            best_c2 = c2_1
            best_function = f1
    except:
        pass
    
    # Approach 2: Optimized step function
    try:
        f2 = [0.0] * 100
        # Put some mass in the middle
        for i in range(30, 70):
            f2[i] = 1.0
        f2 = [val/sum(f2) for val in f2]  # normalize
        c2_2 = evaluate_c2(f2)
        if c2_2 > best_c2:
            best_c2 = c2_2
            best_function = f2
    except:
        pass
    
    # Approach 3: Evolutionary optimization
    try:
        f3 = evolve_step_function()
        c2_3 = evaluate_c2(f3)
        if c2_3 > best_c2:
            best_c2 = c2_3
            best_function = f3
    except:
        pass
    
    # Return the best function found
    if not best_function:
        # Fallback to a simple symmetric function
        f = [0.0] * 100
        for i in range(50):
            f[i] = 1.0 - abs(i - 49.5)/50
        f = [val/sum(f) for val in f]
        return f
    
    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
