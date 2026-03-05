# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
import time

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using fast convolution
    # Using 'full' mode to get complete convolution result
    g = convolve(f, f, mode='full')
    
    # Adjust indices for symmetric case around zero
    # The convolution result has length 2*n - 1, centered at index n-1
    n = len(f)
    center_idx = n - 1
    
    # We only care about the part that corresponds to [-1/4, 1/4] interval
    # For simplicity, we'll work with the full convolution and then extract relevant parts
    # But since we're dealing with step functions on [-1/4, 1/4], 
    # the convolution will be defined on [-1/2, 1/2] which contains our domain
    
    # Compute norms
    g_squared = g**2
    g_abs = np.abs(g)
    
    # ||g||₂² (L2 norm squared)
    # Using trapezoidal rule approximation
    # Since we're working with discrete points, we approximate integral
    # with sum of areas of trapezoids formed by consecutive points
    # For simplicity, assume uniform spacing of 1/n over [-1/2, 1/2]
    # So spacing is 1/n, but we need to be more precise about the actual domain
    
    # Actually, let's reframe this properly:
    # Our step function f is defined on [-1/4, 1/4] with n steps
    # So step width is (1/2)/n = 1/(2n)
    # The convolution g will be defined on [-1/2, 1/2] with 2n-1 points
    # But we need to map this correctly to the original interval
    
    # Let's just compute directly using the definition:
    # ||g||₂² = sum of g[i]^2 * delta_x (where delta_x is step size)
    # For now, we'll use a simple approximation
    
    # Compute the norms directly
    # Assuming uniform spacing of 1/(2*n) for the domain [-1/2, 1/2]
    # But we're actually interested in the interval [-1/4, 1/4] so we need to be more careful
    
    # Simpler approach: compute the norms directly from the convolution result
    # The key insight is that we're looking at the convolution over the full domain,
    # but the important thing is to compute the norms correctly
    
    # For ||g||₂² using trapezoidal rule over the full convolution:
    # We'll compute sum of g[i]*g[i+1] for adjacent pairs, weighted by step size
    # But given the nature of convolution, we'll compute sum(g^2) * step_size
    # However, we should be more careful about the actual domain
    
    # Simplification: since we're optimizing over the space of step functions,
    # let's just compute what we can from the convolution
    
    # ||g||₂² (using trapezoidal rule)
    # We compute the area under g^2 using trapezoidal rule
    # Assume uniform spacing, but since we don't know exact spacing, 
    # we'll compute the sum of squares and normalize appropriately
    
    # ||g||₂² using trapezoidal rule
    # For a sequence y[0], y[1], ..., y[n-1] with equal spacing h,
    # ∫ y² dx ≈ h * [y[0]²/2 + y[1]² + ... + y[n-2]² + y[n-1]²/2]
    # But we're using discrete values, so we'll compute sum(y^2) * h
    # We'll assume unit spacing for now and correct later if needed
    
    # Let's compute norms directly from convolution result
    # Convolution result has length 2n - 1
    # The convolution g = f * f is defined on [-1/2, 1/2] with spacing 1/(2n)
    
    # For the purpose of C₂ computation, we can proceed as follows:
    # ||g||₂² = sum of squares of g values (we'll need to multiply by proper scaling)
    # ||g||₁ = sum of absolute values of g  
    # ||g||∞ = maximum absolute value of g
    
    # We'll compute these based on the convolution results
    g_squared_sum = np.sum(g**2)
    g_abs_sum = np.sum(np.abs(g))
    g_max = np.max(np.abs(g))
    
    # For normalization purposes, we need to account for the fact that
    # convolution spacing is related to the original step width
    # If original steps span [-1/4, 1/4] with n steps, step width is 1/(2n)
    # The convolution spans [-1/2, 1/2] with 2n-1 steps, step width is 1/(2n)
    
    # But for our purposes, we can just compute the norms directly
    # and trust that the optimization will handle the scaling appropriately
    
    return g_squared_sum, g_abs_sum, g_max

def compute_c2(f_values: list[float]) -> float:
    """Compute C₂ value for given step function heights."""
    try:
        g_squared_sum, g_abs_sum, g_max = compute_autoconvolution_norms(f_values)
        
        # Handle edge cases
        if g_abs_sum <= 1e-12 or g_max <= 1e-12:
            return 0.0
            
        # Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)
        c2 = g_squared_sum / (g_abs_sum * g_max)
        return c2
    except Exception as e:
        # In case of any numerical issues, return a very small value
        return 0.0

def construct_function() -> list[float]:
    """
    Construct step function with optimized C2 value using gradient-based optimization.
    """
    # Start with a good initial guess - a simple symmetric function
    n = 200  # Number of steps on [-1/4, 1/4]
    f_values = [0.0] * n
    
    # Initialize with a peak in the middle and decreasing tails
    # This is inspired by the optimal shape for maximizing C₂
    half_n = n // 2
    peak_height = 1.0
    
    # Create a bell-shaped distribution
    for i in range(n):
        # Create a Gaussian-like profile
        x = (i - half_n) / half_n  # Normalize to [-1, 1]
        f_values[i] = peak_height * np.exp(-x**2 * 2)
    
    # Normalize to prevent extreme values
    total_sum = sum(f_values)
    if total_sum > 0:
        f_values = [val / total_sum * 100 for val in f_values]
    
    # Apply gradient-based optimization
    # We'll use scipy's minimize function with bounds
    def objective(f_vals):
        return -compute_c2(f_vals)  # Negative because we want to maximize C2
    
    # Set bounds for each variable (non-negative)
    bounds = [(0, None) for _ in range(n)]
    
    # Use L-BFGS-B algorithm which handles bounds well
    try:
        # For computational efficiency, we'll do a few iterations only
        # and return the best result found
        start_time = time.time()
        
        # Simple iterative improvement approach instead of complex optimization
        # This is more efficient for our time constraint
        best_f = f_values.copy()
        best_c2 = compute_c2(best_f)
        
        # Try some local improvements
        for iteration in range(50):  # Limited iterations due to time constraints
            if time.time() - start_time > 55:  # Leave 5 seconds for final computation
                break
                
            # Perturb one element at a time to see if we can improve
            for i in range(len(best_f)):
                old_val = best_f[i]
                # Try increasing slightly
                test_f = best_f.copy()
                test_f[i] = min(old_val + 0.1, 10.0)  # Cap at reasonable values
                c2_new = compute_c2(test_f)
                
                if c2_new > best_c2:
                    best_f = test_f
                    best_c2 = c2_new
                else:
                    # Try decreasing slightly
                    test_f[i] = max(old_val - 0.1, 0.0)
                    c2_new = compute_c2(test_f)
                    if c2_new > best_c2:
                        best_f = test_f
                        best_c2 = c2_new
                        
        return best_f
        
    except Exception:
        # If optimization fails, return the initial guess
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
