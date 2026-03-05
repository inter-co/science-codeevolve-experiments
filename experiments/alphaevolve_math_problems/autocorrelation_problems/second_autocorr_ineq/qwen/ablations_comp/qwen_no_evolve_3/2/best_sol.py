# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
import warnings

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using gradient-based optimization."""
    
    # Set up the optimization problem
    np.random.seed(42)  # For reproducibility
    
    # Start with a reasonable initial guess - a few peaks
    n_steps = 200  # Number of steps
    initial_guess = np.random.exponential(scale=0.5, size=n_steps)
    
    # Normalize to prevent extreme values
    initial_guess = initial_guess / np.sum(initial_guess) * 0.5
    
    def compute_c2(f_values):
        """Compute C2 value for given step function values."""
        # Ensure non-negativity
        f_values = np.maximum(f_values, 0)
        
        # Create the step function on [-1/4, 1/4]
        # We'll discretize this interval with appropriate resolution
        domain_length = 0.5  # From -1/4 to 1/4
        dx = domain_length / len(f_values)
        
        # Compute autoconvolution g = f * f
        # Using discrete convolution
        g = convolve(f_values, f_values, mode='full')
        g = g[len(g)//2:]  # Take the right half
        
        # Truncate to match original domain length
        g = g[:len(f_values)]
        
        # Compute norms
        g_squared = g ** 2
        norm_2_squared = np.sum(g_squared) * dx
        norm_1 = np.sum(np.abs(g)) * dx
        norm_inf = np.max(np.abs(g))
        
        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 0.0
            
        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2
    
    def objective(x):
        """Minimize negative C2 to maximize C2."""
        return -compute_c2(x)
    
    def constraint_positive(x):
        """Ensure all values are non-negative."""
        return np.min(x)
    
    # Set up constraints and bounds
    bounds = [(0, 10) for _ in range(n_steps)]
    constraints = [{'type': 'ineq', 'fun': constraint_positive}]
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_values = result.x
            # Clip negative values and normalize
            optimized_values = np.maximum(optimized_values, 0)
            # Normalize so that sum is reasonable
            optimized_values = optimized_values / np.sum(optimized_values) * 0.5
            return optimized_values.tolist()
        else:
            # Fallback to initial guess if optimization fails
            return initial_guess.tolist()
            
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        return initial_guess.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
