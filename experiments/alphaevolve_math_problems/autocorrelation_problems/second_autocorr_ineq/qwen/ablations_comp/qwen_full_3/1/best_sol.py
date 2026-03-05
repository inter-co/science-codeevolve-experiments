# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
import random
from typing import List
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

def compute_autoconvolution_norms(f_values: List[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    Uses proper discrete convolution and correct integration scheme as specified.
    """
    if not f_values:
        return 0.0, 0.0, 0.0
    
    # Convert to numpy array
    f = np.array(f_values)
    n = len(f)
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = convolve(f, f, mode='full')
    
    # The convolution result has length 2*n - 1
    # We want the portion that corresponds to the domain [-1/2, 1/2] 
    # when considering the convolution of two functions on [-1/4, 1/4]
    # This gives us the central portion of the convolution
    g_center_start = len(g) // 2 - (n - 1)
    g_center_end = len(g) // 2 + n
    g_centered = g[g_center_start:g_center_end]
    
    # Compute norms according to prompt specification:
    # ||g||₂² using piecewise linear integration: 
    # For each adjacent pair of values with width h, 
    # contribution is (h/3)(y1² + y1*y2 + y2²)
    if len(g_centered) < 2:
        g_norm_2_sq = 0.0
    else:
        # Width between consecutive points in original function
        step_width = 0.5 / (n - 1) if n > 1 else 0.5
        g_norm_2_sq = 0.0
        for i in range(len(g_centered) - 1):
            y1, y2 = g_centered[i], g_centered[i+1]
            # Trapezoidal-like piecewise integration: (h/3)(y1² + y1*y2 + y2²)
            g_norm_2_sq += (step_width / 3.0) * (y1**2 + y1*y2 + y2**2)
    
    # ||g||₁: sum of absolute values divided by (len(g) + 1) as per prompt
    g_norm_1 = np.sum(np.abs(g_centered)) / (len(g_centered) + 1)
    
    # ||g||∞: maximum absolute value
    g_norm_inf = np.max(np.abs(g_centered))
    
    return float(g_norm_2_sq), float(g_norm_1), float(g_norm_inf)

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given step function"""
    try:
        g_norm_2_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_norm_1 <= 1e-12 or g_norm_inf <= 1e-12:
            return 0.0
            
        c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
        return c2
    except Exception:
        return 0.0

def construct_function() -> List[float]:
    """
    Novel evolutionary approach using differential evolution with neural-inspired parameterization
    This implements a fundamentally different strategy from the original heuristics
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define problem dimensions - use a reasonable number of parameters
    # We'll use a neural-inspired approach with hidden layer parameters
    n_params = 100  # Number of parameters to optimize
    
    # Define bounds for each parameter (0 to 2, since we want non-negative values)
    bounds = [(0, 2) for _ in range(n_params)]
    
    # Define objective function for optimization
    def objective(params):
        # Convert parameters to step function
        # We'll use a radial basis function inspired approach
        n_steps = 500
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Create function using combination of RBFs with optimized parameters
        f_values = np.zeros(n_steps)
        
        # Each parameter controls a different RBF component
        # We'll distribute the parameters among different RBF centers and widths
        for i in range(0, min(len(params), n_params), 3):
            if i + 2 < len(params):
                center = params[i] * 0.5 - 0.25  # Map to [-0.25, 0.25]
                width = 0.1 + params[i+1] * 0.2  # Width between 0.1 and 0.3
                amplitude = params[i+2]  # Amplitude (0 to 2)
                
                # Create Gaussian RBF
                rbf = amplitude * np.exp(-((x - center)**2) / (2 * width**2))
                f_values += rbf
        
        # Ensure non-negativity
        f_values = np.maximum(f_values, 0)
        
        # Compute C2 score (we want to maximize this, so return negative)
        c2_score = compute_c2(f_values.tolist())
        return -c2_score if not np.isnan(c2_score) and not np.isinf(c2_score) else 1e10
    
    # Use differential evolution with custom bounds
    try:
        # Run differential evolution with limited iterations to stay within time limits
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,  # Reduced iterations to save time
            popsize=15,   # Population size
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        # Generate final function from best parameters
        best_params = result.x
        
        # Convert back to step function
        n_steps = 500
        x = np.linspace(-0.25, 0.25, n_steps)
        f_values = np.zeros(n_steps)
        
        # Reconstruct with the same RBF approach
        for i in range(0, min(len(best_params), n_params), 3):
            if i + 2 < len(best_params):
                center = best_params[i] * 0.5 - 0.25
                width = 0.1 + best_params[i+1] * 0.2
                amplitude = best_params[i+2]
                
                rbf = amplitude * np.exp(-((x - center)**2) / (2 * width**2))
                f_values += rbf
        
        f_values = np.maximum(f_values, 0)
        
        # Final refinement with local search
        current_solution = f_values.tolist()
        current_score = compute_c2(current_solution)
        
        # Perform additional local search
        for _ in range(50):  # Limited iterations
            neighbor = []
            for val in current_solution:
                if val > 0:
                    # Adaptive perturbation
                    step_size = 0.02 * val if val > 0.1 else 0.005
                    perturbation = random.gauss(0, step_size)
                    new_val = max(0, val + perturbation)
                else:
                    new_val = random.random() * 0.5
                neighbor.append(new_val)
            
            neighbor_score = compute_c2(neighbor)
            if neighbor_score > current_score:
                current_solution = neighbor
                current_score = neighbor_score
        
        return current_solution
        
    except Exception as e:
        # Fallback to simpler approach if optimization fails
        print(f"Optimization failed: {e}")
        # Use a simple constructed function with good theoretical properties
        n = 500
        x = np.linspace(-0.25, 0.25, n)
        # Create a function that balances smoothness and sharp features
        f_values = np.exp(-x**2 * 10) * (1 + 0.5 * np.sin(8 * np.pi * x))
        f_values = np.maximum(f_values, 0)
        return f_values.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
