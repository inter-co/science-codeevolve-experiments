# EVOLVE-BLOCK-START

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import time
import random
from typing import List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)
jax.config.update('jax_enable_x64', True)

def compute_autoconvolution_jax(f):
    """Compute autoconvolution using JAX for better performance"""
    # Ensure f is a JAX array
    f = jnp.array(f)
    
    # Compute autoconvolution g = f * f
    g = jnp.convolve(f, f, mode='full')
    
    # Extract the central portion corresponding to [-1/4, 1/4]
    # For a function of n points, the autoconvolution has 2*n-1 points
    # The central portion of length n corresponds to the valid range
    center = len(g) // 2
    half_len = len(f) - 1
    g_trimmed = g[center - half_len:center + half_len + 1]
    
    return g_trimmed

def compute_c2_jax(f):
    """Compute C2 using JAX for automatic differentiation"""
    # Ensure non-negative values
    f = jnp.maximum(f, 0.0)
    
    # Compute autoconvolution
    g = compute_autoconvolution_jax(f)
    
    # Compute norms
    l2_squared = jnp.sum(g**2)
    l1_norm = jnp.sum(jnp.abs(g))
    l_inf_norm = jnp.max(jnp.abs(g))
    
    # Avoid division by zero
    epsilon = 1e-12
    l1_norm = jnp.maximum(l1_norm, epsilon)
    l_inf_norm = jnp.maximum(l_inf_norm, epsilon)
    
    # Compute C2
    c2 = l2_squared / (l1_norm * l_inf_norm)
    
    return c2

@jit
def compute_c2_grad(f):
    """Compute C2 and its gradient using JAX"""
    c2 = compute_c2_jax(f)
    grad_c2 = grad(compute_c2_jax)(f)
    return c2, grad_c2

def compute_autoconvolution_norms(f_values: List[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation using the specified piecewise 
    linear integration method:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Handle edge cases
    if len(f) == 0:
        return 0.0, 0.0, 0.0
    
    # Compute autoconvolution g = f * f using scipy
    g = signal.convolve(f, f, mode='full')
    
    # The convolution result has length 2*len(f) - 1
    # We want to extract the central part corresponding to [-1/4, 1/4]
    center = len(g) // 2
    half_len = len(f) - 1
    
    # Extract the central portion (correctly centered)
    g_start = center - half_len
    g_end = center + half_len + 1
    g_trimmed = g[g_start:g_end]
    
    # Create proper domain for integration
    dx = 0.5 / (len(f) - 1) if len(f) > 1 else 0.5
    
    # Compute norms using the exact integration method specified in problem
    g_abs = np.abs(g_trimmed)
    
    # ||g||₂² using piecewise linear integration as specified: (h/3)(y1² + y1*y2 + y2²)
    if len(g_abs) >= 2:
        norm_2_squared = 0.0
        for i in range(len(g_abs) - 1):
            y1 = g_abs[i]
            y2 = g_abs[i+1]
            norm_2_squared += (dx/3) * (y1**2 + y1*y2 + y2**2)
    else:
        norm_2_squared = np.sum(g_abs ** 2) * dx
    
    # ||g||₁ using trapezoidal rule for integral of |g| - simplified version
    if len(g_abs) >= 2:
        # Trapezoidal rule: (h/2) * (y0 + 2*y1 + ... + 2*y_{n-2} + y_{n-1})
        norm_1 = dx * (0.5 * np.sum(g_abs[:-1]) + 0.5 * np.sum(g_abs[1:]) + 0.5 * (g_abs[0] + g_abs[-1]))
    else:
        norm_1 = np.sum(g_abs) * dx
    
    # ||g||∞ = max(|g|)
    norm_inf = np.max(g_abs)
    
    return norm_2_squared, norm_1, norm_inf

def compute_c2(f_values: List[float]) -> float:
    """
    Compute C2 = ||g||₂² / (||g||₁ · ||g||∞) using the exact numerical integration
    """
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    return norm_2_squared / (norm_1 * norm_inf)

def create_mathematical_initialization(n: int) -> np.ndarray:
    """
    Create a sophisticated mathematical pattern based on successful approaches
    """
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a sophisticated multi-peak pattern that's specifically designed
    # to create favorable autoconvolution properties
    f_init = np.zeros(n)
    
    # Central high peak for maximum concentration
    f_init += 1.5 * np.exp(-x**2 / (2 * 0.02**2))
    
    # Well-placed side peaks for constructive interference
    f_init += 0.8 * np.exp(-((x - 0.12)**2) / (2 * 0.04**2))
    f_init += 0.8 * np.exp(-((x + 0.12)**2) / (2 * 0.04**2))
    
    # Additional peaks for complexity
    f_init += 0.4 * np.exp(-((x - 0.18)**2) / (2 * 0.03**2))
    f_init += 0.4 * np.exp(-((x + 0.18)**2) / (2 * 0.03**2))
    
    # Add oscillation to break symmetry and encourage better convolution behavior
    f_init += 0.15 * np.sin(20 * np.pi * x) * np.exp(-x**2 / 0.03)
    
    # Normalize to reasonable scale
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add small noise for exploration
    noise = np.random.normal(0, 0.01, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_balanced_pattern(n: int) -> np.ndarray:
    """Create a balanced pattern emphasizing both smoothness and constructive interference"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a smooth, symmetric pattern that balances high values with good distribution
    f_init = 0.7 * np.exp(-x**2 / (2 * 0.05**2))
    
    # Add side peaks for better convolution
    f_init += 0.3 * np.exp(-((x - 0.1)**2) / (2 * 0.06**2))
    f_init += 0.3 * np.exp(-((x + 0.1)**2) / (2 * 0.06**2))
    
    # Add some oscillation to create interesting convolution behavior
    f_init += 0.1 * np.sin(15 * np.pi * x)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.9
    
    # Add noise
    noise = np.random.normal(0, 0.015, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_sparse_peak_pattern(n: int) -> np.ndarray:
    """Create a sparse peak pattern to maximize L2^2 term"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Create sparse, high-amplitude peaks
    f_init = np.zeros(n)
    
    # Concentrated peaks with high values
    peak_positions = [-0.18, -0.09, 0.0, 0.09, 0.18]
    peak_heights = [1.8, 1.2, 2.0, 1.2, 1.8]
    
    for pos, height in zip(peak_positions, peak_heights):
        f_init += height * np.exp(-((x - pos)**2) / (2 * 0.025**2))
    
    # Add oscillation for better convolution properties
    f_init += 0.1 * np.sin(25 * np.pi * x)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.8
    
    # Add noise
    noise = np.random.normal(0, 0.01, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_advanced_mathematical_pattern(n: int) -> np.ndarray:
    """
    Create an advanced mathematical pattern combining multiple elements from inspirations
    """
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a sophisticated pattern with multiple peaks and oscillations
    f_init = np.zeros(n)
    
    # Primary bell shape (centered)
    sigma_primary = 0.05
    f_init += 0.7 * np.exp(-x**2 / (2 * sigma_primary**2))
    
    # Secondary peaks for better structure
    sigma_secondary = 0.1
    f_init += 0.15 * np.exp(-((x - 0.1)**2) / (2 * sigma_secondary**2))
    f_init += 0.15 * np.exp(-((x + 0.1)**2) / (2 * sigma_secondary**2))
    
    # Add some oscillation to break symmetry
    f_init += 0.05 * np.sin(20 * np.pi * x) * np.exp(-x**2 / 0.02)
    
    # Normalize and ensure positivity
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add controlled noise for exploration
    noise = np.random.normal(0, 0.01, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def aggressive_gradient_optimization(f_init, max_iter=5000, patience=50):
    """Run highly aggressive gradient optimization with fast convergence"""
    f_opt = jnp.array(f_init)
    
    # Very aggressive Adam parameters for rapid convergence
    learning_rate = 0.3  # Much higher learning rate
    beta1 = 0.99         # Very high momentum
    beta2 = 0.9999       # Nearly perfect momentum  
    epsilon = 1e-8
    
    # Initialize Adam moments
    m = jnp.zeros_like(f_opt)
    v = jnp.zeros_like(f_opt)
    
    best_c2 = -float('inf')
    best_f = f_opt
    patience_counter = 0
    last_improvement_iter = 0
    
    # Optimization loop with very aggressive parameters
    for i in range(max_iter):
        # Compute C2 and gradient
        c2_val, grad_f = compute_c2_grad(f_opt)
        
        # Update Adam moments
        m = beta1 * m + (1 - beta1) * grad_f
        v = beta2 * v + (1 - beta2) * grad_f**2
        
        # Bias correction
        m_hat = m / (1 - beta1**(i+1))
        v_hat = v / (1 - beta2**(i+1))
        
        # Update parameters
        f_opt = f_opt + learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        # Ensure non-negativity
        f_opt = jnp.maximum(f_opt, 0.0)
        
        # Track best solution
        if c2_val > best_c2:
            best_c2 = c2_val
            best_f = f_opt
            patience_counter = 0
            last_improvement_iter = i
        else:
            patience_counter += 1
            
        # Aggressive early stopping with learning rate reduction
        if patience_counter >= patience:
            if i - last_improvement_iter > 200:
                # Reduce learning rate more aggressively
                learning_rate *= 0.92
                patience_counter = 0
                if learning_rate < 1e-4:
                    break
            elif i > 500:
                # If we're still not improving after 500 iterations, reduce learning rate
                learning_rate *= 0.95
                patience_counter = 0
    
    return best_f, best_c2

def hybrid_optimization_approach() -> list[float]:
    """
    Hybrid optimization approach that leverages both global and local search strategies
    """
    start_time = time.time()
    
    # Use a high-resolution pattern for better optimization
    n_steps = 1500  # Higher resolution for better results
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Multiple highly specialized initialization strategies
    init_strategies = [
        ("mathematical", create_mathematical_initialization),
        ("balanced", create_balanced_pattern),
        ("sparse_peak", create_sparse_peak_pattern),
        ("advanced_math", create_advanced_mathematical_pattern),
        ("random", lambda n: np.abs(np.random.normal(0.6, 0.2, n))),
    ]
    
    # Strategy: Multiple restarts with different patterns
    num_restarts = 25  # More restarts for better exploration
    
    for restart in range(num_restarts):
        if time.time() - start_time > 55:  # Leave buffer time
            break
            
        # Choose initialization strategy
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        f_init = init_func(n_steps)
        
        # Run extremely aggressive gradient optimization
        f_opt, c2_val = aggressive_gradient_optimization(f_init, max_iter=3000, patience=50)
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_solution = f_opt
    
    # Strategy 2: Final refinement with ultra-aggressive optimization
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            # Run ultra-aggressive optimization
            f_opt, c2_val = aggressive_gradient_optimization(f_init, max_iter=4000, patience=30)
            
            if c2_val > best_c2:
                best_c2 = c2_val
                best_solution = f_opt
        except Exception:
            pass
    
    # Strategy 3: Final differential evolution refinement
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_final = np.array(best_solution)
            
            def objective(f_vals):
                # Clip negative values to 0
                f_vals = np.maximum(f_vals, 0)
                # Compute C2 using the same formula as in the original
                c2 = compute_c2_jax(f_vals)
                return -c2  # Minimize negative to maximize C2
            
            # Bounds for each parameter (non-negative)
            bounds = [(0, 1) for _ in range(len(f_final))]
            
            # Run differential evolution with more iterations
            result = differential_evolution(
                objective,
                bounds,
                maxiter=30,
                popsize=20,
                mutation=(0.5, 1),
                recombination=0.8,
                seed=42,
                disp=False
            )
            
            if result.success:
                refined_f = np.maximum(result.x, 0)
                refined_c2 = compute_c2_jax(refined_f)
                if refined_c2 > best_c2:
                    best_c2 = refined_c2
                    best_solution = refined_f
                    
        except Exception:
            pass
    
    # Final fallback: return the best solution found
    if best_solution is None:
        # Create a robust default solution
        f_default = np.ones(n_steps) * 0.5
        best_solution = jnp.array(f_default)
    
    # Final evaluation and conversion to list
    return list(best_solution)

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses an enhanced hybrid approach that leverages aggressive optimization
    and diverse initialization patterns.
    """
    return hybrid_optimization_approach()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
