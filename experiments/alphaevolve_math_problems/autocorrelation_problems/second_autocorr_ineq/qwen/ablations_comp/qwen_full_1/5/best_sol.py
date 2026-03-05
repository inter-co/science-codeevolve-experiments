# EVOLVE-BLOCK-START

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy import signal
from scipy.optimize import differential_evolution
import time
import random

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

def create_optimized_pattern(n: int) -> np.ndarray:
    """Create a highly optimized initialization pattern"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a pattern that should yield very high C2 values
    # Based on mathematical analysis of optimal convolution structures
    f_init = np.zeros(n)
    
    # Central strong peak
    f_init += 1.3 * np.exp(-x**2 / (2 * 0.025**2))
    
    # Two symmetric peaks to create good convolution
    f_init += 0.7 * np.exp(-((x - 0.12)**2) / (2 * 0.05**2))
    f_init += 0.7 * np.exp(-((x + 0.12)**2) / (2 * 0.05**2))
    
    # Additional small peaks for structure
    f_init += 0.2 * np.exp(-((x - 0.18)**2) / (2 * 0.04**2))
    f_init += 0.2 * np.exp(-((x + 0.18)**2) / (2 * 0.04**2))
    
    # Add oscillation for better structure
    f_init += 0.1 * np.sin(25 * np.pi * x)
    
    # Normalize properly
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.9
    
    # Add noise
    noise = np.random.normal(0, 0.005, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_peak_distributed_pattern(n: int) -> np.ndarray:
    """Create a pattern with distributed peaks"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Distribute peaks to maximize convolution effects
    f_init = np.zeros(n)
    
    # Central peak
    f_init += 1.0 * np.exp(-x**2 / (2 * 0.03**2))
    
    # Multiple peaks at different locations
    peaks = [(-0.18, 0.5), (-0.1, 0.6), (0.1, 0.6), (0.18, 0.5)]
    for pos, height in peaks:
        f_init += height * np.exp(-((x - pos)**2) / (2 * 0.04**2))
    
    # Add oscillation
    f_init += 0.1 * np.sin(20 * np.pi * x)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add noise
    noise = np.random.normal(0, 0.008, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_highly_concentrated_pattern(n: int) -> np.ndarray:
    """Create a highly concentrated pattern"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Very concentrated with minimal spread
    f_init = 1.8 * np.exp(-x**2 / (2 * 0.015**2))
    
    # Add secondary peaks
    f_init += 0.4 * np.exp(-((x - 0.15)**2) / (2 * 0.03**2))
    f_init += 0.4 * np.exp(-((x + 0.15)**2) / (2 * 0.03**2))
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add noise
    noise = np.random.normal(0, 0.003, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_wide_spread_pattern(n: int) -> np.ndarray:
    """Create a wide-spread pattern"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Wide spread pattern to encourage better L2/L1 balance
    f_init = np.zeros(n)
    
    # Broad peaks
    f_init += 0.6 * np.exp(-((x - 0.1)**2) / (2 * 0.08**2))
    f_init += 0.6 * np.exp(-((x + 0.1)**2) / (2 * 0.08**2))
    f_init += 0.8 * np.exp(-x**2 / (2 * 0.06**2))
    
    # Add oscillation
    f_init += 0.15 * np.sin(10 * np.pi * x)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.8
    
    # Add noise
    noise = np.random.normal(0, 0.01, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def super_aggressive_gradient_optimization(f_init, max_iter=6000, patience=80):
    """Super aggressive gradient optimization with maximum parameters"""
    f_opt = jnp.array(f_init)
    
    # SUPER AGGRESSIVE Adam parameters for maximum speed and exploration
    learning_rate = 0.3  # Extremely high learning rate
    beta1 = 0.995        # Very high momentum
    beta2 = 0.99999      # Nearly perfect momentum  
    epsilon = 1e-8
    
    # Initialize Adam moments
    m = jnp.zeros_like(f_opt)
    v = jnp.zeros_like(f_opt)
    
    best_c2 = -float('inf')
    best_f = f_opt
    patience_counter = 0
    last_improvement_iter = 0
    
    # Optimization loop with super aggressive convergence tracking
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
        
        # Track best solution with super aggressive criteria
        if c2_val > best_c2:
            best_c2 = c2_val
            best_f = f_opt
            patience_counter = 0
            last_improvement_iter = i
        else:
            patience_counter += 1
            
        # Super aggressive early stopping
        if patience_counter >= patience and i - last_improvement_iter > 150:
            # Super aggressive learning rate reduction
            if i > 1000:
                learning_rate *= 0.95
            if patience_counter >= patience * 3:
                break
    
    return best_f, best_c2

def construct_function() -> list[float]:
    """
    Super aggressive hybrid optimization approach that maximizes C2 effectively within time constraints.
    Uses super aggressive optimization and comprehensive search strategies to exceed benchmark.
    """
    
    start_time = time.time()
    
    # Problem parameters - using maximum resolution possible within time limits
    n_steps = 2000  # Maximum resolution for better optimization
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Strategy 1: Super aggressive multiple restarts with diverse initialization patterns
    init_strategies = [
        ("optimized", create_optimized_pattern),
        ("peak_distributed", create_peak_distributed_pattern),
        ("highly_concentrated", create_highly_concentrated_pattern),
        ("wide_spread", create_wide_spread_pattern),
        ("random", lambda n: np.abs(np.random.normal(0.5, 0.2, n))),
        ("gaussian", lambda n: np.exp(-((np.linspace(-0.25, 0.25, n))**2) / (2 * 0.05**2))),
    ]
    
    # Try EVEN MORE restarts with different strategies - super aggressive exploration
    num_restarts = 50  # Many more restarts for better chance of global optimum
    for restart in range(num_restarts):
        if time.time() - start_time > 40:  # Leave more buffer time
            break
            
        # Choose initialization strategy
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        f_init = init_func(n_steps)
        
        # Run super aggressive gradient optimization
        f_opt, c2_val = super_aggressive_gradient_optimization(f_init, max_iter=5000, patience=60)
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_solution = f_opt
    
    # Strategy 2: Super aggressive differential evolution as final refinement
    if best_solution is None or best_c2 < 0.97:
        try:
            # Use the best solution found so far or a mathematically-informed initialization
            if best_solution is not None:
                f_init = np.array(best_solution)
            else:
                f_init = create_optimized_pattern(n_steps)
            
            def objective(f_vals):
                f_vals = np.maximum(f_vals, 0)
                c2 = compute_c2_jax(f_vals)
                return -c2  # Minimize negative to maximize C2
            
            # Bounds for each parameter (non-negative)
            bounds = [(0, 5.0) for _ in range(len(f_init))]
            
            # Run super aggressive differential evolution with maximum effort
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,  # MANY more iterations for thorough search
                popsize=40,   # VERY large population for better exploration
                mutation=(0.98, 1.0),  # Extremely high mutation rate for maximum exploration
                recombination=0.99,   # Very high recombination for maximum mixing
                seed=42,
                disp=False,
                atol=1e-12,  # Extremely tight tolerance
                rtol=1e-12
            )
            
            if result.success:
                final_f = np.maximum(result.x, 0)
                de_c2 = compute_c2_jax(final_f)
                if de_c2 > best_c2:
                    best_c2 = de_c2
                    best_solution = jnp.array(final_f)
        except Exception:
            pass
    
    # Strategy 3: Final super intense gradient optimization if time allows
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            f_opt, c2_val = super_aggressive_gradient_optimization(f_init, max_iter=3000, patience=50)
            
            if c2_val > best_c2:
                best_c2 = c2_val
                best_solution = f_opt
        except Exception:
            pass
    
    # Final fallback: return the best solution found
    if best_solution is None:
        # Create a robust default solution
        f_default = np.ones(n_steps) * 0.5
        best_solution = jnp.array(f_default)
    
    # Final evaluation and conversion to list
    final_c2 = compute_c2_jax(best_solution)
    
    return list(best_solution)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
