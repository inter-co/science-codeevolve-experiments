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

def create_optimized_initialization(n: int) -> np.ndarray:
    """Create an optimized initialization pattern based on mathematical insights from inspirations"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Pattern inspired by mathematical analysis of optimal convolution structures
    # Emphasis on creating strong central mass with strategic side peaks
    f_init = np.zeros(n)
    
    # Central Gaussian peak (very strong)
    f_init += 1.6 * np.exp(-x**2 / (2 * 0.02**2))
    
    # Symmetric side peaks with specific positioning for constructive interference
    f_init += 0.8 * np.exp(-((x - 0.13)**2) / (2 * 0.045**2))
    f_init += 0.8 * np.exp(-((x + 0.13)**2) / (2 * 0.045**2))
    
    # Additional fine structure for better convolution properties
    f_init += 0.4 * np.exp(-((x - 0.18)**2) / (2 * 0.025**2))
    f_init += 0.4 * np.exp(-((x + 0.18)**2) / (2 * 0.025**2))
    
    # Oscillation component to create beneficial convolution characteristics
    f_init += 0.15 * np.sin(25 * np.pi * x)
    
    # Normalize properly
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.92
    
    # Add small noise for exploration
    noise = np.random.normal(0, 0.002, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_multi_scale_pattern(n: int) -> np.ndarray:
    """Create a multi-scale pattern combining different frequency components"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Mix different scales for rich convolution behavior
    f_init = np.zeros(n)
    
    # Broad central peak for strong L2 norm
    f_init += 0.9 * np.exp(-x**2 / (2 * 0.08**2))
    
    # Medium peaks for constructive interference
    f_init += 0.7 * np.exp(-((x - 0.1)**2) / (2 * 0.04**2))
    f_init += 0.7 * np.exp(-((x + 0.1)**2) / (2 * 0.04**2))
    
    # Fine structure for detailed convolution properties
    f_init += 0.3 * np.exp(-((x - 0.18)**2) / (2 * 0.02**2))
    f_init += 0.3 * np.exp(-((x + 0.18)**2) / (2 * 0.02**2))
    
    # Oscillation components
    f_init += 0.12 * np.sin(20 * np.pi * x)
    f_init += 0.08 * np.sin(30 * np.pi * x)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.88
    
    # Add noise
    noise = np.random.normal(0, 0.003, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_concentrated_mass_pattern(n: int) -> np.ndarray:
    """Create a highly concentrated mass pattern for maximum convolution efficiency"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Extremely concentrated mass pattern
    f_init = 2.0 * np.exp(-x**2 / (2 * 0.015**2))
    
    # Secondary peaks for better structure
    f_init += 0.6 * np.exp(-((x - 0.1)**2) / (2 * 0.03**2))
    f_init += 0.6 * np.exp(-((x + 0.1)**2) / (2 * 0.03**2))
    
    # Additional fine structure
    f_init += 0.4 * np.exp(-((x - 0.18)**2) / (2 * 0.018**2))
    f_init += 0.4 * np.exp(-((x + 0.18)**2) / (2 * 0.018**2))
    
    # Oscillation for constructive interference
    f_init += 0.1 * np.sin(30 * np.pi * x)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add noise
    noise = np.random.normal(0, 0.002, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_balanced_asymmetric_pattern(n: int) -> np.ndarray:
    """Create a balanced asymmetric pattern for optimal convolution"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Asymmetric pattern with mathematical balance
    f_init = np.zeros(n)
    
    # Strong central peak
    f_init += 1.4 * np.exp(-x**2 / (2 * 0.025**2))
    
    # Left side with larger peak
    f_init += 0.7 * np.exp(-((x + 0.12)**2) / (2 * 0.04**2))
    
    # Right side with smaller peak
    f_init += 0.5 * np.exp(-((x - 0.1)**2) / (2 * 0.035**2))
    
    # Additional oscillation
    f_init += 0.1 * np.sin(20 * np.pi * x)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.9
    
    # Add noise
    noise = np.random.normal(0, 0.004, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_adaptive_gradient_optimization(f_init, max_iter=3000, patience=70):
    """Adaptive gradient optimization with intelligent parameter adjustment"""
    f_opt = jnp.array(f_init)
    
    # Adaptive Adam parameters - starting with good defaults but allowing adaptation
    learning_rate = 0.15
    beta1 = 0.95
    beta2 = 0.999
    epsilon = 1e-8
    
    # Initialize Adam moments
    m = jnp.zeros_like(f_opt)
    v = jnp.zeros_like(f_opt)
    
    best_c2 = -float('inf')
    best_f = f_opt
    patience_counter = 0
    last_improvement_iter = 0
    
    # Optimization loop with adaptive behavior
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
            
        # Adaptive learning rate adjustment based on progress
        if patience_counter > 50 and i > 1000:
            learning_rate *= 0.98
            # Reset patience if we're still making progress
            if patience_counter > 100:
                patience_counter = 0
                
        # Early stopping with adaptive patience
        if patience_counter >= patience and i - last_improvement_iter > 200:
            break
    
    return best_f, best_c2

def construct_function() -> list[float]:
    """
    Optimized hybrid approach that maximizes C2 effectively within time constraints.
    Combines mathematical insights, adaptive optimization, and strategic search.
    """
    
    start_time = time.time()
    
    # Problem parameters - balanced for time and quality
    n_steps = 1500  # Good compromise between resolution and time
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Strategy 1: Multiple restarts with carefully chosen initialization patterns
    init_strategies = [
        ("optimized_initial", create_optimized_initialization),
        ("multi_scale", create_multi_scale_pattern),
        ("concentrated_mass", create_concentrated_mass_pattern),
        ("balanced_asymmetric", create_balanced_asymmetric_pattern),
        ("random", lambda n: np.abs(np.random.normal(0.5, 0.2, n))),
    ]
    
    # Try fewer but more targeted restarts for efficiency
    num_restarts = 15  # Reduced from previous versions for better time management
    for restart in range(num_restarts):
        if time.time() - start_time > 55:  # Leave buffer time
            break
            
        # Choose initialization strategy
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        f_init = init_func(n_steps)
        
        # Run adaptive gradient optimization
        f_opt, c2_val = create_adaptive_gradient_optimization(f_init, max_iter=2500, patience=80)
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_solution = f_opt
    
    # Strategy 2: Differential evolution refinement for final polish
    if best_solution is not None and time.time() - start_time < 50:
        try:
            # Use the best solution found so far
            f_init = np.array(best_solution)
            
            def objective(f_vals):
                f_vals = np.maximum(f_vals, 0)
                c2 = compute_c2_jax(f_vals)
                return -c2  # Minimize negative to maximize C2
            
            # Bounds for each parameter (non-negative)
            bounds = [(0, 3.0) for _ in range(len(f_init))]
            
            # Run differential evolution with moderate effort
            result = differential_evolution(
                objective,
                bounds,
                maxiter=50,  # Reduced iterations for time efficiency
                popsize=20,   # Moderate population size
                mutation=(0.8, 1.0),  # Standard mutation range
                recombination=0.7,   # Moderate recombination
                seed=42,
                disp=False,
                atol=1e-10,
                rtol=1e-10
            )
            
            if result.success:
                final_f = np.maximum(result.x, 0)
                de_c2 = compute_c2_jax(final_f)
                if de_c2 > best_c2:
                    best_c2 = de_c2
                    best_solution = jnp.array(final_f)
        except Exception:
            pass
    
    # Strategy 3: Final refinement if time allows
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            # Run final optimization with more iterations but shorter patience
            f_opt, c2_val = create_adaptive_gradient_optimization(f_init, max_iter=2000, patience=50)
            
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
