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
    """Create an optimized initialization pattern with enhanced mathematical structure"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Pattern inspired by mathematical analysis of optimal convolution structures
    # Based on successful patterns from inspirations - even stronger central mass with optimal side structure
    f_init = np.zeros(n)
    
    # Central dominant peak - even sharper and higher amplitude for maximum concentration
    f_init += 2.5 * np.exp(-x**2 / (2 * 0.01**2))
    
    # Strategic side peaks with optimal spacing for constructive interference
    f_init += 1.5 * np.exp(-((x - 0.12)**2) / (2 * 0.03**2))
    f_init += 1.5 * np.exp(-((x + 0.12)**2) / (2 * 0.03**2))
    
    # Additional fine-scale structure for better convolution properties
    f_init += 1.0 * np.exp(-((x - 0.18)**2) / (2 * 0.018**2))
    f_init += 1.0 * np.exp(-((x + 0.18)**2) / (2 * 0.018**2))
    
    # Controlled oscillation for better convolution behavior
    f_init += 0.3 * np.sin(40 * np.pi * x) * np.exp(-x**2 / 0.03)
    
    # Normalize properly
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.92
    
    # Add small noise for exploration
    noise = np.random.normal(0, 0.0003, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_concentrated_mass_pattern(n: int) -> np.ndarray:
    """Create a highly concentrated mass pattern with enhanced mathematical structure"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Extremely concentrated mass pattern with improved scaling
    f_init = 2.5 * np.exp(-x**2 / (2 * 0.012**2))
    
    # Secondary peaks with better positioning for constructive interference
    f_init += 1.0 * np.exp(-((x - 0.1)**2) / (2 * 0.035**2))
    f_init += 1.0 * np.exp(-((x + 0.1)**2) / (2 * 0.035**2))
    
    # Additional fine structure with precise scaling
    f_init += 0.6 * np.exp(-((x - 0.18)**2) / (2 * 0.015**2))
    f_init += 0.6 * np.exp(-((x + 0.18)**2) / (2 * 0.015**2))
    
    # Enhanced oscillation with better amplitude control
    f_init += 0.2 * np.sin(35 * np.pi * x) * np.exp(-x**2 / 0.025)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.88
    
    # Add noise
    noise = np.random.normal(0, 0.001, n)
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

def create_aggressive_gradient_optimization(f_init, max_iter=4000, patience=40):
    """Very aggressive gradient optimization with parameters from successful inspirations"""
    f_opt = jnp.array(f_init)
    
    # Very aggressive Adam parameters - based on the most successful inspirations
    learning_rate = 0.35  # Even higher learning rate for fastest convergence
    beta1 = 0.99          # Very high momentum for faster convergence
    beta2 = 0.99999       # Nearly perfect momentum
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
            
        # Even more aggressive early stopping with faster learning rate reduction
        if patience_counter >= patience:
            if i - last_improvement_iter > 150:
                # Very aggressive learning rate reduction
                learning_rate *= 0.95
                patience_counter = 0
                if learning_rate < 1e-4:
                    break
            elif i > 300:
                # If we're still not improving after 300 iterations, reduce learning rate
                learning_rate *= 0.97
                patience_counter = 0
    
    return best_f, best_c2

def construct_function() -> list[float]:
    """
    Enhanced hybrid approach that maximizes C2 effectively within time constraints.
    Combines mathematical insights, aggressive optimization, and strategic search.
    """
    
    start_time = time.time()
    
    # Problem parameters - slightly higher resolution for better optimization
    n_steps = 1600  # Slightly higher resolution for better results
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Strategy 1: Even more diverse initialization strategies with focus on top performers
    # Incorporating the best elements from all inspirations
    init_strategies = [
        ("optimized_initial", create_optimized_initialization),
        ("concentrated_mass", create_concentrated_mass_pattern),
        ("multi_scale", create_multi_scale_pattern),
        ("balanced_asymmetric", create_balanced_asymmetric_pattern),
        ("optimized_initial_v2", create_optimized_initialization),  # Duplicate for more emphasis
        ("random", lambda n: np.abs(np.random.normal(0.6, 0.2, n))),  # Fallback random
        ("random2", lambda n: np.abs(np.random.normal(0.5, 0.25, n))),  # Another random
        ("random3", lambda n: np.abs(np.random.normal(0.7, 0.15, n))),  # Another random
    ]
    
    # Try even more restarts to ensure we don't miss the global optimum
    num_restarts = 50  # Even more restarts for better exploration
    for restart in range(num_restarts):
        if time.time() - start_time > 45:  # Leave more buffer for final polish
            break
            
        # Choose initialization strategy
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        f_init = init_func(n_steps)
        
        # Run very aggressive gradient optimization with more patience
        f_opt, c2_val = create_aggressive_gradient_optimization(f_init, max_iter=4000, patience=35)
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_solution = f_opt
    
    # Strategy 2: Differential evolution refinement for final polish with highest precision
    if best_solution is not None and time.time() - start_time < 48:
        try:
            # Use the best solution found so far
            f_init = np.array(best_solution)
            
            def objective(f_vals):
                f_vals = np.maximum(f_vals, 0)
                c2 = compute_c2_jax(f_vals)
                return -c2  # Minimize negative to maximize C2
            
            # Bounds for each parameter (non-negative)
            bounds = [(0, 3.0) for _ in range(len(f_init))]
            
            # Run differential evolution with MOST thorough search - most aggressive settings
            result = differential_evolution(
                objective,
                bounds,
                maxiter=150,  # Even more iterations for better refinement
                popsize=45,   # Larger population size for better exploration
                mutation=(0.95, 1.0),  # Highest mutation for more exploration
                recombination=0.95,   # Highest recombination for better mixing
                seed=42,
                disp=False,
                atol=1e-15,  # Tightest tolerance for best convergence
                rtol=1e-15
            )
            
            if result.success:
                final_f = np.maximum(result.x, 0)
                de_c2 = compute_c2_jax(final_f)
                if de_c2 > best_c2:
                    best_c2 = de_c2
                    best_solution = jnp.array(final_f)
        except Exception:
            pass
    
    # Strategy 3: Final refinement with ultra-aggressive optimization if time allows
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            # Run ultra-aggressive optimization with maximum iterations
            f_opt, c2_val = create_aggressive_gradient_optimization(f_init, max_iter=5000, patience=20)
            
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
