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

def create_ultimate_pattern(n: int) -> np.ndarray:
    """Create ultimate pattern with very aggressive parameters for maximum C2"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Ultimate pattern designed specifically for maximum convolution efficiency
    f_init = np.zeros(n)
    
    # Very strong central peak
    f_init += 2.2 * np.exp(-x**2 / (2 * 0.01**2))
    
    # Precisely positioned symmetric peaks with exact spacing
    f_init += 1.0 * np.exp(-((x - 0.12)**2) / (2 * 0.03**2))
    f_init += 1.0 * np.exp(-((x + 0.12)**2) / (2 * 0.03**2))
    
    # Additional structure for better convolution properties
    f_init += 0.4 * np.exp(-((x - 0.18)**2) / (2 * 0.025**2))
    f_init += 0.4 * np.exp(-((x + 0.18)**2) / (2 * 0.025**2))
    
    # High frequency oscillation for maximum structural complexity
    oscillation = 0.25 * np.sin(40 * np.pi * x)
    
    f_init = f_init + oscillation
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add very small noise for exploration
    noise = np.random.normal(0, 0.001, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_super_optimized_pattern(n: int) -> np.ndarray:
    """Create a super optimized pattern based on advanced mathematical insights"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Highly strategic multi-peak arrangement with precise amplitudes and positions
    # Based on mathematical analysis of optimal convolution structures
    
    # Central strong peak with very high amplitude
    central = 1.5 * np.exp(-x**2 / (2 * 0.02**2))
    
    # Two symmetric peaks with precise positioning
    left_peak = 0.8 * np.exp(-((x + 0.14)**2) / (2 * 0.04**2))
    right_peak = 0.8 * np.exp(-((x - 0.14)**2) / (2 * 0.04**2))
    
    # Additional peaks for better convolution structure
    left_secondary = 0.4 * np.exp(-((x + 0.18)**2) / (2 * 0.03**2))
    right_secondary = 0.4 * np.exp(-((x - 0.18)**2) / (2 * 0.03**2))
    
    # Add oscillation for better structure
    oscillation = 0.15 * np.sin(25 * np.pi * x)
    
    f_init = central + left_peak + right_peak + left_secondary + right_secondary + oscillation
    
    # Normalize to reasonable scale
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.95
    
    # Add noise for exploration
    noise = np.random.normal(0, 0.003, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_concentrated_peak_pattern(n: int) -> np.ndarray:
    """Create a highly concentrated peak pattern for maximum convolution efficiency"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a highly concentrated pattern that focuses mass efficiently
    f_init = 1.8 * np.exp(-x**2 / (2 * 0.015**2))
    
    # Add secondary peaks for better structure
    f_init += 0.5 * np.exp(-((x - 0.12)**2) / (2 * 0.03**2))
    f_init += 0.5 * np.exp(-((x + 0.12)**2) / (2 * 0.03**2))
    
    # Add oscillation for constructive interference
    oscillation = 0.1 * np.sin(30 * np.pi * x)
    
    f_init = f_init + oscillation
    
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
    
    # Asymmetric pattern with balanced structure for optimal convolution
    f_init = np.zeros(n)
    
    # Strong central peak
    f_init += 1.3 * np.exp(-x**2 / (2 * 0.025**2))
    
    # Left side with larger peak
    f_init += 0.7 * np.exp(-((x + 0.1)**2) / (2 * 0.04**2))
    
    # Right side with smaller peak
    f_init += 0.5 * np.exp(-((x - 0.12)**2) / (2 * 0.035**2))
    
    # Add oscillation for structure
    oscillation = 0.1 * np.sin(20 * np.pi * x)
    
    f_init = f_init + oscillation
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.9
    
    # Add noise
    noise = np.random.normal(0, 0.005, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_multi_peak_structure(n: int) -> np.ndarray:
    """Create a multi-peak structure for maximum convolution efficiency"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Multi-peak structure designed for excellent convolution properties
    f_init = np.zeros(n)
    
    # Central strong peak
    f_init += 1.2 * np.exp(-x**2 / (2 * 0.02**2))
    
    # Multiple strategically placed peaks
    f_init += 0.6 * np.exp(-((x - 0.1)**2) / (2 * 0.03**2))
    f_init += 0.6 * np.exp(-((x + 0.1)**2) / (2 * 0.03**2))
    f_init += 0.4 * np.exp(-((x - 0.18)**2) / (2 * 0.025**2))
    f_init += 0.4 * np.exp(-((x + 0.18)**2) / (2 * 0.025**2))
    
    # Add oscillation for better structure
    oscillation = 0.12 * np.sin(25 * np.pi * x)
    
    f_init = f_init + oscillation
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.88
    
    # Add noise
    noise = np.random.normal(0, 0.003, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def ultimate_aggressive_gradient_optimization(f_init, max_iter=7000, patience=50):
    """Ultimate aggressive gradient optimization with maximum parameters"""
    f_opt = jnp.array(f_init)
    
    # ULTIMATE AGGRESSIVE Adam parameters for maximum speed and exploration
    learning_rate = 0.4  # Extremely high learning rate
    beta1 = 0.998        # Very high momentum
    beta2 = 0.999999     # Nearly perfect momentum  
    epsilon = 1e-8
    
    # Initialize Adam moments
    m = jnp.zeros_like(f_opt)
    v = jnp.zeros_like(f_opt)
    
    best_c2 = -float('inf')
    best_f = f_opt
    patience_counter = 0
    last_improvement_iter = 0
    
    # Optimization loop with ultimate aggressive convergence tracking
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
        
        # Track best solution with ultimate aggressive criteria
        if c2_val > best_c2:
            best_c2 = c2_val
            best_f = f_opt
            patience_counter = 0
            last_improvement_iter = i
        else:
            patience_counter += 1
            
        # Ultimate aggressive early stopping
        if patience_counter >= patience and i - last_improvement_iter > 100:
            # Ultimate aggressive learning rate reduction
            if i > 500:
                learning_rate *= 0.96
            if patience_counter >= patience * 4:
                break
    
    return best_f, best_c2

def construct_function() -> list[float]:
    """
    Ultimate aggressive hybrid optimization approach that maximizes C2 effectively within time constraints.
    Uses ultimate aggressive optimization and comprehensive search strategies to exceed benchmark.
    """
    
    start_time = time.time()
    
    # Problem parameters - using maximum resolution possible within time limits
    n_steps = 2500  # MAXIMUM resolution for better optimization
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Strategy 1: Ultimate aggressive multiple restarts with diverse initialization patterns
    init_strategies = [
        ("ultimate", create_ultimate_pattern),
        ("super_optimized", create_super_optimized_pattern),
        ("concentrated_peak", create_concentrated_peak_pattern),
        ("balanced_asymmetric", create_balanced_asymmetric_pattern),
        ("multi_peak_structure", create_multi_peak_structure),
        ("random", lambda n: np.abs(np.random.normal(0.5, 0.2, n))),
        ("gaussian", lambda n: np.exp(-((np.linspace(-0.25, 0.25, n))**2) / (2 * 0.05**2))),
    ]
    
    # Try EVEN MORE restarts with different strategies - ultimate aggressive exploration
    num_restarts = 60  # Even more restarts for better chance of global optimum
    for restart in range(num_restarts):
        if time.time() - start_time > 35:  # Leave more buffer time
            break
            
        # Choose initialization strategy
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        f_init = init_func(n_steps)
        
        # Run ultimate aggressive gradient optimization
        f_opt, c2_val = ultimate_aggressive_gradient_optimization(f_init, max_iter=6000, patience=40)
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_solution = f_opt
    
    # Strategy 2: Ultimate aggressive differential evolution as final refinement
    if best_solution is None or best_c2 < 0.97:
        try:
            # Use the best solution found so far or a mathematically-informed initialization
            if best_solution is not None:
                f_init = np.array(best_solution)
            else:
                f_init = create_ultimate_pattern(n_steps)
            
            def objective(f_vals):
                f_vals = np.maximum(f_vals, 0)
                c2 = compute_c2_jax(f_vals)
                return -c2  # Minimize negative to maximize C2
            
            # Bounds for each parameter (non-negative)
            bounds = [(0, 5.0) for _ in range(len(f_init))]
            
            # Run ultimate aggressive differential evolution with maximum effort
            result = differential_evolution(
                objective,
                bounds,
                maxiter=120,  # MANY more iterations for thorough search
                popsize=50,   # VERY large population for better exploration
                mutation=(0.99, 1.0),  # Extremely high mutation rate for maximum exploration
                recombination=0.995,   # Very high recombination for maximum mixing
                seed=42,
                disp=False,
                atol=1e-13,  # Extremely tight tolerance
                rtol=1e-13
            )
            
            if result.success:
                final_f = np.maximum(result.x, 0)
                de_c2 = compute_c2_jax(final_f)
                if de_c2 > best_c2:
                    best_c2 = de_c2
                    best_solution = jnp.array(final_f)
        except Exception:
            pass
    
    # Strategy 3: Final ultimate intense gradient optimization if time allows
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            f_opt, c2_val = ultimate_aggressive_gradient_optimization(f_init, max_iter=4000, patience=30)
            
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
