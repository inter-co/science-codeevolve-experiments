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

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
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

def compute_c2(f_values: list[float]) -> float:
    """
    Compute C2 = ||g||₂² / (||g||₁ · ||g||∞) using the exact numerical integration
    """
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    return norm_2_squared / (norm_1 * norm_inf)

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
    
    # Compute norms using exact formula from problem description
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
    # Based on successful patterns from inspirations - stronger central mass with better side structure
    f_init = np.zeros(n)
    
    # Central dominant peak - sharper than before to maximize concentration effect
    f_init += 2.1 * np.exp(-x**2 / (2 * 0.012**2))
    
    # Strategic side peaks with optimal spacing for constructive interference
    f_init += 1.2 * np.exp(-((x - 0.12)**2) / (2 * 0.03**2))
    f_init += 1.2 * np.exp(-((x + 0.12)**2) / (2 * 0.03**2))
    
    # Additional fine-scale structure for better convolution properties
    f_init += 0.8 * np.exp(-((x - 0.18)**2) / (2 * 0.018**2))
    f_init += 0.8 * np.exp(-((x + 0.18)**2) / (2 * 0.018**2))
    
    # Controlled oscillation for better convolution behavior
    f_init += 0.25 * np.sin(35 * np.pi * x) * np.exp(-x**2 / 0.04)
    
    # Normalize properly
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.95
    
    # Add small noise for exploration
    noise = np.random.normal(0, 0.0005, n)
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
    """Create a highly concentrated mass pattern with enhanced mathematical structure"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Extremely concentrated mass pattern with improved scaling
    f_init = 2.2 * np.exp(-x**2 / (2 * 0.013**2))
    
    # Secondary peaks with better positioning for constructive interference
    f_init += 0.8 * np.exp(-((x - 0.1)**2) / (2 * 0.035**2))
    f_init += 0.8 * np.exp(-((x + 0.1)**2) / (2 * 0.035**2))
    
    # Additional fine structure with precise scaling
    f_init += 0.5 * np.exp(-((x - 0.18)**2) / (2 * 0.02**2))
    f_init += 0.5 * np.exp(-((x + 0.18)**2) / (2 * 0.02**2))
    
    # Enhanced oscillation with better amplitude control
    f_init += 0.15 * np.sin(30 * np.pi * x) * np.exp(-x**2 / 0.03)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.9
    
    # Add noise
    noise = np.random.normal(0, 0.0015, n)
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

def create_sparse_peak_initialization(n: int) -> np.ndarray:
    """Create a sparse peak pattern that maximizes L2^2 term"""
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
    
    # Add small noise for exploration
    noise = np.random.normal(0, 0.01, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_aggressive_gradient_optimization(f_init, max_iter=5000, patience=50):
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

def enhanced_hybrid_optimization() -> list[float]:
    """
    Enhanced hybrid approach that maximizes C2 effectively within time constraints.
    Incorporates mathematical insights from inspirations and uses more aggressive optimization.
    """
    
    start_time = time.time()
    
    # Use higher resolution for better optimization potential
    n_steps = 1800
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Strategy 1: Use the most effective initialization patterns from all inspirations
    # Based on INSPIRATION PROGRAM 3, which shows more diverse and effective patterns
    init_strategies = [
        ("concentrated_mass", create_concentrated_mass_pattern),
        ("optimized_initial", create_optimized_initialization),
        ("multi_scale", create_multi_scale_pattern),
        ("sparse_peak", create_sparse_peak_initialization),
        ("balanced_asymmetric", create_balanced_asymmetric_pattern),
        # Add some patterns from INSPIRATION PROGRAM 2
        ("sinc_like", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 * 8) * np.cos(15 * np.pi * np.linspace(-0.25, 0.25, n))),
        ("gaussian_peak", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 * 5) * (1 + 0.3 * np.sin(18 * np.pi * np.linspace(-0.25, 0.25, n)))),
    ]
    
    # Try even fewer restarts but with ultra-aggressive optimization parameters
    num_restarts = 4  # Even fewer restarts but ultra-aggressive optimization
    for restart in range(num_restarts):
        if time.time() - start_time > 48:  # Leave buffer for final polish
            break
            
        # Choose initialization strategy
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        try:
            f_init = init_func(n_steps)
        except:
            # Fallback to a simple pattern if initialization fails
            f_init = np.ones(n_steps) * 0.5
            
        # Run ULTRA-aggressive gradient optimization with maximum patience
        f_opt, c2_val = create_aggressive_gradient_optimization(f_init, max_iter=12000, patience=100)
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_solution = f_opt
    
    # Strategy 2: Differential evolution refinement for final polish with very high precision
    if best_solution is not None and time.time() - start_time < 40:
        try:
            # Use the best solution found so far
            f_init = np.array(best_solution)
            
            def objective(f_vals):
                f_vals = np.maximum(f_vals, 0)
                c2 = compute_c2_jax(f_vals)
                return -c2  # Minimize negative to maximize C2
            
            # Bounds for each parameter (non-negative)
            bounds = [(0, 3.0) for _ in range(len(f_init))]
            
            # Run differential evolution with maximum thoroughness and precision
            result = differential_evolution(
                objective,
                bounds,
                maxiter=300,  # More iterations for better refinement
                popsize=80,   # Larger population size for better exploration
                mutation=(0.999, 1.0),  # Very high mutation for more exploration
                recombination=0.998,   # Very high recombination for better mixing
                seed=42,
                disp=False,
                atol=1e-16,
                rtol=1e-16
            )
            
            if result.success:
                final_f = np.maximum(result.x, 0)
                de_c2 = compute_c2_jax(final_f)
                if de_c2 > best_c2:
                    best_c2 = de_c2
                    best_solution = jnp.array(final_f)
        except Exception:
            pass
    
    # Strategy 3: Final ultra-aggressive optimization pass if time allows
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            # Run ultra-final optimization with maximum effort and patience
            f_opt, c2_val = create_aggressive_gradient_optimization(f_init, max_iter=15000, patience=50)
            
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
    
    # Final evaluation using the exact method consistent with the optimization process
    final_c2 = compute_c2_jax(best_solution)
    
    return list(best_solution)

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses an enhanced hybrid approach that leverages mathematical insights
    and aggressive optimization strategies.
    """
    return enhanced_hybrid_optimization()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
