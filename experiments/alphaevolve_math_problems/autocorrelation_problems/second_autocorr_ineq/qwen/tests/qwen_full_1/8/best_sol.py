# EVOLVE-BLOCK-START

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy import signal
from scipy.optimize import differential_evolution
import time
from typing import List
import random
import nevergrad as ng
from scipy.special import legendre

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)
jax.config.update('jax_enable_x64', True)

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
    
    # ||g||₁ as specified in problem: sum(|g|) / (len(g) + 1)
    norm_1 = np.sum(g_abs) / (len(g) + 1)
    
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

def create_advanced_mathematical_pattern(n: int) -> np.ndarray:
    """Create an advanced mathematical pattern inspired by successful approaches"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Create sophisticated pattern with mathematical structure
    f_init = np.zeros(n)
    
    # Central peak with very sharp focus for high L2^2
    f_init += 2.0 * np.exp(-x**2 / (2 * 0.01**2))
    
    # Secondary peaks for constructive interference
    f_init += 1.0 * np.exp(-((x - 0.12)**2) / (2 * 0.03**2))
    f_init += 1.0 * np.exp(-((x + 0.12)**2) / (2 * 0.03**2))
    
    # Fine structure for rich convolution properties
    f_init += 0.6 * np.exp(-((x - 0.18)**2) / (2 * 0.02**2))
    f_init += 0.6 * np.exp(-((x + 0.18)**2) / (2 * 0.02**2))
    
    # Oscillatory component with careful amplitude
    f_init += 0.25 * np.sin(35 * np.pi * x) * np.exp(-x**2 / 0.05)
    
    # Asymmetric modulation to break degeneracy
    asymmetry = 0.1 * np.sin(18 * np.pi * x) * np.exp(-x**2 / 0.03)
    f_init += asymmetry
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.95
    
    # Add noise for exploration
    noise = np.random.normal(0, 0.002, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_legendre_based_pattern(n: int) -> np.ndarray:
    """Create pattern based on Legendre polynomials for good approximation properties"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Use Legendre polynomials for good distribution properties
    f_init = np.zeros(n)
    
    # Combine several Legendre polynomials of different degrees
    for i in range(1, 6):
        # Use Legendre polynomials P_i(x) scaled appropriately
        poly = legendre(i)(x * 2)  # Scale to [-0.25, 0.25] domain
        # Add with decreasing weights to create interesting structure
        f_init += (1.0 / (i * 2)) * poly * np.exp(-x**2 / 0.01)
    
    # Add oscillatory component
    f_init += 0.2 * np.sin(20 * np.pi * x)
    
    # Ensure positivity and normalize
    f_init = np.maximum(f_init, 0)
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add noise for exploration
    noise = np.random.normal(0, 0.002, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_multi_scale_peak_pattern(n: int) -> np.ndarray:
    """Create a multi-scale peak pattern with varying amplitudes and spacings"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Create pattern with multiple scales for rich convolution properties
    f_init = np.zeros(n)
    
    # Very broad central peak
    f_init += 1.2 * np.exp(-x**2 / (2 * 0.08**2))
    
    # Medium peaks with high amplitudes
    f_init += 1.0 * np.exp(-((x - 0.1)**2) / (2 * 0.04**2))
    f_init += 1.0 * np.exp(-((x + 0.1)**2) / (2 * 0.04**2))
    
    # Fine structure with multiple oscillations
    f_init += 0.3 * np.exp(-((x - 0.18)**2) / (2 * 0.02**2))
    f_init += 0.3 * np.exp(-((x + 0.18)**2) / (2 * 0.02**2))
    f_init += 0.25 * np.exp(-((x - 0.08)**2) / (2 * 0.015**2))
    f_init += 0.25 * np.exp(-((x + 0.08)**2) / (2 * 0.015**2))
    
    # Add rich oscillation for convolution properties
    f_init += 0.22 * np.sin(25 * np.pi * x)
    f_init += 0.15 * np.cos(18 * np.pi * x)
    
    # Add some asymmetry to break degeneracy
    asymmetry = 0.08 * np.sin(14 * np.pi * x) * np.exp(-x**2 / 0.018)
    f_init += asymmetry
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.9
    
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

def create_hybrid_multi_component_pattern(n: int) -> np.ndarray:
    """Create a hybrid pattern combining multiple mathematical approaches"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a complex hybrid pattern with enhanced structure
    f_init = np.zeros(n)
    
    # Main central peak with very high amplitude
    f_init += 1.8 * np.exp(-x**2 / (2 * 0.015**2))
    
    # Side peaks with strong amplitudes
    f_init += 1.3 * np.exp(-((x - 0.12)**2) / (2 * 0.022**2))
    f_init += 1.3 * np.exp(-((x + 0.12)**2) / (2 * 0.022**2))
    
    # Fine structure with many components
    f_init += 0.4 * np.exp(-((x - 0.18)**2) / (2 * 0.01**2))
    f_init += 0.4 * np.exp(-((x + 0.18)**2) / (2 * 0.01**2))
    f_init += 0.3 * np.exp(-((x - 0.08)**2) / (2 * 0.008**2))
    f_init += 0.3 * np.exp(-((x + 0.08)**2) / (2 * 0.008**2))
    f_init += 0.25 * np.exp(-((x - 0.14)**2) / (2 * 0.012**2))
    f_init += 0.25 * np.exp(-((x + 0.14)**2) / (2 * 0.012**2))
    
    # Rich oscillations for convolution enhancement
    f_init += 0.2 * np.sin(28 * np.pi * x)
    f_init += 0.18 * np.cos(20 * np.pi * x)
    f_init += 0.12 * np.sin(16 * np.pi * x)
    
    # Add asymmetry and complex modulation
    asymmetry = 0.08 * np.sin(12 * np.pi * x) * np.exp(-x**2 / 0.015)
    f_init += asymmetry
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.92
    
    # Add noise
    noise = np.random.normal(0, 0.0025, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def ultra_aggressive_gradient_optimization(f_init, max_iter=10000, patience=50):
    """Ultra-aggressive gradient optimization with extreme parameters"""
    f_opt = jnp.array(f_init)
    
    # Even more aggressive Adam parameters for rapid convergence
    learning_rate = 0.6  # Even higher learning rate
    beta1 = 0.995  # Very high momentum
    beta2 = 0.99999  # Extremely high momentum for variance
    epsilon = 1e-8
    
    # Initialize Adam moments
    m = jnp.zeros_like(f_opt)
    v = jnp.zeros_like(f_opt)
    
    best_c2 = -float('inf')
    best_f = f_opt
    patience_counter = 0
    last_improvement_iter = 0
    
    # Optimization loop with ultra-aggressive parameters
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
            
        # Early stopping if no improvement
        if patience_counter >= patience:
            if i - last_improvement_iter > 200:
                # Reduce learning rate more aggressively
                learning_rate *= 0.92
                patience_counter = 0
                if learning_rate < 1e-6:
                    break
            elif i > 500:
                # If we're still not improving after 500 iterations, reduce learning rate
                learning_rate *= 0.95
                patience_counter = 0
    
    return best_f, best_c2

def create_evolutionary_optimization(n_steps: int, max_evaluations: int = 1000) -> List[float]:
    """Use Nevergrad evolutionary optimization as fundamentally different approach"""
    
    def objective_function(params):
        # params is a 1D array of step heights
        f_vals = np.maximum(params, 0)
        c2 = compute_c2_jax(f_vals)
        return -c2  # Minimize negative to maximize C2
    
    # Define search space
    instrumentation = ng.p.Array(shape=(n_steps,), lower=0, upper=3.0)
    
    # Use a suitable evolutionary optimizer
    optimizer = ng.optimizers.DifferentialEvolution(
        instrumentation=instrumentation,
        budget=max_evaluations,
        num_workers=1
    )
    
    # Optimize
    recommendation = optimizer.minimize(objective_function)
    
    # Return the best solution
    best_params = recommendation.value
    return list(np.maximum(best_params, 0))

def enhanced_hybrid_optimization() -> list[float]:
    """
    Enhanced hybrid approach that maximizes C2 effectively within time constraints.
    Combines mathematical insights, aggressive optimization, and evolutionary methods.
    """
    
    start_time = time.time()
    
    # Use high resolution for better optimization potential
    n_steps = 1600  # Higher resolution for better results
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Strategy 1: Multiple sophisticated initialization strategies with aggressive optimization
    init_strategies = [
        ("advanced_math", create_advanced_mathematical_pattern),
        ("legendre", create_legendre_based_pattern),
        ("multi_scale", create_multi_scale_peak_pattern),
        ("concentrated", create_concentrated_mass_pattern),
        ("balanced_asymmetric", create_balanced_asymmetric_pattern),
        ("hybrid", create_hybrid_multi_component_pattern),
        ("random", lambda n: np.abs(np.random.normal(0.6, 0.2, n))),
        ("sinc_like", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 * 8) * np.cos(15 * np.pi * np.linspace(-0.25, 0.25, n))),
        ("wide_plateau", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 * 5) * (1 + 0.3 * np.sin(18 * np.pi * np.linspace(-0.25, 0.25, n)))),
    ]
    
    # Try different initialization strategies with ultra-aggressive optimization
    for strategy_name, init_func in init_strategies:
        if time.time() - start_time > 45:  # Leave buffer for final polish
            break
            
        try:
            # Create initialization
            f_init = init_func(n_steps)
            
            # Run ultra-aggressive optimization with more iterations
            f_opt, c2_val = ultra_aggressive_gradient_optimization(f_init, max_iter=8000, patience=80)
            
            if c2_val > best_c2:
                best_c2 = c2_val
                best_solution = f_opt
        except Exception:
            continue
    
    # Strategy 2: Differential evolution refinement for final polish
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
            
            # Run differential evolution with maximum thoroughness
            result = differential_evolution(
                objective,
                bounds,
                maxiter=150,  # More iterations for better refinement
                popsize=50,   # Larger population size for better exploration
                mutation=(0.98, 1.0),  # Very high mutation for more exploration
                recombination=0.95,   # Very high recombination for better mixing
                seed=42,
                disp=False,
                atol=1e-14,
                rtol=1e-14
            )
            
            if result.success:
                final_f = np.maximum(result.x, 0)
                de_c2 = compute_c2_jax(final_f)
                if de_c2 > best_c2:
                    best_c2 = de_c2
                    best_solution = jnp.array(final_f)
        except Exception:
            pass
    
    # Strategy 3: Add evolutionary optimization as fundamentally different approach
    if best_solution is not None and time.time() - start_time < 40:
        try:
            # Run evolutionary optimization with fewer evaluations to save time
            evol_solution = create_evolutionary_optimization(n_steps, max_evaluations=1000)
            evol_c2 = compute_c2(evol_solution)
            
            if evol_c2 > best_c2:
                best_c2 = evol_c2
                best_solution = jnp.array(evol_solution)
        except Exception:
            pass
    
    # Strategy 4: Final intensive optimization pass
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            # Run final optimization with maximum iterations and patience
            f_opt, c2_val = ultra_aggressive_gradient_optimization(f_init, max_iter=5000, patience=40)
            
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
    
    # Final evaluation using exact computation method
    final_c2 = compute_c2_jax(best_solution)
    
    return list(best_solution)

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses an enhanced hybrid approach that leverages mathematical insights
    and aggressive optimization strategies with evolutionary methods.
    """
    return enhanced_hybrid_optimization()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
