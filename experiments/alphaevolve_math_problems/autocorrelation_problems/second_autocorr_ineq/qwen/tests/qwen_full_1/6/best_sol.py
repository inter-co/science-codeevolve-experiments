# EVOLVE-BLOCK-START

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy import signal
import time
from scipy.optimize import differential_evolution
import warnings
import random
from typing import List
from scipy.special import legendre

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)
jax.config.update('jax_enable_x64', True)

def compute_autoconvolution_exact(f_values: List[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation using the EXACT evaluator style method:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    This matches exactly what the evaluator expects.
    """
    if not f_values:
        return 0.0, 0.0, 0.0
    
    f = np.array(f_values)
    n = len(f)
    
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Compute autoconvolution g = f * f using scipy.signal.convolve
    g = signal.convolve(f, f, mode='full')
    
    # The evaluator works on the domain [-1/4, 1/4] which is 0.5 wide
    # With n points, the step size is 0.5/n
    dx = 0.5 / n
    
    # Trim to the central region corresponding to [-1/4, 1/4] 
    center_idx = len(g) // 2
    half_width = n - 1  # This ensures we get the right convolution size
    g_trimmed = g[center_idx - half_width:center_idx + half_width + 1]
    
    # Compute norms exactly as specified in the problem description:
    # ||g||₂² = sum of (h/3)(y1² + y1*y2 + y2²) for consecutive points
    # ||g||₁ = sum(|g|) * dx (as per evaluator specification)
    # ||g||∞ = max(|g|)
    
    g_abs = np.abs(g_trimmed)
    
    # Compute ||g||₂² using the exact trapezoidal-like integration method:
    # For each pair of consecutive points with heights y1, y2 and step size dx:
    # contribution = (dx/3)(y1² + y1*y2 + y2²)
    norm_2_squared = 0.0
    
    # First point (special case for single point)
    if len(g_abs) > 0:
        norm_2_squared += (dx / 3.0) * (g_abs[0]**2)
    
    # Middle points (consecutive pairs)
    for i in range(len(g_abs) - 1):
        y1, y2 = g_abs[i], g_abs[i+1]
        norm_2_squared += (dx / 3.0) * (y1**2 + y1*y2 + y2**2)
    
    # Compute ||g||₁ = sum(|g|) * dx (as per evaluator specification)
    norm_1 = np.sum(g_abs) * dx
    
    # Compute ||g||∞ = max(|g|)
    norm_inf = np.max(g_abs) if len(g_abs) > 0 else 0.0
    
    return norm_2_squared, norm_1, norm_inf

def compute_c2_exact(f_values: List[float]) -> float:
    """
    Compute C2 = ||g||₂² / (||g||₁ · ||g||∞) using exact evaluator method
    """
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_exact(f_values)
    
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
    """
    Create an optimized initialization pattern based on mathematical insights from inspirations.
    """
    x = np.linspace(-0.25, 0.25, n)
    
    # Pattern inspired by mathematical analysis of optimal convolution structures
    # Stronger central mass with better side structure from inspiration 3
    f_init = np.zeros(n)
    
    # Central dominant peak - sharper to maximize concentration effect
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

def create_concentrated_mass_pattern(n: int) -> np.ndarray:
    """
    Create a highly concentrated mass pattern with enhanced mathematical structure.
    """
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

def create_multi_scale_pattern(n: int) -> np.ndarray:
    """
    Create a multi-scale pattern combining different frequency components.
    """
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
    """
    Create a balanced asymmetric pattern for optimal convolution.
    """
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
    """
    Create a sparse peak pattern that maximizes L2^2 term.
    """
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

def create_legendre_based_pattern(n: int) -> np.ndarray:
    """
    Create pattern based on Legendre polynomials for good approximation properties.
    """
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

def create_hybrid_multi_component_pattern(n: int) -> np.ndarray:
    """
    Create a hybrid pattern combining multiple mathematical approaches.
    """
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

def create_aggressive_gradient_optimization(f_init, max_iter=6000, patience=50):
    """Run highly aggressive gradient optimization with fast convergence"""
    f_opt = jnp.array(f_init)
    
    # Very aggressive Adam parameters for rapid convergence - from inspiration 2
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

def create_evolutionary_optimization(n_steps: int, max_evaluations: int = 2000) -> List[float]:
    """Use Nevergrad evolutionary optimization as fundamentally different approach"""
    
    import nevergrad as ng
    
    def objective_function(params):
        # params is a 1D array of step heights
        f_vals = np.maximum(params, 0)
        c2 = compute_c2_jax(f_vals)
        return -c2  # Minimize negative to maximize C2
    
    # Define search space
    instrumentation = ng.p.Array(shape=(n_steps,), lower=0, upper=3.0)
    
    # Use a suitable evolutionary optimizer - from inspiration 1
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

def enhanced_hybrid_optimization():
    """
    Enhanced hybrid approach that maximizes C2 effectively within time constraints.
    Combines mathematical insights from inspirations with evolutionary optimization.
    """
    
    start_time = time.time()
    
    # Use high resolution for better optimization potential
    n_steps = 1600  # Even higher resolution for better results - from inspiration 3
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Strategy 1: More diverse initialization strategies with aggressive optimization
    init_strategies = [
        ("optimized_initial", create_optimized_initialization),
        ("concentrated_mass", create_concentrated_mass_pattern),
        ("multi_scale", create_multi_scale_pattern),
        ("balanced_asymmetric", create_balanced_asymmetric_pattern),
        ("sparse_peak", create_sparse_peak_initialization),
        ("legendre", create_legendre_based_pattern),
        ("hybrid", create_hybrid_multi_component_pattern),
        ("random", lambda n: np.abs(np.random.normal(0.6, 0.2, n))),  # Fallback random
        ("sinc_like", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 * 8) * np.cos(15 * np.pi * np.linspace(-0.25, 0.25, n))),
        ("gaussian_peak", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 * 5) * (1 + 0.3 * np.sin(18 * np.pi * np.linspace(-0.25, 0.25, n)))),
        ("sharp_peak", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 * 15) * np.cos(20 * np.pi * np.linspace(-0.25, 0.25, n))),
        ("cosine_peak", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 * 10) * np.cos(25 * np.pi * np.linspace(-0.25, 0.25, n))),
        ("multimodal", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 * 3) * (1 + 0.5 * np.sin(30 * np.pi * np.linspace(-0.25, 0.25, n)))),
        ("bump_function", lambda n: 1.5 * np.exp(-((np.linspace(-0.25, 0.25, n) - 0.05)**2) / (2 * 0.03**2)) + 
                                 1.5 * np.exp(-((np.linspace(-0.25, 0.25, n) + 0.05)**2) / (2 * 0.03**2))),
    ]
    
    # Try many more restarts to ensure we don't miss the global optimum
    num_restarts = 60  # Even more restarts for better exploration - from inspiration 3
    for restart in range(num_restarts):
        if time.time() - start_time > 45:  # Leave buffer for final polish
            break
            
        # Choose initialization strategy
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        f_init = init_func(n_steps)
        
        # Run very aggressive gradient optimization with more patience
        f_opt, c2_val = create_aggressive_gradient_optimization(f_init, max_iter=6000, patience=80)
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_solution = f_opt
    
    # Strategy 2: Differential evolution refinement for final polish - from inspiration 2
    if best_solution is not None and time.time() - start_time < 45:
        try:
            # Use the best solution found so far
            f_init = np.array(best_solution)
            
            def objective(f_vals):
                f_vals = np.maximum(f_vals, 0)
                c2 = compute_c2_jax(f_vals)
                return -c2  # Minimize negative to maximize C2
            
            # Bounds for each parameter (non-negative)
            bounds = [(0, 3.0) for _ in range(len(f_init))]
            
            # Run differential evolution with maximum thoroughness - inspired by Program 3's aggressive settings
            result = differential_evolution(
                objective,
                bounds,
                maxiter=150,  # Even more iterations for better refinement
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
    
    # Strategy 3: Add evolutionary optimization as fundamentally different approach - from inspiration 1
    # This is the key addition that distinguishes this approach
    if best_solution is not None and time.time() - start_time < 45:
        try:
            # Run evolutionary optimization with fewer evaluations to save time
            evol_solution = create_evolutionary_optimization(n_steps, max_evaluations=1000)
            evol_c2 = compute_c2_exact(evol_solution)
            
            if evol_c2 > best_c2:
                best_c2 = evol_c2
                best_solution = jnp.array(evol_solution)
        except Exception:
            pass
    
    # Strategy 4: Final intensive optimization pass with maximum patience
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            # Run final optimization with maximum iterations and patience
            f_opt, c2_val = create_aggressive_gradient_optimization(f_init, max_iter=5000, patience=40)
            
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
    final_c2 = compute_c2_exact(list(best_solution))
    
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
