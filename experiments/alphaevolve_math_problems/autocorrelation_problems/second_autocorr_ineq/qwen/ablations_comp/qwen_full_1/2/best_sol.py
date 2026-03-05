# EVOLVE-BLOCK-START

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy import signal
from scipy.optimize import differential_evolution
import random
import time
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)
jax.config.update('jax_enable_x64', True)

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation using proper numerical integration:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f (autoconvolution)
    Uses the evaluator's exact piecewise linear integration approach.
    """
    # Convert to numpy array for easier handling
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using convolution
    g = signal.convolve(f, f, mode='full')
    
    # Compute step size for the original function
    n = len(f)
    dx = 0.5 / (n - 1) if n > 1 else 1.0
    
    # Scale the convolution result properly - this is crucial for correct norms
    # The convolution needs to be scaled by dx to account for the step width
    g = g * dx
    
    # Compute the three norms using evaluator's method:
    # ||g||₂² = sum of (h/3)(y1² + y1*y2 + y2²) contributions for piecewise integration
    # ||g||₁ = sum(|g|) / (len(g) + 1) 
    # ||g||∞ = max |g|
    
    g_abs = np.abs(g)
    
    # Compute ||g||₂² using evaluator's exact piecewise integration method
    # Formula: (h/3)(y1² + y1*y2 + y2²) for adjacent segments
    g_2_norm_squared = 0.0
    if len(g) >= 2:
        for i in range(len(g) - 1):
            h = dx
            y1 = g_abs[i]
            y2 = g_abs[i+1]
            # Evaluator's exact formula for piecewise linear integration of g²:
            # (h/3)(y1² + y1*y2 + y2²) - this is correct for quadratic integration
            g_2_norm_squared += (h/3) * (y1**2 + y1*y2 + y2**2)
    
    # ||g||₁ = sum(|g|) / (len(g) + 1) as specified in evaluator
    g_1_norm = np.sum(g_abs) / (len(g) + 1)
    
    # ||g||∞ = max |g|
    g_inf_norm = np.max(g_abs)
    
    return g_2_norm_squared, g_1_norm, g_inf_norm

def compute_c2(f_values: list[float]) -> float:
    """
    Compute C2 = ||g||₂² / (||g||₁ · ||g||∞) where g = f*f
    """
    try:
        g_2_sq, g_1, g_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_1 <= 1e-15 or g_inf <= 1e-15:
            return 0.0
            
        c2 = g_2_sq / (g_1 * g_inf)
        return c2
    except Exception:
        return 0.0

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

def create_advanced_mathematical_initialization(n: int) -> np.ndarray:
    """
    Create an advanced mathematical pattern inspired by successful approaches from inspirations.
    This pattern is designed to maximize the favorable properties for autoconvolution.
    """
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a sophisticated pattern with multiple peaks and oscillations
    # This is inspired by the most successful patterns from the inspirations
    f_init = np.zeros(n)
    
    # Primary bell shape (centered) with optimized parameters
    sigma_primary = 0.05
    f_init += 0.7 * np.exp(-x**2 / (2 * sigma_primary**2))
    
    # Secondary peaks for better structure and overlap
    sigma_secondary = 0.1
    f_init += 0.2 * np.exp(-((x - 0.1)**2) / (2 * sigma_secondary**2))
    f_init += 0.2 * np.exp(-((x + 0.1)**2) / (2 * sigma_secondary**2))
    
    # Add oscillation to break symmetry and encourage better convolution properties
    f_init += 0.08 * np.sin(20 * np.pi * x) * np.exp(-x**2 / 0.02)
    
    # Normalize to reasonable scale
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.9
    
    # Add controlled noise for exploration
    noise = np.random.normal(0, 0.01, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_peak_cluster_initialization(n: int) -> np.ndarray:
    """
    Create a peak cluster pattern that focuses on creating high concentration 
    in specific areas for better autoconvolution properties.
    """
    x = np.linspace(-0.25, 0.25, n)
    
    # Create clustered peaks that will produce favorable autoconvolution
    f_init = np.zeros(n)
    
    # High amplitude peaks in strategic locations
    peaks = [
        (0.0, 0.03, 1.2),      # Central peak
        (-0.15, 0.04, 0.8),    # Left peak
        (0.15, 0.04, 0.8),     # Right peak
        (-0.07, 0.02, 0.5),    # Mid-left peak
        (0.07, 0.02, 0.5),     # Mid-right peak
    ]
    
    for center, width, height in peaks:
        f_init += height * np.exp(-((x - center)**2) / (2 * width**2))
    
    # Add oscillation for structure
    oscillation = 0.1 * np.sin(15 * np.pi * x)
    f_init = np.maximum(f_init + oscillation, 0)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add noise
    noise = np.random.normal(0, 0.015, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_high_concentration_initialization(n: int) -> np.ndarray:
    """
    Create a pattern with high concentration in the center and smooth transitions.
    This tends to produce flatter autoconvolution profiles which improve C2.
    """
    x = np.linspace(-0.25, 0.25, n)
    
    # Central plateau with smooth transitions
    f_init = np.zeros(n)
    
    # Main central region with high values
    central_mask = np.abs(x) <= 0.12
    f_init[central_mask] = 1.0
    
    # Smooth transition regions
    transition_left = (x > 0.12) & (x <= 0.2)
    transition_right = (x < -0.12) & (x >= -0.2)
    
    # Exponential decay for smooth transitions
    f_init[transition_left] = np.exp(-((x[transition_left] - 0.12) / 0.05)**2)
    f_init[transition_right] = np.exp(-((x[transition_right] + 0.12) / 0.05)**2)
    
    # Add some oscillations to break symmetry but maintain structure
    oscillation = 0.1 * np.sin(15 * np.pi * x) * np.exp(-x**2 / 0.05)
    f_init = np.maximum(f_init + oscillation, 0)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add small noise for exploration
    noise = np.random.normal(0, 0.01, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def aggressive_gradient_optimization(f_init, max_iter=5000, patience=50):
    """
    Run highly aggressive gradient optimization with fast convergence and adaptive parameters
    """
    f_opt = jnp.array(f_init)
    
    # Very aggressive Adam parameters for rapid convergence
    learning_rate = 0.2  # Much higher learning rate
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

def enhanced_hybrid_optimization_approach() -> list[float]:
    """
    Enhanced hybrid approach that leverages aggressive optimization and sophisticated 
    initialization patterns to maximize C2 effectively within time constraints.
    """
    start_time = time.time()
    
    # Use a high-resolution pattern for better optimization
    n_steps = 1200  # Higher resolution for better results
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Multiple highly specialized initialization strategies
    init_strategies = [
        ("advanced_math", create_advanced_mathematical_initialization),
        ("peak_cluster", create_peak_cluster_initialization),
        ("high_concentration", create_high_concentration_initialization),
        ("random", lambda n: np.abs(np.random.normal(0.6, 0.2, n))),
    ]
    
    # Strategy: Multiple restarts with different patterns
    num_restarts = 30  # More restarts for better exploration
    
    for restart in range(num_restarts):
        if time.time() - start_time > 55:  # Leave buffer time
            break
            
        # Choose initialization strategy
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        f_init = init_func(n_steps)
        
        # Run extremely aggressive gradient optimization
        f_opt, c2_val = aggressive_gradient_optimization(f_init, max_iter=4000, patience=50)
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_solution = f_opt
    
    # Strategy 2: Final refinement with ultra-aggressive optimization
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            # Run ultra-aggressive optimization
            f_opt, c2_val = aggressive_gradient_optimization(f_init, max_iter=5000, patience=30)
            
            if c2_val > best_c2:
                best_c2 = c2_val
                best_solution = f_opt
        except Exception:
            pass
    
    # Strategy 3: Final differential evolution refinement with aggressive parameters
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
            
            # Run differential evolution with more aggressive parameters
            result = differential_evolution(
                objective,
                bounds,
                maxiter=35,  # More iterations for better refinement
                popsize=25,   # Even larger population
                mutation=(0.5, 1.0),  # Better mutation range
                recombination=0.9,    # Even higher recombination
                seed=42,
                disp=False
            )
            
            if result.success:
                refined_f = np.maximum(result.x, 0)
                # Evaluate the DE result
                de_c2 = compute_c2_jax(refined_f)
                if de_c2 > best_c2:
                    best_c2 = de_c2
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
    return enhanced_hybrid_optimization_approach()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
