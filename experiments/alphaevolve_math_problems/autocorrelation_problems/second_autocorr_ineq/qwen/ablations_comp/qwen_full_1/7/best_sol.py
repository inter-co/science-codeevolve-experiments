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

def compute_autoconvolution_exact(f_values: list[float]) -> tuple[float, float, float]:
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
    # ||g||₁ = sum(|g_i|) * dx (as per evaluator specification)
    # ||g||∞ = max(|g_i|)
    
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

def compute_c2_exact(f_values: list[float]) -> float:
    """
    Compute C2 = ||g||₂² / (||g||₁ · ||g||∞) using exact evaluator method
    """
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_exact(f_values)
    
    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    return norm_2_squared / (norm_1 * norm_inf)

@jit
def compute_c2_jax(f):
    """Compute C2 using JAX for automatic differentiation"""
    # Ensure non-negative values
    f = jnp.maximum(f, 0.0)
    
    # Compute autoconvolution using JAX
    g = jnp.convolve(f, f, mode='full')
    
    # Extract central portion (same as the exact method)
    center = len(g) // 2
    half_len = len(f) - 1
    g_trimmed = g[center - half_len:center + half_len + 1]
    
    # Compute norms
    l2_squared = jnp.sum(g_trimmed**2)
    l1_norm = jnp.sum(jnp.abs(g_trimmed))
    l_inf_norm = jnp.max(jnp.abs(g_trimmed))
    
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
    """Create the most advanced mathematical pattern inspired by deep optimization theory"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a highly optimized pattern with multiple components
    # Based on mathematical analysis of extremal functions in harmonic analysis
    
    # Primary central peak
    f_init = 2.0 * np.exp(-x**2 / (2 * 0.015**2))
    
    # Secondary peaks for constructive interference
    f_init += 1.0 * np.exp(-((x - 0.15)**2) / (2 * 0.04**2))
    f_init += 1.0 * np.exp(-((x + 0.15)**2) / (2 * 0.04**2))
    
    # Fine structure peaks
    f_init += 0.5 * np.exp(-((x - 0.2)**2) / (2 * 0.02**2))
    f_init += 0.5 * np.exp(-((x + 0.2)**2) / (2 * 0.02**2))
    
    # Oscillation component for complex convolution
    f_init += 0.15 * np.sin(25 * np.pi * x)
    
    # Additional modulation
    f_init += 0.08 * np.cos(10 * np.pi * x) * np.exp(-x**2 / 0.03)
    
    # Normalize to reasonable scale
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add noise for exploration
    noise = np.random.normal(0, 0.004, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_multi_scale_pattern(n: int) -> np.ndarray:
    """Create a multi-scale pattern for rich convolution properties"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Mix of scales for complex behavior
    f_init = np.zeros(n)
    
    # Broad central region
    f_init += 1.2 * np.exp(-x**2 / (2 * 0.08**2))
    
    # Medium peaks
    f_init += 0.8 * np.exp(-((x - 0.12)**2) / (2 * 0.04**2))
    f_init += 0.8 * np.exp(-((x + 0.12)**2) / (2 * 0.04**2))
    
    # Fine structure
    f_init += 0.4 * np.exp(-((x - 0.18)**2) / (2 * 0.02**2))
    f_init += 0.4 * np.exp(-((x + 0.18)**2) / (2 * 0.02**2))
    
    # Add oscillation for rich convolution
    f_init += 0.1 * np.sin(18 * np.pi * x)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.8
    
    # Add noise
    noise = np.random.normal(0, 0.003, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_concentrated_peak_pattern(n: int) -> np.ndarray:
    """Create a highly concentrated peak pattern for maximum L2 norm"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Extremely concentrated mass for strong L2 norm
    f_init = 2.5 * np.exp(-x**2 / (2 * 0.012**2))
    
    # Add secondary peaks for better convolution
    f_init += 0.7 * np.exp(-((x - 0.1)**2) / (2 * 0.03**2))
    f_init += 0.7 * np.exp(-((x + 0.1)**2) / (2 * 0.03**2))
    
    # Add oscillation
    f_init += 0.1 * np.sin(20 * np.pi * x)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.75
    
    # Add noise
    noise = np.random.normal(0, 0.002, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_extremely_aggressive_gradient_optimization(f_init, max_iter=4000, patience=150):
    """Extremely aggressive gradient optimization with maximum learning rate"""
    f_opt = jnp.array(f_init)
    
    # Very aggressive parameters to maximize convergence speed
    learning_rate = 0.3  # Even higher learning rate
    beta1 = 0.99         # Nearly full momentum
    beta2 = 0.9999       # Super high momentum
    epsilon = 1e-8
    
    # Initialize Adam moments
    m = jnp.zeros_like(f_opt)
    v = jnp.zeros_like(f_opt)
    
    best_c2 = -float('inf')
    best_f = f_opt
    patience_counter = 0
    last_improvement_iter = 0
    adaptive_lr = learning_rate
    
    # Optimization loop with extreme aggressiveness
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
        f_opt = f_opt + adaptive_lr * m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        # Ensure non-negativity
        f_opt = jnp.maximum(f_opt, 0.0)
        
        # Track best solution with very aggressive criteria
        if c2_val > best_c2:
            best_c2 = c2_val
            best_f = f_opt
            patience_counter = 0
            last_improvement_iter = i
        else:
            patience_counter += 1
            
        # Very aggressive adaptive learning rate reduction
        if patience_counter > 30 and i > 300:
            adaptive_lr *= 0.98
        if patience_counter > 100 and i > 1000:
            adaptive_lr *= 0.95
        if patience_counter > 200 and i > 2000:
            adaptive_lr *= 0.92
            
        # Extremely aggressive early stopping
        if patience_counter >= patience and i - last_improvement_iter > 200:
            break
    
    return best_f, best_c2

def construct_function() -> list[float]:
    """
    Extreme optimization approach that pushes beyond normal boundaries to beat the benchmark.
    """
    
    start_time = time.time()
    
    # Use higher resolution and more aggressive optimization
    n_steps = 1500  # Even higher resolution for better optimization
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Strategy 1: Many more restarts with the most aggressive optimization
    init_strategies = [
        ("advanced_math", create_advanced_mathematical_pattern),
        ("multi_scale", create_multi_scale_pattern),
        ("concentrated", create_concentrated_peak_pattern),
        ("random", lambda n: np.abs(np.random.normal(0.6, 0.25, n))),
        ("gaussian", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 / 0.012)),
    ]
    
    # Use even more restarts for maximum exploration
    num_restarts = 30  # Significantly more restarts
    for restart in range(num_restarts):
        if time.time() - start_time > 55:  # Leave buffer time
            break
            
        # Choose initialization strategy
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        f_init = init_func(n_steps)
        
        # Run extremely aggressive gradient optimization with even more iterations
        f_opt, c2_val = create_extremely_aggressive_gradient_optimization(
            f_init, 
            max_iter=3500,  # Even more iterations
            patience=100    # Less patience for faster convergence
        )
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_solution = f_opt
    
    # Strategy 2: Final refinement with differential evolution on best solution
    if best_solution is not None and time.time() - start_time < 50:
        try:
            # Convert to numpy for scipy compatibility
            f_init = np.array(best_solution)
            
            def objective(f_vals):
                f_vals = np.maximum(f_vals, 0)
                # Use the exact evaluator for consistency
                c2 = compute_c2_exact(f_vals.tolist())
                return -c2  # Minimize negative to maximize C2
            
            # Bounds for each parameter (non-negative)
            bounds = [(0, 3.0) for _ in range(len(f_init))]
            
            # Run differential evolution with very high precision
            result = differential_evolution(
                objective,
                bounds,
                maxiter=40,  # More iterations for better refinement
                popsize=20,   # Even larger population size
                mutation=(0.9, 1.0),  # Very high mutation
                recombination=0.95,   # Even better recombination
                seed=42,
                disp=False,
                atol=1e-9,
                rtol=1e-9
            )
            
            if result.success:
                final_f = np.maximum(result.x, 0)
                de_c2 = compute_c2_exact(final_f.tolist())
                if de_c2 > best_c2:
                    return final_f.tolist()
        except Exception:
            pass
    
    # Strategy 3: One final super aggressive optimization run
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            f_opt, c2_val = create_extremely_aggressive_gradient_optimization(
                f_init, 
                max_iter=2000, 
                patience=50
            )
            
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
    
    # Final evaluation and conversion to list using exact method
    final_c2 = compute_c2_exact(list(best_solution))
    
    return list(best_solution)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
