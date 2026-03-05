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

def create_mathematically_optimized_pattern(n: int) -> np.ndarray:
    """Create a mathematically optimized pattern based on deep insights from inspiration programs"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a sophisticated pattern designed to maximize C2:
    # Based on mathematical analysis of extremal functions
    # 1. Central peak for strong L2 norm contribution
    # 2. Side peaks to create constructive interference in convolution
    # 3. Controlled oscillation to avoid excessive peakiness
    # 4. Specific amplitude ratios based on mathematical optimization
    
    # Central Gaussian with very high amplitude
    f_init = 2.0 * np.exp(-x**2 / (2 * 0.015**2))
    
    # Symmetric side peaks with moderate amplitudes
    f_init += 1.0 * np.exp(-((x - 0.12)**2) / (2 * 0.035**2))
    f_init += 1.0 * np.exp(-((x + 0.12)**2) / (2 * 0.035**2))
    
    # Add controlled oscillation to create beneficial convolution properties
    f_init += 0.15 * np.sin(25 * np.pi * x)
    
    # Add fine structure for enhanced convolution
    f_init += 0.08 * np.exp(-((x - 0.08)**2) / (2 * 0.015**2))
    f_init += 0.08 * np.exp(-((x + 0.08)**2) / (2 * 0.015**2))
    
    # Normalize to reasonable scale
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add small noise for exploration
    noise = np.random.normal(0, 0.004, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_multi_scale_optimized_pattern(n: int) -> np.ndarray:
    """Create a multi-scale pattern optimized for convolution behavior"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Mix different scales to create complex convolution behavior
    f_init = np.zeros(n)
    
    # Broad central peak
    f_init += 1.0 * np.exp(-x**2 / (2 * 0.08**2))
    
    # Medium peaks
    f_init += 0.8 * np.exp(-((x - 0.12)**2) / (2 * 0.04**2))
    f_init += 0.8 * np.exp(-((x + 0.12)**2) / (2 * 0.04**2))
    
    # Fine structure with oscillations
    f_init += 0.3 * np.exp(-((x - 0.18)**2) / (2 * 0.02**2))
    f_init += 0.3 * np.exp(-((x + 0.18)**2) / (2 * 0.02**2))
    
    # Add oscillation for rich convolution properties
    f_init += 0.18 * np.sin(20 * np.pi * x)
    
    # Add some asymmetry to break degeneracy
    asymmetry = 0.08 * np.sin(15 * np.pi * x) * np.exp(-x**2 / 0.03)
    f_init += asymmetry
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.8
    
    # Add noise
    noise = np.random.normal(0, 0.003, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_concentrated_peak_pattern(n: int) -> np.ndarray:
    """Create a highly concentrated peak pattern"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Extremely concentrated mass for strong L2 norm
    f_init = 2.5 * np.exp(-x**2 / (2 * 0.01**2))
    
    # Add secondary peaks
    f_init += 0.8 * np.exp(-((x - 0.1)**2) / (2 * 0.03**2))
    f_init += 0.8 * np.exp(-((x + 0.1)**2) / (2 * 0.03**2))
    
    # Add oscillation
    f_init += 0.12 * np.sin(20 * np.pi * x)
    
    # Add fine structure
    f_init += 0.05 * np.exp(-((x - 0.08)**2) / (2 * 0.015**2))
    f_init += 0.05 * np.exp(-((x + 0.08)**2) / (2 * 0.015**2))
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.75
    
    # Add noise
    noise = np.random.normal(0, 0.002, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_asymmetric_balanced_pattern(n: int) -> np.ndarray:
    """Create an asymmetric pattern that balances different convolution properties"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Asymmetric structure with careful balance
    f_init = np.zeros(n)
    
    # Left side - sharper peak
    left_region = x < 0
    f_init[left_region] = 1.5 * np.exp(-x[left_region]**2 / (2 * 0.015**2))
    
    # Right side - broader peak  
    right_region = x >= 0
    f_init[right_region] = 1.0 * np.exp(-x[right_region]**2 / (2 * 0.05**2))
    
    # Add modulation to create beneficial convolution
    modulation = 0.15 * np.sin(15 * np.pi * x) * np.exp(-x**2 / 0.025)
    f_init += modulation
    
    # Add fine structure
    f_init += 0.08 * np.exp(-((x - 0.05)**2) / (2 * 0.01**2))
    f_init += 0.08 * np.exp(-((x + 0.05)**2) / (2 * 0.01**2))
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    # Add noise for exploration
    noise = np.random.normal(0, 0.004, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_optimized_bimodal_pattern(n: int) -> np.ndarray:
    """Create a bimodal pattern with two well-separated peaks"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Two separated peaks to maximize convolution energy
    f_init = 1.2 * np.exp(-((x - 0.12)**2) / (2 * 0.03**2))
    f_init += 1.2 * np.exp(-((x + 0.12)**2) / (2 * 0.03**2))
    
    # Add oscillation to enhance convolution properties
    f_init += 0.18 * np.sin(25 * np.pi * x)
    
    # Add central peak for additional energy
    f_init += 0.8 * np.exp(-x**2 / (2 * 0.018**2))
    
    # Add fine structure
    f_init += 0.1 * np.exp(-((x - 0.06)**2) / (2 * 0.012**2))
    f_init += 0.1 * np.exp(-((x + 0.06)**2) / (2 * 0.012**2))
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.8
    
    # Add noise
    noise = np.random.normal(0, 0.003, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_aggressive_gradient_optimization(f_init, max_iter=3000, patience=100):
    """Aggressive gradient optimization with improved parameters and faster convergence"""
    f_opt = jnp.array(f_init)
    
    # Very aggressive Adam parameters for rapid convergence
    learning_rate = 0.3  # Even higher learning rate for faster convergence
    beta1 = 0.99         # Extremely high momentum
    beta2 = 0.9999       # Nearly perfect momentum
    epsilon = 1e-8
    
    # Initialize Adam moments
    m = jnp.zeros_like(f_opt)
    v = jnp.zeros_like(f_opt)
    
    best_c2 = -float('inf')
    best_f = f_opt
    patience_counter = 0
    last_improvement_iter = 0
    adaptive_lr = learning_rate
    
    # Optimization loop with aggressive convergence tracking
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
        
        # Track best solution with aggressive criteria
        if c2_val > best_c2:
            best_c2 = c2_val
            best_f = f_opt
            patience_counter = 0
            last_improvement_iter = i
        else:
            patience_counter += 1
            
        # More aggressive adaptive learning rate reduction
        if patience_counter > 30 and i > 300:
            adaptive_lr *= 0.98
        elif patience_counter > 100 and i > 1000:
            adaptive_lr *= 0.95
        elif patience_counter > 200 and i > 2000:
            adaptive_lr *= 0.92
            
        # More aggressive early stopping
        if patience_counter >= patience and i - last_improvement_iter > 200:
            break
    
    return best_f, best_c2

def construct_function() -> list[float]:
    """
    Advanced hybrid optimization approach that maximizes C2 effectively within time constraints.
    Incorporates the best elements from inspirations with aggressive optimization strategies.
    """
    
    start_time = time.time()
    
    # Problem parameters - optimized for time and quality balance
    n_steps = 1200  # Slightly increased for better resolution
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Multiple initialization strategies to ensure diverse exploration
    init_strategies = [
        ("math_optimized", create_mathematically_optimized_pattern),
        ("multi_scale", create_multi_scale_optimized_pattern),
        ("concentrated", create_concentrated_peak_pattern),
        ("asymmetric_balanced", create_asymmetric_balanced_pattern),
        ("bimodal", create_optimized_bimodal_pattern),
        ("random", lambda n: np.abs(np.random.normal(0.5, 0.2, n))),
        ("gaussian", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 / 0.01)),
        ("sine_modulated", lambda n: np.exp(-np.linspace(-0.25, 0.25, n)**2 / 0.02) * 
                              (1.0 + 0.2 * np.sin(20 * np.pi * np.linspace(-0.25, 0.25, n)))),
    ]
    
    # Strategy 1: Multiple restarts with different initialization patterns
    num_restarts = 20  # Increased for better exploration
    for restart in range(num_restarts):
        if time.time() - start_time > 55:  # Leave buffer time
            break
            
        # Choose initialization strategy
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        f_init = init_func(n_steps)
        
        # Run aggressive gradient optimization with more iterations
        f_opt, c2_val = create_aggressive_gradient_optimization(f_init, max_iter=2500, patience=70)
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_solution = f_opt
    
    # Strategy 2: Differential evolution refinement if time permits and solution is promising
    if best_solution is not None and time.time() - start_time < 50 and best_c2 < 0.96:
        try:
            # Convert to numpy for scipy compatibility
            f_init = np.array(best_solution)
            
            def objective(f_vals):
                f_vals = np.maximum(f_vals, 0)
                # Use the exact evaluator for consistency
                c2 = compute_c2_exact(f_vals.tolist())
                return -c2  # Minimize negative to maximize C2
            
            # Bounds for each parameter (non-negative)
            bounds = [(0, 4.0) for _ in range(len(f_init))]
            
            # Run differential evolution with more iterations for better refinement
            result = differential_evolution(
                objective,
                bounds,
                maxiter=25,  # More iterations for better refinement
                popsize=15,   # Larger population size
                mutation=(0.8, 1.0),  # Good mutation range
                recombination=0.9,   # Good recombination
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
    
    # Strategy 3: Final aggressive optimization on best solution if time allows
    if best_solution is not None and time.time() - start_time < 55:
        try:
            f_init = np.array(best_solution)
            f_opt, c2_val = create_aggressive_gradient_optimization(f_init, max_iter=1000, patience=50)
            
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
