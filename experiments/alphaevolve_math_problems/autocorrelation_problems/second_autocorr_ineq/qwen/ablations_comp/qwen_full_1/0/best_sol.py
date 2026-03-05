# EVOLVE-BLOCK-START

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy import signal
from scipy.optimize import differential_evolution
import time
from scipy.optimize import minimize
import warnings
import random
from typing import List

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
    
    # ||g||₁ using trapezoidal rule for integral of |g| - simplified version
    if len(g_abs) >= 2:
        # Trapezoidal rule: (h/2) * (y0 + 2*y1 + ... + 2*y_{n-2} + y_{n-1})
        norm_1 = dx * (0.5 * np.sum(g_abs[:-1]) + 0.5 * np.sum(g_abs[1:]) + 0.5 * (g_abs[0] + g_abs[-1]))
    else:
        norm_1 = np.sum(g_abs) * dx
    
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

def create_optimized_peak_initialization(n: int) -> np.ndarray:
    """
    Create an optimized initialization strategy inspired by the most successful 
    approaches from inspirations - focused on creating high-quality solutions quickly
    """
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a sophisticated function with multiple peaks and oscillations
    f_init = np.zeros(n)
    
    # Strong central peak - optimized for autoconvolution sharpness
    f_init += 0.85 * np.exp(-x**2 / (2 * 0.02**2))
    
    # Secondary peaks for better convolution structure
    f_init += 0.15 * np.exp(-((x - 0.12)**2) / (2 * 0.04**2))
    f_init += 0.15 * np.exp(-((x + 0.12)**2) / (2 * 0.04**2))
    
    # Add oscillatory component to break symmetry and create better convolution
    f_init += 0.1 * np.sin(25 * np.pi * x) * np.exp(-x**2 / 0.03)
    
    # Normalize and ensure positivity
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.9
    
    # Add small amount of noise for better exploration
    noise = np.random.normal(0, 0.01, n)
    f_init = np.maximum(f_init + noise, 0)
    
    return f_init

def create_broad_plateau_initialization(n: int) -> np.ndarray:
    """Create a broad plateau initialization for better distribution"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Create a function with a broad central plateau
    f_init = np.zeros(n)
    
    # Broad central plateau
    central_mask = np.abs(x) <= 0.15
    f_init[central_mask] = 1.0
    
    # Smooth transitions at edges
    edge_left = (x > -0.25) & (x < -0.15)
    edge_right = (x > 0.15) & (x < 0.25)
    
    # Exponential decay for smooth transitions
    f_init[edge_left] = np.exp((x[edge_left] + 0.25) / 0.02)
    f_init[edge_right] = np.exp(-(x[edge_right] - 0.25) / 0.02)
    
    # Add oscillation for structure
    oscillation = 0.1 * np.sin(15 * np.pi * x)
    f_init = np.maximum(f_init + oscillation, 0)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.85
    
    return f_init

def create_multi_component_initialization(n: int) -> np.ndarray:
    """Create a multi-component initialization with strategic peaks"""
    x = np.linspace(-0.25, 0.25, n)
    
    # Multi-component function
    f_init = np.zeros(n)
    
    # Dominant central peak
    f_init += 0.9 * np.exp(-x**2 / (2 * 0.015**2))
    
    # Multiple side peaks
    f_init += 0.1 * np.exp(-((x - 0.1)**2) / (2 * 0.03**2))
    f_init += 0.1 * np.exp(-((x + 0.1)**2) / (2 * 0.03**2))
    f_init += 0.08 * np.exp(-((x - 0.18)**2) / (2 * 0.04**2))
    f_init += 0.08 * np.exp(-((x + 0.18)**2) / (2 * 0.04**2))
    
    # Add oscillation
    oscillation = 0.15 * np.sin(30 * np.pi * x) * np.exp(-x**2 / 0.05)
    f_init = np.maximum(f_init + oscillation, 0)
    
    # Normalize
    if np.max(f_init) > 0:
        f_init = f_init / np.max(f_init) * 0.8
    
    return f_init

def create_adaptive_initialization(n: int) -> np.ndarray:
    """Create an adaptive initialization using multiple strategies"""
    # Try different initialization strategies and pick the best
    strategies = [
        create_optimized_peak_initialization(n),
        create_broad_plateau_initialization(n),
        create_multi_component_initialization(n),
        np.random.uniform(0, 0.6, n)
    ]
    
    # Evaluate each candidate to select the best one using fast computation
    best_candidate = strategies[0]
    best_c2 = -float('inf')
    
    for candidate in strategies:
        try:
            c2 = compute_c2_jax(candidate)
            if c2 > best_c2:
                best_c2 = c2
                best_candidate = candidate
        except:
            continue
    
    return best_candidate

def advanced_gradient_optimization(f_init, max_iter=3000, patience=100):
    """Advanced gradient optimization with improved parameters"""
    f_opt = jnp.array(f_init)
    
    # Optimized Adam parameters for better convergence
    learning_rate = 0.3  # Higher learning rate for faster convergence
    beta1 = 0.99  # High momentum for stable updates
    beta2 = 0.999
    epsilon = 1e-8
    
    # Initialize Adam moments
    m = jnp.zeros_like(f_opt)
    v = jnp.zeros_like(f_opt)
    
    best_c2 = -float('inf')
    best_f = f_opt
    patience_counter = 0
    
    # Optimization loop with adaptive learning rate
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
        else:
            patience_counter += 1
            
        # Early stopping if no improvement
        if patience_counter >= patience:
            break
            
        # Adaptive learning rate decay
        if i > 500 and patience_counter > 50:
            learning_rate *= 0.95
            
        # Aggressive decay for final phase
        if i > 1500 and learning_rate > 0.05:
            learning_rate *= 0.97
    
    return best_f, best_c2

def enhanced_hybrid_optimization():
    """Enhanced hybrid optimization with multiple strategies"""
    # Use high resolution for better optimization potential
    n_steps = 1200  # Increased resolution
    
    # Try multiple initialization strategies
    init_strategies = [
        ("optimized_peaks", create_optimized_peak_initialization),
        ("broad_plateau", create_broad_plateau_initialization),
        ("multi_component", create_multi_component_initialization),
        ("adaptive", create_adaptive_initialization)
    ]
    
    best_solution = None
    best_c2 = -float('inf')
    
    # Try different initialization strategies
    for strategy_name, init_func in init_strategies:
        try:
            # Create initialization
            f_init = init_func(n_steps)
            
            # Run advanced gradient optimization
            f_opt, c2_val = advanced_gradient_optimization(f_init, max_iter=2000, patience=100)
            
            if c2_val > best_c2:
                best_c2 = c2_val
                best_solution = f_opt
        except Exception as e:
            continue
    
    # Final refinement with intensive optimization
    if best_solution is not None:
        try:
            # Run very intensive optimization
            final_f_opt, final_c2_val = advanced_gradient_optimization(
                np.array(best_solution), max_iter=2500, patience=150
            )
            if final_c2_val > best_c2:
                best_c2 = final_c2_val
                best_solution = final_f_opt
        except Exception:
            pass
    
    # If no good solution found, fallback to best initialization
    if best_solution is None:
        f_init = create_optimized_peak_initialization(n_steps)
        best_solution, best_c2 = advanced_gradient_optimization(f_init, max_iter=2000)
    
    # Final evaluation using exact computation method
    final_c2 = compute_c2(list(best_solution))
    
    return list(best_solution)

def construct_function() -> list[float]:
    """
    Enhanced hybrid approach that maximizes C2 by combining:
    1. Multiple sophisticated initialization strategies
    2. Advanced gradient optimization with aggressive parameters
    3. Intelligent fallback mechanisms
    """
    return enhanced_hybrid_optimization()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
