# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
import optax
import random
from typing import List
import time
from scipy import signal

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_autoconvolution_norms_correct(f_values: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation using correct piecewise linear integration.
    This matches exactly what the evaluator uses.
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Step width (interval [-1/4, 1/4] divided into n steps)
    n_steps = len(f)
    if n_steps == 0:
        return 0.0, 0.0, 0.0
    
    dx = 0.5 / n_steps
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = signal.convolve(f, f, mode='full')
    
    # Extract central portion that corresponds to the valid autoconvolution on [-1/4, 1/4]
    # For a function with n steps on [-1/4, 1/4], the convolution has 2*n-1 elements
    # The center n elements correspond to the interval [-1/4, 1/4] 
    center_idx = len(g) // 2
    g_centered = g[center_idx - (n_steps - 1):center_idx + n_steps - 1]
    
    # Compute ||g||₂² using correct piecewise linear integration:
    # For adjacent points with heights y1, y2 and width dx:
    # contribution = (dx/3)(y1² + y1*y2 + y2²)
    if len(g_centered) < 2:
        g2_norm_sq = 0.0
    else:
        g2_norm_sq = 0.0
        for i in range(len(g_centered) - 1):
            y1, y2 = g_centered[i], g_centered[i+1]
            g2_norm_sq += (dx/3) * (y1*y1 + y1*y2 + y2*y2)
    
    # ||g||₁: L1-norm, approximated as sum(|g|) / (len(g) + 1)
    g_abs = np.abs(g_centered)
    g1_norm = np.sum(g_abs) / (len(g_centered) + 1) if len(g_centered) > 0 else 0.0
    
    # ||g||∞: Infinity-norm, computed as max(|g|)
    g_infty_norm = np.max(g_abs) if len(g_abs) > 0 else 0.0
    
    return g2_norm_sq, g1_norm, g_infty_norm

@jit
def compute_C2_jax(f_values: jnp.ndarray) -> jnp.ndarray:
    """Compute C2 value for given step function values using JAX with corrected logic"""
    try:
        # Compute autoconvolution using numpy (more accurate for this specific problem)
        f = jnp.array(f_values)
        n_steps = len(f)
        if n_steps == 0:
            return jnp.array(0.0)
        
        # Compute dx correctly
        dx = 0.5 / n_steps
        
        # Compute autoconvolution
        g = jnp.convolve(f, f, mode='full')
        
        # Extract the central portion
        center_idx = len(g) // 2
        g_centered = g[center_idx - (n_steps - 1):center_idx + n_steps - 1]
        
        # Compute ||g||₂² using correct piecewise linear integration
        if len(g_centered) < 2:
            g2_norm_sq = 0.0
        else:
            # Vectorized computation of the piecewise integration
            g_vals = g_centered[:-1]
            g_next_vals = g_centered[1:]
            g2_norm_sq = jnp.sum((g_vals**2 + g_vals * g_next_vals + g_next_vals**2) * dx / 3.0)
        
        # L1 norm = sum(|g|) / (len(g) + 1)
        g1_norm = jnp.sum(jnp.abs(g_centered)) / (len(g_centered) + 1)
        
        # L-infinity norm = max(|g|)
        ginf_norm = jnp.max(jnp.abs(g_centered))
        
        # Avoid division by zero
        epsilon = 1e-12
        g1_norm = jnp.maximum(g1_norm, epsilon)
        ginf_norm = jnp.maximum(ginf_norm, epsilon)
        
        # Compute C2
        c2 = g2_norm_sq / (g1_norm * ginf_norm)
        return c2
    except Exception:
        return jnp.array(0.0)

def create_best_mathematical_pattern(n_steps: int) -> List[float]:
    """
    Create the best mathematical pattern based on successful approaches from inspirations.
    This pattern achieved high performance in prior runs and combines Gaussian components
    with oscillatory terms that create favorable convolution properties.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # This is a refined version of the pattern that achieved success in inspirations:
    # Strong central peak with oscillatory components that enhance convolution properties
    main_peak = 1.0 * np.exp(-25 * x**2)
    osc1 = 0.4 * np.sin(15 * np.pi * x) * np.exp(-10 * x**2)
    osc2 = 0.3 * np.cos(20 * np.pi * x) * np.exp(-8 * x**2)
    osc3 = 0.2 * np.sin(25 * np.pi * x) * np.exp(-6 * x**2)
    tail = 0.1 * (1 - 3 * x**2) * np.exp(-4 * x**2)
    
    values = main_peak + osc1 + osc2 + osc3 + tail
    
    # Ensure non-negativity and normalize appropriately
    values = np.clip(values, 0, None)
    if np.max(values) > 0:
        values = values / np.max(values) * 1.2
    
    return values.tolist()

def create_balanced_multimodal_pattern(n_steps: int) -> List[float]:
    """
    Create a balanced multimodal pattern with multiple peaks for robust convolution properties.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Create a pattern with multiple well-balanced peaks
    # Central peak
    central = 0.8 * np.exp(-x**2 * 10)
    
    # Left and right secondary peaks
    left_peak = 0.3 * np.exp(-((x + 0.15)**2) * 12)
    right_peak = 0.3 * np.exp(-((x - 0.15)**2) * 12)
    
    # Oscillatory component for better convolution mixing
    oscillation = 0.1 * np.sin(12 * np.pi * x) * np.exp(-x**2 * 3)
    
    # Fine structure component
    fine_structure = 0.05 * np.cos(20 * np.pi * x) * np.exp(-x**2 * 2)
    
    values = central + left_peak + right_peak + oscillation + fine_structure
    
    # Ensure non-negativity and normalize
    values = np.clip(values, 0, None)
    if np.max(values) > 0:
        values = values / np.max(values) * 1.2
    
    return values.tolist()

def create_symmetric_peak_pattern(n_steps: int) -> List[float]:
    """
    Create a symmetric peak pattern that promotes good autoconvolution properties.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Create a symmetric pattern with strong central concentration
    # This creates autoconvolutions that are relatively flat, which improves C2
    envelope = 0.7 * np.exp(-x**2 * 8)
    secondary1 = 0.2 * np.exp(-((x - 0.15)**2) * 12)
    secondary2 = 0.2 * np.exp(-((x + 0.15)**2) * 12)
    oscillation = 0.1 * np.sin(15 * np.pi * x) * np.exp(-x**2 * 4)
    
    values = envelope + secondary1 + secondary2 + oscillation
    
    # Ensure non-negativity and normalize
    values = np.clip(values, 0, None)
    if np.max(values) > 0:
        values = values / np.max(values) * 1.0
    
    return values.tolist()

def create_aggressive_optimization_approach() -> List[float]:
    """
    Implement the most aggressive optimization approach combining best practices from inspirations.
    """
    n_steps = 2500  # Use higher resolution as in successful inspirations
    max_time = 80  # Leave some time for final processing
    start_time = time.time()
    
    # Strategy: Try multiple high-quality initial patterns
    initial_patterns = [
        create_best_mathematical_pattern(n_steps),
        create_balanced_multimodal_pattern(n_steps),
        create_symmetric_peak_pattern(n_steps)
    ]
    
    # Add a few variations to increase diversity
    x = np.linspace(-0.25, 0.25, n_steps)
    # Variation with different oscillation frequencies
    variation1 = 0.8 * np.exp(-x**2 * 12) + 0.2 * np.sin(20 * np.pi * x) * np.exp(-x**2 * 3)
    variation1 = np.clip(variation1, 0, None)
    if np.max(variation1) > 0:
        variation1 = variation1 / np.max(variation1) * 1.0
    initial_patterns.append(variation1.tolist())
    
    # Variation with more pronounced peaks
    variation2 = 0.9 * np.exp(-((x - 0.0)**2) / (2 * 0.06**2)) + \
                0.5 * np.exp(-((x - 0.15)**2) / (2 * 0.05**2)) + \
                0.5 * np.exp(-((x + 0.15)**2) / (2 * 0.05**2))
    variation2 = np.clip(variation2, 0, None)
    if np.max(variation2) > 0:
        variation2 = variation2 / np.max(variation2) * 1.1
    initial_patterns.append(variation2.tolist())
    
    best_c2 = 0.0
    best_result = None
    
    # Test all initial patterns with aggressive optimization
    for i, initial_pattern in enumerate(initial_patterns):
        if time.time() - start_time > max_time:
            break
            
        try:
            # Initialize with JAX array
            f_initial = jnp.array(initial_pattern)
            
            # Use the same aggressive parameters from successful inspirations
            # Adam optimizer with learning rate 0.05 (as used in inspirations 1 & 2)
            optimizer = optax.adam(0.05)  # This learning rate worked well
            opt_state = optimizer.init(f_initial)
            
            @jit
            def loss_fn(params):
                """Compute negative C2 (since we want to maximize C2)"""
                c2_val = compute_C2_jax(params)
                return -c2_val
            
            @jit
            def update_step(params, opt_state):
                """Perform one optimization step"""
                grad_val = grad(loss_fn)(params)
                updates, opt_state = optimizer.update(grad_val, opt_state)
                params = optax.apply_updates(params, updates)
                # Ensure non-negativity
                params = jnp.maximum(params, 0.0)
                return params, opt_state
            
            # Run optimization with maximum iterations for best possible result
            current_params = f_initial
            current_opt_state = opt_state
            
            # Use more aggressive iteration count as in inspirations
            num_iterations = 2000  # Increased from original
            
            # Track improvement for early stopping
            last_c2 = -float('inf')
            patience_counter = 0
            patience_limit = 30  # Allow for more patience
            
            for iteration in range(num_iterations):
                if time.time() - start_time > max_time:
                    break
                    
                current_params, current_opt_state = update_step(current_params, current_opt_state)
                
                # Periodic evaluation for early stopping
                if iteration % 10 == 0:
                    current_c2 = compute_C2_jax(current_params).item()
                    if current_c2 - last_c2 < 1e-6:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                    last_c2 = current_c2
                    
                    if patience_counter >= patience_limit:
                        break
            
            # Final evaluation
            final_c2 = compute_C2_jax(current_params)
            print(f"Initialization {i+1}, Final C2 = {final_c2:.6f}")
            
            if final_c2 > best_c2:
                best_c2 = final_c2
                best_result = current_params.tolist()
                
        except Exception as e:
            print(f"Failed with initialization {i+1}: {e}")
            continue
    
    # Return the best result found or fallback to a proven pattern
    if best_result is not None:
        return best_result
    else:
        # Fallback to the most successful mathematical pattern
        return create_best_mathematical_pattern(n_steps)

def construct_function() -> List[float]:
    """
    Main function implementing the most effective optimization strategy.
    Uses the aggressive approach with multiple high-quality initial patterns
    and optimal JAX-based gradient optimization.
    """
    return create_aggressive_optimization_approach()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
