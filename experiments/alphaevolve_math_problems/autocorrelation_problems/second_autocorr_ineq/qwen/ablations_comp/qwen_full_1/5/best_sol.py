# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
import warnings
from typing import List
import jax
import jax.numpy as jnp
from jax import jit, grad
import optax

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autoconvolution_norms(f_values: List[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    Uses the exact piecewise linear integration method as specified in problem description.
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Ensure non-negative values
    f = np.maximum(f, 0)
    
    # Create step function on [-1/4, 1/4] with equal spacing
    n_steps = len(f)
    if n_steps == 0:
        return 0.0, 0.0, 0.0
    
    # Step width (interval [-1/4, 1/4] divided into n_steps)
    dx = 0.5 / n_steps
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = signal.convolve(f, f, mode='full')
    
    # Extract central portion of the convolution result
    mid = len(g) // 2
    g_centered = g[mid-(n_steps-1):mid+n_steps-1]
    
    # Compute ||g||₂² using the specified piecewise linear integration
    g2_norm_squared = 0.0
    if len(g_centered) > 1:
        for i in range(len(g_centered) - 1):
            h = dx
            y1 = g_centered[i]
            y2 = g_centered[i + 1]
            contribution = (h/3) * (y1**2 + y1*y2 + y2**2)
            g2_norm_squared += contribution
    else:
        g2_norm_squared = g_centered[0]**2 if len(g_centered) > 0 else 0.0
    
    # ||g||₁: L1-norm, approximated as sum(|g|) / (len(g) + 1)
    g_abs = np.abs(g_centered)
    g1_norm = np.sum(g_abs) / (len(g_centered) + 1) if len(g_centered) > 0 else 0.0
    
    # ||g||∞: Infinity-norm, computed as max(|g|)
    g_infty_norm = np.max(g_abs) if len(g_abs) > 0 else 0.0
    
    return g2_norm_squared, g1_norm, g_infty_norm

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

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given step function."""
    try:
        g2_norm_sq, g1_norm, g_infty_norm = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g1_norm <= 1e-15 or g_infty_norm <= 1e-15:
            return 0.0
            
        c2 = g2_norm_sq / (g1_norm * g_infty_norm)
        return c2
    except Exception as e:
        warnings.warn(f"Error computing C2: {e}")
        return 0.0

def create_optimized_pattern(n_steps: int) -> List[float]:
    """
    Create an optimized mathematical pattern based on the best practices from inspirations.
    This pattern is designed to achieve high C2 values.
    """
    # Create position array
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Create a highly optimized pattern that achieves superior C2 values
    # Based on the most successful approaches from inspirations
    
    # Main central component with strong peak - inspired by Program 3
    main_peak = 1.0 * np.exp(-20 * x**2)
    
    # First oscillatory component - creates constructive interference
    osc1 = 0.4 * np.sin(15 * np.pi * x) * np.exp(-10 * x**2)
    
    # Second oscillatory component - adds additional structure  
    osc2 = 0.3 * np.cos(20 * np.pi * x) * np.exp(-8 * x**2)
    
    # Third oscillatory component - fine structure for enhanced performance
    osc3 = 0.2 * np.sin(25 * np.pi * x) * np.exp(-6 * x**2)
    
    # Polynomial tail adjustment for smooth edges
    tail = 0.1 * (1 - 3 * x**2) * np.exp(-4 * x**2)
    
    # Combine all components
    values = main_peak + osc1 + osc2 + osc3 + tail
    
    # Ensure non-negativity and normalize appropriately
    values = np.clip(values, 0, None)
    if np.max(values) > 0:
        values = values / np.max(values) * 1.1  # Slightly lower normalization to maintain peak
    
    return values.tolist()

def create_multipeak_optimized(n_steps: int) -> List[float]:
    """
    Create an optimized multi-peak pattern that has shown excellent performance.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Optimized multi-peak construction with precise parameters
    f_vals = np.zeros(n_steps)
    
    # Central peak (most important) - sharper than previous versions
    f_vals += 1.3 * np.exp(-((x - 0.0)**2) / (2 * 0.06**2))
    
    # Left side peak - slightly smaller but positioned optimally
    f_vals += 0.7 * np.exp(-((x - (-0.12))**2) / (2 * 0.055**2))
    
    # Right side peak - balanced with left peak
    f_vals += 0.6 * np.exp(-((x - 0.12)**2) / (2 * 0.05**2))
    
    # Add additional structure to encourage good convolution properties
    # Smooth transitions to reduce sharp edges that might hurt the optimization
    for i in range(n_steps):
        if abs(x[i]) > 0.18:
            # Reduce values at the edges to create smoother boundaries
            f_vals[i] *= (1.0 - 0.3 * min(1.0, abs(x[i]) - 0.18) / 0.07)
    
    # Ensure non-negativity
    f_vals = np.clip(f_vals, 0, None)
    
    # Normalize to reasonable scale
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 0.9
    
    return f_vals.tolist()

def create_diverse_patterns(n_steps: int) -> List[List[float]]:
    """
    Create a set of diverse high-quality patterns for initialization.
    """
    patterns = []
    
    # High-quality optimized pattern (most important)
    patterns.append(create_optimized_pattern(n_steps))
    
    # Multi-peak pattern
    patterns.append(create_multipeak_optimized(n_steps))
    
    # Alternative mathematical pattern
    x = np.linspace(-0.25, 0.25, n_steps)
    alt_pattern = 0.8 * np.exp(-15 * x**2) + 0.3 * np.sin(12 * np.pi * x) * np.exp(-8 * x**2)
    alt_pattern = np.clip(alt_pattern, 0, None)
    if np.max(alt_pattern) > 0:
        alt_pattern = alt_pattern / np.max(alt_pattern) * 1.2
    patterns.append(alt_pattern.tolist())
    
    # Another mathematical pattern
    x = np.linspace(-0.25, 0.25, n_steps)
    alt_pattern2 = 0.9 * np.exp(-10 * x**2) + 0.2 * np.cos(18 * np.pi * x) * np.exp(-6 * x**2)
    alt_pattern2 = np.clip(alt_pattern2, 0, None)
    if np.max(alt_pattern2) > 0:
        alt_pattern2 = alt_pattern2 / np.max(alt_pattern2) * 1.1
    patterns.append(alt_pattern2.tolist())
    
    return patterns

def hybrid_optimization_approach() -> List[float]:
    """
    Hybrid approach combining gradient-based optimization with evolutionary refinement.
    This combines the best of both worlds from the inspirations.
    """
    # Parameters optimized for performance and quality
    n_steps = 2500  # Higher resolution for better optimization (from inspirations)
    learning_rate = 0.05  # Optimal from inspirations
    num_iterations = 2000  # More iterations for convergence (from inspirations)
    
    # Start with the best mathematical pattern
    initial_pattern = create_optimized_pattern(n_steps)
    
    # Convert to JAX array for gradient-based optimization
    f_initial = jnp.array(initial_pattern)
    
    # Create optimizer with optimal settings
    optimizer = optax.adam(learning_rate)
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
    
    # Run gradient-based optimization
    current_params = f_initial
    current_opt_state = opt_state
    
    for iteration in range(num_iterations):
        current_params, current_opt_state = update_step(current_params, current_opt_state)
    
    # Final refinement with local search
    final_result = current_params.tolist()
    final_c2 = compute_c2(final_result)
    
    # Local search refinement with more aggressive perturbations
    for iteration in range(500):
        # Make more aggressive perturbations to escape local optima
        neighbor = final_result.copy()
        # Perturb many elements with larger variance
        num_perturb = max(1, len(neighbor) // 20)
        indices_to_perturb = random.sample(range(len(neighbor)), num_perturb)
        for idx in indices_to_perturb:
            neighbor[idx] = max(0, neighbor[idx] + random.gauss(0, 0.03))
        
        neighbor_c2 = compute_c2(neighbor)
        if neighbor_c2 > final_c2:
            final_result = neighbor
            final_c2 = neighbor_c2
    
    return final_result

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses the most successful hybrid approach from inspirations.
    """
    try:
        # Use the hybrid optimization approach that achieved best results in inspirations
        result = hybrid_optimization_approach()
        return result
    except Exception as e:
        # Fallback to mathematical pattern if everything fails
        warnings.warn(f"Fallback due to error: {e}")
        return create_optimized_pattern(2500)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
