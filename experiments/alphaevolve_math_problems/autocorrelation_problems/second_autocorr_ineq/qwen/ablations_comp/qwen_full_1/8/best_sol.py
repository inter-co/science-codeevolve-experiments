# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List
import time
import warnings
import jax
import jax.numpy as jnp
from jax import jit, grad
import optax
warnings.filterwarnings('ignore')

# Set seeds for reproducibility like inspiration programs
np.random.seed(42)
random.seed(42)

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation using correct piecewise linear integration.
    This implements the exact formula from the prompt:
    For adjacent points with heights y1, y2 and width h, contribution is (h/3)(y1² + y1*y2 + y2²)
    """
    # Convert to numpy array for easier handling
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using scipy's convolve
    g = signal.convolve(f, f, mode='full')
    
    # Get the actual convolution result (length 2*n-1)
    n = len(f)
    
    # Clip negative values to zero (as specified in problem)
    g_positive = np.maximum(g, 0)
    
    # Compute ||g||₂² using correct piecewise linear integration as specified
    # This is the key improvement - use the exact formula from prompt
    if len(g_positive) < 2:
        norm_g2_squared = 0.0
    else:
        # Width between points in the convolution result
        # Since original f was on [-1/4, 1/4] with n points, 
        # the step size is 0.5/(n-1) for original function
        dx = 0.5 / (n - 1) if n > 1 else 1.0
        
        # Use the trapezoidal-like integration formula for piecewise linear segments
        # Formula from prompt: (h/3)(y1² + y1*y2 + y2²)
        norm_g2_squared = 0.0
        for i in range(len(g_positive) - 1):
            y1, y2 = g_positive[i], g_positive[i+1]
            norm_g2_squared += (dx/3) * (y1*y1 + y1*y2 + y2*y2)
    
    # Compute ||g||₁ using the evaluator's approach: sum(|g|) / (len(g) + 1)
    norm_g1 = np.sum(g_positive) / (len(g_positive) + 1)
    
    # Compute ||g||∞
    norm_ginf = np.max(g_positive)
    
    return norm_g2_squared, norm_g1, norm_ginf

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given step function values"""
    try:
        norm_g2_squared, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_g1 <= 1e-15 or norm_ginf <= 1e-15:
            return 0.0
            
        c2 = norm_g2_squared / (norm_g1 * norm_ginf)
        return c2
    except Exception:
        return 0.0

@jit
def compute_c2_jax(f_values: jnp.ndarray) -> jnp.ndarray:
    """JAX version for faster computation"""
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

def create_advanced_multimodal_pattern(n_steps: int) -> List[float]:
    """
    Create an advanced multimodal pattern inspired by successful mathematical approaches.
    Based on the best-performing patterns from inspirations.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Enhanced pattern with carefully tuned parameters for optimal C2
    f_vals = (
        0.45 * np.exp(-((x - 0.18)**2) * 40) +   # Strong peak near right edge
        0.45 * np.exp(-((x + 0.18)**2) * 40) +   # Strong peak near left edge  
        0.1 * np.exp(-x**2 * 18) +               # Central peak
        0.05 * np.sin(20 * np.pi * x) * np.exp(-x**2 * 8)  # Oscillation with envelope
    )
    
    # Ensure positivity and normalize properly
    f_vals = np.maximum(f_vals, 0)
    total = np.sum(f_vals)
    if total > 0:
        f_vals = f_vals / total * 140  # Slightly higher scaling factor
    
    return f_vals.tolist()

def create_improved_multipeak_pattern(n_steps: int) -> List[float]:
    """
    Create an improved multi-peak pattern with better tuning and mathematical properties.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Multi-peak construction with parameters tuned for optimal convolution properties
    f_vals = np.zeros(n_steps)
    
    # Central peak - higher amplitude to maximize autoconvolution energy
    central_amp = 2.6
    central_width = 0.025
    f_vals += central_amp * np.exp(-((x - 0.0)**2) / (2 * central_width**2))
    
    # Left peak - balanced strength and positioning
    left_amp = 1.6
    left_width = 0.035
    left_pos = -0.18
    f_vals += left_amp * np.exp(-((x - left_pos)**2) / (2 * left_width**2))
    
    # Right peak - mirror of left with slight variation
    right_amp = 1.5
    right_width = 0.035
    right_pos = 0.18
    f_vals += right_amp * np.exp(-((x - right_pos)**2) / (2 * right_width**2))
    
    # Add oscillatory component to enhance convolution mixing
    oscillation_amp = 0.45
    oscillation_freq = 22
    f_vals += oscillation_amp * np.sin(oscillation_freq * np.pi * x) * np.exp(-x**2 * 6)
    
    # Add smoothing at edges to reduce boundary artifacts
    edge_smooth = 0.04
    for i in range(n_steps):
        if abs(x[i]) > (0.25 - edge_smooth):
            dist_from_edge = abs(abs(x[i]) - 0.25)
            reduction_factor = max(0, 1.0 - dist_from_edge / edge_smooth)
            f_vals[i] *= reduction_factor
    
    # Ensure non-negativity
    f_vals = np.clip(f_vals, 0, None)
    
    # Normalize to appropriate scale
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 1.8
    
    return f_vals.tolist()

def create_sinc_pattern(n_steps: int) -> List[float]:
    """
    Create a sinc-like pattern that can produce favorable convolution properties.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Sinc-like function with controlled decay
    sinc_vals = np.sinc(2 * x) * np.exp(-x**2 * 5)
    sinc_vals = np.clip(sinc_vals, 0, None)
    
    if np.max(sinc_vals) > 0:
        sinc_vals = sinc_vals / np.max(sinc_vals) * 1.2
    
    return sinc_vals.tolist()

def create_highly_optimized_pattern(n_steps: int) -> List[float]:
    """
    Create a highly optimized pattern combining best elements from inspirations.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # This pattern is specifically crafted to maximize the ratio of ||g||₂² to (||g||₁ · ||g||∞)
    # by creating peaks that generate strong autoconvolution without excessive peakiness
    f_vals = (
        0.5 * np.exp(-((x - 0.15)**2) * 45) +   # Strong right peak
        0.5 * np.exp(-((x + 0.15)**2) * 45) +   # Strong left peak
        0.15 * np.exp(-x**2 * 20) +             # Central peak
        0.1 * np.sin(25 * np.pi * x) * np.exp(-x**2 * 8)  # High frequency oscillation
    )
    
    # Ensure positivity and normalize
    f_vals = np.clip(f_vals, 0, None)
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 1.3
    
    return f_vals.tolist()

def create_balanced_plateau_pattern(n_steps: int) -> List[float]:
    """
    Create a balanced plateau pattern designed to produce favorable autoconvolution.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Create a function with a central plateau and smooth transitions
    f_vals = np.ones_like(x)
    
    # Define plateau region
    plateau_width = 0.12
    plateau_start = -plateau_width/2
    plateau_end = plateau_width/2
    
    # Apply smooth transitions
    for i in range(n_steps):
        xi = x[i]
        if xi < plateau_start:
            # Transition from plateau to zero on left
            dist = abs(xi - plateau_start)
            f_vals[i] = max(0, 1.0 - dist * 8)
        elif xi > plateau_end:
            # Transition from plateau to zero on right
            dist = abs(xi - plateau_end)
            f_vals[i] = max(0, 1.0 - dist * 8)
        # Else remains 1.0 (plateau)
    
    # Add slight oscillation to break symmetry
    f_vals = f_vals * (1 + 0.1 * np.sin(15 * np.pi * x))
    
    # Ensure non-negativity
    f_vals = np.clip(f_vals, 0, None)
    
    # Normalize
    if np.sum(f_vals) > 0:
        f_vals = f_vals / np.sum(f_vals) * 120
    
    return f_vals.tolist()

def hybrid_optimization_approach() -> List[float]:
    """
    Enhanced hybrid approach that combines multiple strategies effectively:
    1. Multiple diverse initial patterns
    2. JAX-accelerated gradient optimization
    3. Multiple restarts with intelligent selection
    4. Adaptive optimization parameters
    """
    n_steps = 2000  # High resolution for better optimization potential
    max_time = 85  # Leave 5 seconds for final processing
    start_time = time.time()
    
    # Strategy 1: Try multiple diverse initial patterns (from all inspirations)
    initial_patterns = []
    
    # Pattern 1: Highly optimized pattern (from inspiration 3)
    initial_patterns.append(create_highly_optimized_pattern(n_steps))
    
    # Pattern 2: Improved multi-peak pattern (from inspiration 3)
    initial_patterns.append(create_improved_multipeak_pattern(n_steps))
    
    # Pattern 3: Sinc pattern (from inspiration 2)
    initial_patterns.append(create_sinc_pattern(n_steps))
    
    # Pattern 4: Balanced plateau pattern (from inspiration 1)
    initial_patterns.append(create_balanced_plateau_pattern(n_steps))
    
    # Pattern 5: Advanced multimodal pattern (from inspiration 1)
    initial_patterns.append(create_advanced_multimodal_pattern(n_steps))
    
    # Pattern 6: Simple Gaussian baseline
    x = np.linspace(-0.25, 0.25, n_steps)
    gaussian_baseline = np.exp(-x**2 * 12)
    gaussian_baseline = np.clip(gaussian_baseline, 0, None)
    if np.max(gaussian_baseline) > 0:
        gaussian_baseline = gaussian_baseline / np.max(gaussian_baseline) * 1.1
    initial_patterns.append(gaussian_baseline.tolist())
    
    best_c2 = 0.0
    best_result = None
    
    # Test all initializations and run optimization on the best one
    for i, initial_pattern in enumerate(initial_patterns):
        if time.time() - start_time > max_time:
            break
            
        try:
            # Initialize with JAX array
            f_initial = jnp.array(initial_pattern)
            
            # Create optimizer with adaptive learning rates
            optimizer = optax.adam(0.05)  # Slightly higher learning rate for faster convergence
            opt_state = optimizer.init(f_initial)
            
            @jit
            def loss_fn(params):
                """Compute negative C2 (since we want to maximize C2)"""
                c2_val = compute_c2_jax(params)
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
            
            # Run optimization with enhanced early stopping
            current_params = f_initial
            current_opt_state = opt_state
            
            # Early stopping based on recent improvements
            last_c2 = -1e10
            patience_counter = 0
            max_patience = 250  # More patience for better convergence
            best_local_c2 = -1e10
            best_local_params = current_params
            
            # Run more iterations for better convergence with smart stopping
            for iteration in range(1800):  # More iterations for better convergence
                if time.time() - start_time > max_time:
                    break
                    
                current_params, current_opt_state = update_step(current_params, current_opt_state)
                
                # Check for improvement
                final_c2 = compute_c2_jax(current_params)
                if final_c2 > last_c2:
                    last_c2 = final_c2
                    patience_counter = 0
                    if final_c2 > best_local_c2:
                        best_local_c2 = final_c2
                        best_local_params = current_params
                else:
                    patience_counter += 1
                
                # Early stopping if no improvement for too long
                if patience_counter > max_patience:
                    break
            
            # Local refinement around the best found solution
            final_c2 = compute_c2_jax(best_local_params)
            print(f"Initialization {i+1}, Final C2 = {final_c2:.6f}")
            
            if final_c2 > best_c2:
                best_c2 = final_c2
                best_result = best_local_params.tolist()
                
        except Exception as e:
            print(f"Failed with initialization {i+1}: {e}")
            continue
    
    # If we didn't find anything, return the best pattern as fallback
    if best_result is not None:
        return best_result
    else:
        # Return the most promising pattern from our initial attempts
        return create_highly_optimized_pattern(n_steps)

def construct_function() -> List[float]:
    """Main function that uses enhanced hybrid optimization approach"""
    # Use the enhanced hybrid approach
    return hybrid_optimization_approach()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
