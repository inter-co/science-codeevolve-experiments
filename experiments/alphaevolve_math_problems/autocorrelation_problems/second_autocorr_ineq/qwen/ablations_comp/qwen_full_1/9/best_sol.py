# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List, Tuple
import time
import warnings
import math
from scipy.optimize import differential_evolution
import jax
import jax.numpy as jnp
from jax import jit, grad
import optax

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
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
    
    # Combine multiple components that have proven effective in optimization
    # This pattern has been optimized for high C2 values
    f_vals = (
        0.48 * np.exp(-((x - 0.18)**2) * 40) +   # Strong peak near right edge
        0.48 * np.exp(-((x + 0.18)**2) * 40) +   # Strong peak near left edge  
        0.08 * np.exp(-x**2 * 18) +              # Central peak
        0.08 * np.sin(20 * np.pi * x) * np.exp(-x**2 * 8)  # Oscillation with envelope
    )
    
    # Ensure positivity and normalize properly
    f_vals = np.maximum(f_vals, 0)
    total = np.sum(f_vals)
    if total > 0:
        f_vals = f_vals / total * 130  # Higher scaling factor for better optimization
    
    return f_vals.tolist()

def create_improved_multipeak_pattern(n_steps: int) -> List[float]:
    """
    Create an improved multi-peak pattern with better tuning and mathematical properties.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Multi-peak construction with parameters tuned for optimal convolution properties
    f_vals = np.zeros(n_steps)
    
    # Central peak - higher amplitude to maximize autoconvolution energy
    central_amp = 2.5
    central_width = 0.03
    f_vals += central_amp * np.exp(-((x - 0.0)**2) / (2 * central_width**2))
    
    # Left peak - balanced strength and positioning
    left_amp = 1.5
    left_width = 0.04
    left_pos = -0.18
    f_vals += left_amp * np.exp(-((x - left_pos)**2) / (2 * left_width**2))
    
    # Right peak - mirror of left with slight variation
    right_amp = 1.4
    right_width = 0.04
    right_pos = 0.18
    f_vals += right_amp * np.exp(-((x - right_pos)**2) / (2 * right_width**2))
    
    # Add oscillatory component to enhance convolution mixing
    oscillation_amp = 0.4
    oscillation_freq = 20
    f_vals += oscillation_amp * np.sin(oscillation_freq * np.pi * x) * np.exp(-x**2 * 5)
    
    # Add smoothing at edges to reduce boundary artifacts
    edge_smooth = 0.05
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

def create_hybrid_optimization_approach() -> List[float]:
    """
    Multi-strategy optimization that tries different approaches and combines their best aspects.
    Uses more aggressive optimization techniques and better parameter selection.
    """
    n_steps = 2000  # High resolution for better optimization potential
    max_time = 70  # Leave 5 seconds for final processing
    start_time = time.time()
    
    # Strategy 1: Try multiple diverse initial patterns
    initial_patterns = []
    
    # Pattern 1: Advanced multimodal pattern (from inspiration 2)
    initial_patterns.append(create_advanced_multimodal_pattern(n_steps))
    
    # Pattern 2: Improved multi-peak pattern
    initial_patterns.append(create_improved_multipeak_pattern(n_steps))
    
    # Pattern 3: Simple Gaussian baseline
    x = np.linspace(-0.25, 0.25, n_steps)
    gaussian_baseline = np.exp(-x**2 * 10)
    gaussian_baseline = np.clip(gaussian_baseline, 0, None)
    if np.max(gaussian_baseline) > 0:
        gaussian_baseline = gaussian_baseline / np.max(gaussian_baseline) * 1.2
    initial_patterns.append(gaussian_baseline.tolist())
    
    # Pattern 4: Symmetric oscillatory pattern
    x = np.linspace(-0.25, 0.25, n_steps)
    oscillatory_pattern = (
        0.5 * np.exp(-((x - 0.15)**2) * 20) +
        0.5 * np.exp(-((x + 0.15)**2) * 20) +
        0.4 * np.exp(-x**2 * 10) +
        0.2 * np.sin(16 * np.pi * x) * np.exp(-x**2 * 6)
    )
    oscillatory_pattern = np.clip(oscillatory_pattern, 0, None)
    if np.max(oscillatory_pattern) > 0:
        oscillatory_pattern = oscillatory_pattern / np.max(oscillatory_pattern) * 1.5
    initial_patterns.append(oscillatory_pattern.tolist())
    
    best_c2 = 0.0
    best_result = None
    
    # Test all initializations with gradient-based refinement
    for i, initial_pattern in enumerate(initial_patterns):
        if time.time() - start_time > max_time:
            break
            
        try:
            # Initialize with JAX array
            f_initial = jnp.array(initial_pattern)
            
            # Create optimizer with adaptive learning rates
            optimizer = optax.adam(0.04)  # Slightly higher learning rate for faster convergence
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
            max_patience = 300  # More patience for better convergence
            best_local_c2 = -1e10
            best_local_params = current_params
            
            # Run more iterations for better convergence
            for iteration in range(2000):  # More iterations for better convergence
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
    
    # If we didn't find anything, try differential evolution as a last resort
    if best_result is None:
        try:
            print("Trying differential evolution optimization...")
            # Create a more targeted differential evolution approach
            def objective(params):
                # Extract parameters for the function construction
                x = np.linspace(-0.25, 0.25, n_steps)
                
                # Construct function with parameters - simplified version
                f_values = (
                    params[0] * np.exp(-((x - params[1])**2) * params[2]) +
                    params[3] * np.exp(-((x + params[1])**2) * params[2]) +
                    params[4] * np.exp(-x**2 * params[5]) +
                    params[6] * np.sin(params[7] * np.pi * x) * np.exp(-x**2 * params[8])
                )
                
                # Ensure positivity
                f_values = np.maximum(f_values, 0)
                
                # Normalize
                total = np.sum(f_values)
                if total > 0:
                    f_values = f_values / total * 130
                
                # Return negative C2 for minimization
                return -compute_c2(f_values.tolist())
            
            # Better bounds for optimization
            bounds = [
                (0.1, 1.2),  # amp1
                (-0.2, 0.2), # loc1  
                (15, 50),     # width1
                (0.1, 1.2),  # amp2
                (-0.2, 0.2), # loc2
                (15, 50),     # width2
                (0.01, 0.3), # amp3
                (10, 25),      # freq
                (3, 10)       # width3
            ]
            
            # Run optimization with more iterations and better settings
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=150, 
                popsize=30,
                seed=42,
                disp=False,
                tol=1e-7
            )
            
            if result.success:
                params = result.x
                x = np.linspace(-0.25, 0.25, n_steps)
                f_values = (
                    params[0] * np.exp(-((x - params[1])**2) * params[2]) +
                    params[3] * np.exp(-((x + params[1])**2) * params[2]) +
                    params[4] * np.exp(-x**2 * params[5]) +
                    params[6] * np.sin(params[7] * np.pi * x) * np.exp(-x**2 * params[8])
                )
                
                f_values = np.maximum(f_values, 0)
                total = np.sum(f_values)
                if total > 0:
                    f_values = f_values / total * 130
                    
                de_result = f_values.tolist()
                de_c2 = compute_c2(de_result)
                if de_c2 > best_c2:
                    best_c2 = de_c2
                    best_result = de_result
        except:
            pass
    
    # If we still didn't find anything, return a good baseline pattern
    if best_result is not None:
        # Do final extensive local refinement on the best result
        try:
            refined_result = best_result.copy()
            current_c2 = compute_c2(refined_result)
            
            # Aggressive local search with many iterations
            for _ in range(1500):  # Even more iterations for thorough refinement
                if time.time() - start_time > max_time:
                    break
                    
                candidate = refined_result.copy()
                # Make small random changes with more adaptive magnitude
                num_changes = max(1, len(candidate) // 15)  # About 7% of elements for more exploration
                indices = np.random.choice(len(candidate), num_changes, replace=False)
                
                for idx in indices:
                    # Small random change with adaptive magnitude - more aggressive
                    change = random.uniform(-0.05, 0.05)
                    candidate[idx] = max(0, candidate[idx] + change)
                
                # Normalize
                total = sum(candidate)
                if total > 0:
                    candidate = [x / total * 110 for x in candidate]
                
                test_c2 = compute_c2(candidate)
                if test_c2 > current_c2:
                    current_c2 = test_c2
                    refined_result = candidate.copy()
                    
            return refined_result
        except:
            pass
        return best_result
    else:
        # Fallback to the best pattern we created
        return create_improved_multipeak_pattern(n_steps)

def construct_function() -> List[float]:
    """Main function that uses multi-strategy optimization approach"""
    # Use the improved multi-strategy approach
    return create_hybrid_optimization_approach()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
