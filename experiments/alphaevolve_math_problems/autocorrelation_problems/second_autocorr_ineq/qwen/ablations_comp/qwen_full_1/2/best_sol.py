# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List, Tuple
import time
import warnings
import jax
import jax.numpy as jnp
from jax import jit, grad
import optax
from scipy.optimize import differential_evolution, minimize
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
    
    # Get the actual convolution result (length 2*n-1)
    n = len(f)
    
    # Compute autoconvolution g = f * f using scipy's convolve
    g = signal.convolve(f, f, mode='full')
    
    # Clip negative values to zero (as specified in problem)
    g_positive = np.maximum(g, 0)
    
    # Compute ||g||₂² using correct piecewise linear integration as specified
    # This is the key improvement - use the exact formula from prompt
    if len(g_positive) < 2:
        norm_g2_squared = 0.0
    else:
        # Width between points in the original function 
        # Original domain [-1/4, 1/4] with n points means step size = 0.5/(n-1) for n>1
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
        
        # Compute dx correctly - step size in original domain [-1/4, 1/4] with n steps
        dx = 0.5 / (n_steps - 1) if n_steps > 1 else 1.0
        
        # Compute autoconvolution using full convolution
        g = jnp.convolve(f, f, mode='full')
        
        # Extract the central portion that corresponds to [-1/4, 1/4] 
        # This ensures we get the right convolution for our domain
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
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Combine multiple components that have proven effective in optimization
    # Based on the successful patterns from inspirations
    f_vals = (
        0.45 * np.exp(-((x - 0.2)**2) * 30) +   # Strong peak near right edge
        0.45 * np.exp(-((x + 0.2)**2) * 30) +   # Strong peak near left edge  
        0.1 * np.exp(-x**2 * 12) +              # Central peak
        0.05 * np.sin(15 * np.pi * x) * np.exp(-x**2 * 6)  # Oscillation with envelope
    )
    
    # Ensure positivity and normalize properly
    f_vals = np.maximum(f_vals, 0)
    total = np.sum(f_vals)
    if total > 0:
        f_vals = f_vals / total * 150  # Higher scaling factor for better optimization
    
    return f_vals.tolist()

def create_improved_multipeak_pattern(n_steps: int) -> List[float]:
    """
    Create an improved multi-peak pattern with better tuning and mathematical properties.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Multi-peak construction with parameters tuned for optimal convolution properties
    f_vals = np.zeros(n_steps)
    
    # Central peak - higher amplitude to maximize autoconvolution energy
    central_amp = 2.2
    central_width = 0.035
    f_vals += central_amp * np.exp(-((x - 0.0)**2) / (2 * central_width**2))
    
    # Left peak - balanced strength and positioning
    left_amp = 1.3
    left_width = 0.045
    left_pos = -0.15
    f_vals += left_amp * np.exp(-((x - left_pos)**2) / (2 * left_width**2))
    
    # Right peak - mirror of left with slight variation
    right_amp = 1.2
    right_width = 0.045
    right_pos = 0.15
    f_vals += right_amp * np.exp(-((x - right_pos)**2) / (2 * right_width**2))
    
    # Add oscillatory component to enhance convolution mixing
    oscillation_amp = 0.35
    oscillation_freq = 18
    f_vals += oscillation_amp * np.sin(oscillation_freq * np.pi * x) * np.exp(-x**2 * 6)
    
    # Add smoothing at edges to reduce boundary artifacts
    edge_smooth = 0.06
    for i in range(n_steps):
        if abs(x[i]) > (0.25 - edge_smooth):
            dist_from_edge = abs(abs(x[i]) - 0.25)
            reduction_factor = max(0, 1.0 - dist_from_edge / edge_smooth)
            f_vals[i] *= reduction_factor
    
    # Ensure non-negativity
    f_vals = np.clip(f_vals, 0, None)
    
    # Normalize to appropriate scale
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 1.6
    
    return f_vals.tolist()

def create_optimized_fourier_pattern(n_steps: int) -> List[float]:
    """
    Create a pattern optimized for Fourier-domain properties that lead to good convolution.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Combine multiple frequencies that tend to produce favorable autoconvolution shapes
    f_vals = (
        0.6 * np.exp(-x**2 * 15) +              # Central smooth peak
        0.3 * np.exp(-((x - 0.12)**2) * 12) +   # Right peak
        0.3 * np.exp(-((x + 0.12)**2) * 12) +   # Left peak
        0.15 * np.sin(12 * np.pi * x) * np.exp(-x**2 * 5)  # Oscillatory component
    )
    
    # Ensure non-negativity and normalize
    f_vals = np.clip(f_vals, 0, None)
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 1.1
    
    return f_vals.tolist()

def create_balanced_plateau_pattern(n_steps: int) -> List[float]:
    """
    Create a balanced plateau pattern designed to produce flatter autoconvolution.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Create a function with a central plateau and smooth transitions
    f_vals = np.ones_like(x)
    
    # Define plateau region
    plateau_width = 0.15
    plateau_start = -plateau_width/2
    plateau_end = plateau_width/2
    
    # Apply smooth transitions
    for i in range(n_steps):
        xi = x[i]
        if xi < plateau_start:
            # Transition from plateau to zero on left
            dist = abs(xi - plateau_start)
            f_vals[i] = max(0, 1.0 - dist * 5)
        elif xi > plateau_end:
            # Transition from plateau to zero on right
            dist = abs(xi - plateau_end)
            f_vals[i] = max(0, 1.0 - dist * 5)
        # Else remains 1.0 (plateau)
    
    # Add slight oscillation to break symmetry
    f_vals = f_vals * (1 + 0.1 * np.sin(10 * np.pi * x))
    
    # Ensure non-negativity
    f_vals = np.clip(f_vals, 0, None)
    
    # Normalize
    if np.sum(f_vals) > 0:
        f_vals = f_vals / np.sum(f_vals) * 150
    
    return f_vals.tolist()

def create_adaptive_multimodal_pattern(n_steps: int) -> List[float]:
    """
    Create an adaptive multimodal pattern that adjusts based on mathematical insights.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Create a pattern with multiple components that have been shown to work well
    # Based on mathematical optimization and experimental evidence
    f_vals = (
        0.5 * np.exp(-((x - 0.15)**2) * 20) +  # Right peak
        0.5 * np.exp(-((x + 0.15)**2) * 20) +  # Left peak
        0.4 * np.exp(-x**2 * 10) +             # Center peak
        0.2 * np.sin(16 * np.pi * x) * np.exp(-x**2 * 6)  # Oscillation
    )
    
    # Add some additional structure to encourage good convolution behavior
    # This helps create a more favorable autoconvolution profile
    additional_structure = 0.1 * np.exp(-((x - 0.0)**2) * 8) * np.exp(-((x - 0.1)**2) * 8)
    f_vals += additional_structure
    
    # Ensure non-negativity
    f_vals = np.clip(f_vals, 0, None)
    
    # Normalize appropriately
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 1.3
    
    return f_vals.tolist()

def create_mathematical_pattern(n_steps: int) -> List[float]:
    """
    Create a mathematically-informed pattern based on theoretical insights.
    """
    # Use a combination of Gaussian and oscillatory components
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Main component: Gaussian-like shape with controlled width
    main_component = np.exp(-x**2 / (2 * 0.1**2))
    
    # Add oscillation to encourage good convolution properties
    oscillation = 0.3 * np.sin(8 * np.pi * x) * np.exp(-x**2 / (2 * 0.15**2))
    
    # Combine and ensure non-negativity
    pattern = main_component + oscillation
    pattern = np.maximum(pattern, 0)
    
    # Normalize appropriately
    if np.sum(pattern) > 0:
        pattern = pattern / np.sum(pattern) * 10
    
    return pattern.tolist()

def create_simple_symmetric_pattern(n_steps: int) -> List[float]:
    """
    Create a simple symmetric pattern that works well for convolution.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Create a symmetric pattern with a central peak
    amplitude = 1.0 - 2.0 * abs(x)
    pattern = np.maximum(amplitude, 0)
    
    # Add some oscillation to encourage good convolution properties
    oscillation = 0.1 * np.sin(12 * np.pi * x) * np.exp(-x**2 / (2 * 0.1**2))
    pattern = pattern + oscillation
    
    # Ensure non-negativity
    pattern = np.maximum(pattern, 0)
    
    # Normalize
    if np.sum(pattern) > 0:
        pattern = pattern / np.sum(pattern) * 10
    
    return pattern.tolist()

def create_differential_evolution_optimized_function() -> List[float]:
    """
    Use differential evolution to optimize parameters for a mathematical function.
    """
    n_steps = 1000
    
    def objective(params):
        # Extract parameters
        amp1, loc1, width1, amp2, loc2, width2, amp3, freq, width3 = params
        
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Construct function with parameters
        f_values = (
            amp1 * np.exp(-((x - loc1)**2) * width1) +
            amp2 * np.exp(-((x + loc1)**2) * width1) +
            amp3 * np.exp(-x**2 * width2) +
            amp3 * np.sin(freq * np.pi * x) * np.exp(-x**2 * width3)
        )
        
        # Ensure positivity
        f_values = np.maximum(f_values, 0)
        
        # Normalize
        total = np.sum(f_values)
        if total > 0:
            f_values = f_values / total * 120
        
        # Return negative C2 for minimization
        return -compute_c2(f_values.tolist())
    
    # Better bounds for optimization
    bounds = [
        (0.1, 1.0),  # amp1
        (-0.2, 0.2), # loc1  
        (10, 40),     # width1
        (0.1, 1.0),  # amp2
        (-0.2, 0.2), # loc2
        (10, 40),     # width2
        (0.01, 0.2), # amp3
        (8, 20),      # freq
        (2, 8)       # width3
    ]
    
    # Run optimization with more iterations and better settings
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=20,
            seed=42,
            disp=False,
            tol=1e-6
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
                f_values = f_values / total * 120
                
            return f_values.tolist()
    except:
        pass
    
    # Fallback to a good pattern if optimization fails
    return create_advanced_multimodal_pattern(n_steps)

def create_multi_strategy_optimization() -> List[float]:
    """
    Multi-strategy optimization that tries different approaches and combines their best aspects.
    """
    n_steps = 1000  # Reduced resolution for faster testing while maintaining quality
    max_time = 75  # Leave 5 seconds for final processing
    start_time = time.time()
    
    # Strategy 1: Try multiple diverse initial patterns
    initial_patterns = []
    
    # Pattern 1: Mathematical pattern (from inspiration 2)
    initial_patterns.append(create_mathematical_pattern(n_steps))
    
    # Pattern 2: Simple symmetric pattern
    initial_patterns.append(create_simple_symmetric_pattern(n_steps))
    
    # Pattern 3: Advanced multimodal pattern (from inspiration 1)
    initial_patterns.append(create_advanced_multimodal_pattern(n_steps))
    
    # Pattern 4: Improved multi-peak pattern
    initial_patterns.append(create_improved_multipeak_pattern(n_steps))
    
    # Pattern 5: Optimized Fourier pattern
    initial_patterns.append(create_optimized_fourier_pattern(n_steps))
    
    # Pattern 6: Balanced plateau pattern
    initial_patterns.append(create_balanced_plateau_pattern(n_steps))
    
    # Pattern 7: Adaptive multimodal pattern
    initial_patterns.append(create_adaptive_multimodal_pattern(n_steps))
    
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
            optimizer = optax.adam(0.03)  # Slightly higher learning rate for faster convergence
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
            max_patience = 100  # Less patience since we're optimizing faster
            best_local_c2 = -1e10
            best_local_params = current_params
            
            for iteration in range(1000):  # More iterations for better convergence
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
            de_result = create_differential_evolution_optimized_function()
            de_c2 = compute_c2(de_result)
            if de_c2 > best_c2:
                best_c2 = de_c2
                best_result = de_result
        except Exception as e:
            print(f"Differential evolution failed: {e}")
    
    # If we still didn't find anything, return a good baseline pattern
    if best_result is not None:
        # Do final extensive local refinement on the best result
        try:
            refined_result = best_result.copy()
            current_c2 = compute_c2(refined_result)
            
            # Aggressive local search with many iterations
            for _ in range(500):  # Moderate iterations for thorough refinement
                if time.time() - start_time > max_time:
                    break
                    
                candidate = refined_result.copy()
                # Make small random changes with more adaptive magnitude
                num_changes = max(1, len(candidate) // 20)  # About 5% of elements
                indices = np.random.choice(len(candidate), num_changes, replace=False)
                
                for idx in indices:
                    # Small random change with adaptive magnitude - more aggressive
                    change = random.uniform(-0.02, 0.02)
                    candidate[idx] = max(0, candidate[idx] + change)
                
                # Normalize
                total = sum(candidate)
                if total > 0:
                    candidate = [x / total * 100 for x in candidate]
                
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
    return create_multi_strategy_optimization()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
