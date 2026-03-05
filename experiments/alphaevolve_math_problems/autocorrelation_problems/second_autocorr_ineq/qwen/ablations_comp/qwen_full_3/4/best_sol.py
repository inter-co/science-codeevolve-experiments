# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import minimize
from scipy.linalg import toeplitz
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)
    
    # Create the step function on [-1/4, 1/4] with equal spacing
    n = len(f)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Autoconvolve f with itself using full mode
    g = signal.convolve(f, f, mode='full')
    
    # For symmetric convolution, we need the middle part
    # The result is of length 2*n-1, so we extract the center n values
    start_idx = n - 1
    g_middle = g[start_idx:start_idx + n]
    
    # Compute the three norms according to the specification:
    # ||g||₂² (L2 norm squared) using piecewise linear integration
    # This is done by computing the sum of (g[i]^2 + g[i]*g[i+1] + g[i+1]^2)/3 for adjacent pairs
    norm_g_2_squared = 0.0
    if len(g_middle) > 1:
        # For piecewise linear integration of g^2 over segments
        # Each segment [i, i+1] contributes (1/3)*(g[i]^2 + g[i]*g[i+1] + g[i+1]^2) 
        for i in range(len(g_middle) - 1):
            contribution = (g_middle[i]**2 + g_middle[i]*g_middle[i+1] + g_middle[i+1]**2) / 3.0
            norm_g_2_squared += contribution
    else:
        norm_g_2_squared = g_middle[0]**2
    
    # ||g||₁ (L1 norm) - approximate as sum divided by number of points + 1
    norm_g_1 = np.sum(np.abs(g_middle)) / (len(g_middle) + 1) if len(g_middle) > 0 else 0.0
    
    # ||g||∞ (infinity norm)
    norm_g_inf = np.max(np.abs(g_middle)) if len(g_middle) > 0 else 0.0
    
    return norm_g_2_squared, norm_g_1, norm_g_inf

def compute_c2(f_values: list[float]) -> float:
    """
    Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞) where g = f*f is the autoconvolution.
    """
    try:
        norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_g_1 <= 1e-15 or norm_g_inf <= 1e-15:
            return 0.0
            
        # Calculate C2
        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception:
        return 0.0

def construct_function() -> list[float]:
    """
    Construct a step function using a novel analytical and sparse optimization approach:
    1. Analytical construction based on extremal principles
    2. Sparse optimization focusing on key support points
    3. Hierarchical refinement strategy
    """
    # Use fewer steps to avoid memory issues and computational overload
    n_steps = 1000  # Reduced from 2000 to stay within limits
    
    # Strategy 1: Analytical construction using known extremal principles
    # Based on the principle that to maximize C₂, we want g to have a flat profile
    # This suggests a function with minimal variation in its autoconvolution
    def analytical_initialization(n):
        # Create a function that tends to produce flatter autoconvolutions
        # Start with a symmetric pattern that's peaked at center
        x = np.linspace(-0.25, 0.25, n)
        # Use a pattern that's more uniform in nature
        # A good candidate: piecewise constant with sharp transitions
        # But let's try a smoother approach that still promotes flat g
        pattern = np.ones(n)
        
        # Add some structure that helps with flat autoconvolution
        # Use a cosine-based pattern with specific frequencies
        frequencies = [1, 2, 3, 4]  # Low frequencies
        for freq in frequencies:
            pattern *= (1 + 0.3 * np.cos(freq * np.pi * x / 0.25))
        
        # Make it symmetric and non-negative
        pattern = np.maximum(0, pattern)
        
        # Normalize to reasonable scale
        if np.max(pattern) > 0:
            pattern = pattern / np.max(pattern) * 100
            
        return pattern.tolist()
    
    # Strategy 2: Sparse optimization approach - focus on key parameters
    def sparse_optimization():
        # Instead of optimizing all 1000 parameters, optimize a smaller set of key parameters
        # Then interpolate to fill the rest
        
        # Start with a simpler version - focus on 5 key points
        n_key_points = 5
        key_positions = np.linspace(0, n_steps - 1, n_key_points, dtype=int)
        
        # Initialize key points with good values
        key_values = [80, 100, 120, 100, 80]  # Symmetric peak pattern
        
        # Create full function by interpolating
        def interpolate_full(key_vals):
            # Create full array with interpolated values
            full_func = np.zeros(n_steps)
            
            # Interpolate between key points
            positions = np.linspace(0, n_steps - 1, n_key_points)
            for i in range(n_key_points):
                if i < n_key_points - 1:
                    # Linear interpolation between points
                    start_pos = positions[i]
                    end_pos = positions[i + 1]
                    start_val = key_vals[i]
                    end_val = key_vals[i + 1]
                    
                    # Fill in the range
                    mask = (np.arange(n_steps) >= start_pos) & (np.arange(n_steps) <= end_pos)
                    if np.any(mask):
                        # Linear interpolation
                        t = (np.arange(n_steps)[mask] - start_pos) / (end_pos - start_pos)
                        full_func[mask] = start_val + t * (end_val - start_val)
                else:
                    # Last point
                    full_func[int(positions[i]):] = key_vals[i]
            
            return full_func.tolist()
        
        # Optimization of key points
        def objective(key_vals):
            full_func = interpolate_full(key_vals)
            # Ensure non-negativity
            full_func = [max(0, val) for val in full_func]
            return -compute_c2(full_func)
        
        # Optimize just the key values
        try:
            # Use a simpler optimization approach
            from scipy.optimize import differential_evolution
            
            # Bounds for key values (reasonable range)
            bounds = [(0, 200) for _ in range(n_key_points)]
            
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                disp=False
            )
            
            if result.success:
                final_key_vals = result.x
                return interpolate_full(final_key_vals)
                
        except Exception as e:
            print(f"Sparse optimization failed: {e}")
        
        # Fallback to initial pattern
        return interpolate_full(key_values)
    
    # Strategy 3: Fourier domain approach - analyze what frequency characteristics are optimal
    def fourier_analysis_approach():
        # Analyze the frequency domain properties that would lead to good C₂
        # The key insight is that for maximizing C₂, we want ||g||₂² to dominate the product ||g||₁ · ||g||∞
        # This typically means we want g to have a relatively flat profile
        
        # Create a function that tries to maximize energy concentration in the middle
        # This often leads to flatter autoconvolutions
        
        # Start with a simple symmetric pattern that's peaked in center
        # But make it more uniform to encourage flat g
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Try a pattern that balances peak and spread
        # Use a hyperbolic tangent-like shape for smooth transitions
        # But keep it peaked enough to create good autoconvolution
        pattern = 100 * (1 - np.tanh(8 * x)**2)
        pattern = np.maximum(0, pattern)
        
        return pattern.tolist()
    
    # Strategy 4: Hierarchical refinement approach
    def hierarchical_refinement():
        # Start with coarse grid, then refine
        coarse_n = 100
        coarse_pattern = fourier_analysis_approach()[:coarse_n]
        
        # Refine using interpolation or local optimization
        # For simplicity, we'll just return the fourier approach result
        return fourier_analysis_approach()
    
    # Execute the most promising approach
    try:
        # Try sparse optimization first (most efficient)
        result = sparse_optimization()
        
        # Verify and improve if needed
        if result and len(result) == n_steps:
            # Check if we got a reasonable value
            c2_val = compute_c2(result)
            if c2_val > 0.5:  # If it's reasonably good, return it
                return result
            else:
                # Fall back to analytical approach
                return analytical_initialization(n_steps)
        else:
            # Fall back to analytical approach
            return analytical_initialization(n_steps)
            
    except Exception as e:
        print(f"Hierarchical approach failed: {e}")
        # Final fallback
        return analytical_initialization(n_steps)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
