# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import time

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    f_values: step function heights
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    # Create step function with proper spacing on [-1/4, 1/4]
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Define the domain
    x = np.linspace(-0.25, 0.25, 2*n)  # Double resolution for better accuracy
    dx = x[1] - x[0]
    
    # Create step function (piecewise constant)
    f = np.zeros_like(x)
    step_width = 0.5 / n
    for i in range(n):
        start_idx = int(i * 2)
        end_idx = int((i + 1) * 2)
        f[start_idx:end_idx] = f_values[i]
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = convolve(f, f, mode='full')
    g = g[len(g)//2:]  # Take the right half
    g = g[:len(x)]     # Trim to original size
    
    # Compute norms
    # ||g||₂² = ∫ g² dx ≈ sum(g² * dx)
    norm_g2_squared = np.sum(g**2) * dx
    
    # ||g||₁ = ∫ |g| dx ≈ sum(|g|) * dx  
    norm_g1 = np.sum(np.abs(g)) * dx
    
    # ||g||∞ = max |g|
    norm_ginf = np.max(np.abs(g))
    
    return norm_g2_squared, norm_g1, norm_ginf

def evaluate_c2(f_values: list[float]) -> float:
    """
    Evaluate C2 = ||g||₂² / (||g||₁ · ||g||∞)
    Returns negative C2 since we're maximizing (but scipy minimizes)
    """
    try:
        norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_g1 <= 1e-15 or norm_ginf <= 1e-15:
            return -1e10  # Very poor value
            
        c2 = norm_g2_sq / (norm_g1 * norm_ginf)
        return -c2  # Return negative for minimization
    except Exception:
        return -1e10

def construct_function() -> list[float]:
    """
    Construct step function using evolutionary optimization approach.
    Uses a hybrid strategy combining global search with local refinement.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Start with a good baseline
    initial_guess = [0.5] * 50  # Simple uniform distribution
    
    # Use differential evolution for global optimization
    # This is a robust evolutionary algorithm suitable for this problem
    bounds = [(0, 1) for _ in range(50)]
    
    try:
        result = differential_evolution(
            evaluate_c2, 
            bounds, 
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        # Extract best solution and refine it
        best_solution = result.x
        # Clip to [0,1] and round to reasonable precision
        final_solution = [max(0, min(1, val)) for val in best_solution]
        
        # Apply additional refinement with simple hill climbing
        current_eval = evaluate_c2(final_solution)
        for _ in range(20):  # Limited local search
            # Perturb one element at a time
            idx = np.random.randint(len(final_solution))
            old_val = final_solution[idx]
            
            # Try small perturbations
            perturbations = [-0.05, -0.02, 0.02, 0.05]
            best_perturbed = final_solution.copy()
            best_perturbed_eval = current_eval
            
            for delta in perturbations:
                test_solution = final_solution.copy()
                test_solution[idx] = max(0, min(1, old_val + delta))
                eval_result = evaluate_c2(test_solution)
                
                if eval_result < best_perturbed_eval:
                    best_perturbed = test_solution
                    best_perturbed_eval = eval_result
            
            if best_perturbed_eval < current_eval:
                final_solution = best_perturbed
                current_eval = best_perturbed_eval
            else:
                break
                
        return final_solution
        
    except Exception:
        # Fallback to simple heuristic if optimization fails
        return [0.5] * 50

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
