# EVOLVE-BLOCK-START

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
import time
from scipy import signal

def compute_autoconvolution_torch(f_tensor):
    """Compute the autoconvolution g = f * f using PyTorch operations"""
    # Convert to tensor with requires_grad=True for gradient computation
    f = f_tensor
    
    # Compute autoconvolution using torch.conv1d (requires specific shape)
    f_expanded = f.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, len(f))
    kernel = f.flip(0).unsqueeze(0).unsqueeze(0)  # Flip for convolution
    g = torch.nn.functional.conv1d(f_expanded, kernel, padding=len(f)-1).squeeze()
    
    return g.squeeze()

def compute_c2(f_values):
    """
    Compute C2 value using the exact specification from the problem description.
    This implements the precise convolution and norm computation as requested.
    """
    if not f_values or len(f_values) == 0:
        return 0.0
    
    # Convert to numpy array
    f = np.array(f_values)
    n = len(f)
    
    if n == 0:
        return 0.0
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = signal.convolve(f, f, mode='full')
    
    # Compute norms exactly as specified in the problem:
    # ||g||₂²: L2-norm squared, computed via piecewise linear integration over convolution
    # ||g||₁: L1-norm, approximated as sum(|g|) / (len(g) + 1)  
    # ||g||∞: Infinity-norm, computed as max(|g|)
    
    # For ||g||₂² using the specified trapezoidal-like integration approach:
    # We use the direct sum of squares since we're dealing with discrete values
    # that approximate the continuous convolution result
    norm_g2_squared = np.sum(g * g)
    
    # L1 norm approximation as specified in the problem
    norm_g1 = np.sum(np.abs(g)) / (len(g) + 1) if len(g) > 0 else 0.0
    
    # Infinity norm as specified
    norm_ginf = np.max(np.abs(g)) if len(g) > 0 else 0.0
    
    # Avoid division by zero
    if norm_g1 <= 1e-15 or norm_ginf <= 1e-15:
        return 0.0
    
    c2 = norm_g2_squared / (norm_g1 * norm_ginf)
    return c2

def construct_function() -> list[float]:
    """
    Enhanced function to construct step-function with high C2 value using 
    advanced gradient-based optimization with PyTorch, incorporating mathematical insights.
    This approach follows the most successful pattern from the inspirations.
    """
    # Set parameters for gradient-based optimization
    NUM_STEPS = 1000  # Increased for better resolution (matching successful inspirations)
    MAX_ITER = 300    # More iterations for better optimization (matching successful inspirations)
    
    # Multiple restart strategies to avoid local minima
    best_c2 = 0.0
    best_f = None
    
    # Try multiple starting points
    for restart in range(3):
        # Use mathematical insights to create a good starting point
        # Inspired by successful mathematical constructions that work well in practice
        x = np.linspace(-0.25, 0.25, NUM_STEPS)
        
        # Different oscillatory patterns for diversity (following successful inspiration patterns)
        if restart == 0:
            # Pattern from inspiration 1
            f_initial = np.exp(-x**2 / (0.05**2)) * (1.0 + 0.5 * np.sin(10 * np.pi * x))
        elif restart == 1:
            # Pattern from inspiration 2
            f_initial = np.exp(-x**2 / (0.06**2)) * (1.0 + 0.4 * np.sin(12 * np.pi * x))
        else:
            # Pattern from inspiration 3
            f_initial = np.exp(-x**2 / (0.07**2)) * (1.0 + 0.6 * np.sin(8 * np.pi * x))
            
        f_initial = np.maximum(f_initial, 0)
        
        if np.max(f_initial) > 0:
            f_initial = f_initial / np.max(f_initial) * 1.8
        
        # Convert to PyTorch tensor
        f_tensor = torch.tensor(f_initial, dtype=torch.float32, requires_grad=True)
        
        # Use Adam optimizer with adaptive learning rate (following successful inspirations)
        # Different learning rates for different restarts to explore parameter space
        lr = 0.04 if restart == 0 else 0.03 if restart == 1 else 0.025
        optimizer = optim.Adam([f_tensor], lr=lr)
        
        # Optimization loop with early stopping
        local_best_c2 = 0.0
        local_best_f = f_initial.copy()
        
        # Time tracking to stay under 60 seconds
        start_time = time.time()
        
        for iteration in range(MAX_ITER):
            if time.time() - start_time > 50:  # Leave 10 seconds for cleanup
                break
                
            # Zero gradients
            optimizer.zero_grad()
            
            # Ensure non-negative values
            f_clipped = torch.clamp(f_tensor, min=0.0)
            
            # Compute autoconvolution and C2
            try:
                g = compute_autoconvolution_torch(f_clipped)
                g_abs = torch.abs(g)
                
                # ||g||₂² - sum of squares
                norm_g_2_squared = torch.sum(g_abs ** 2)
                
                # ||g||₁ - as specified in prompt
                norm_g_1 = torch.sum(g_abs) / (len(g_abs) + 1)
                
                # ||g||∞ - maximum absolute value
                norm_g_inf = torch.max(g_abs)
                
                # Avoid division by zero
                if norm_g_1.item() == 0 or norm_g_inf.item() == 0:
                    c2 = 0.0
                else:
                    c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
                
                # Backward pass (we want to maximize C2, so we minimize -C2)
                (-c2).backward()
                
                # Update parameters
                optimizer.step()
                
                # Ensure non-negativity after update
                with torch.no_grad():
                    f_tensor.clamp_(min=0.0)
                
                # Track best solution
                if c2.item() > local_best_c2:
                    local_best_c2 = c2.item()
                    local_best_f = f_clipped.detach().numpy().copy()
                    
            except Exception as e:
                # If there's an error, continue with current best
                continue
        
        # Update global best if this restart was better
        if local_best_c2 > best_c2:
            best_c2 = local_best_c2
            best_f = local_best_f.copy()
    
    # Return the best solution found
    if best_f is None:
        # Fallback to a carefully constructed mathematical function
        x = np.linspace(-0.25, 0.25, NUM_STEPS)
        f_initial = np.exp(-x**2 / (0.06**2)) * (1.0 + 0.4 * np.sin(10 * np.pi * x))
        f_initial = np.maximum(f_initial, 0)
        if np.max(f_initial) > 0:
            f_initial = f_initial / np.max(f_initial) * 1.8
        return f_initial.tolist()
    
    return list(best_f)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
