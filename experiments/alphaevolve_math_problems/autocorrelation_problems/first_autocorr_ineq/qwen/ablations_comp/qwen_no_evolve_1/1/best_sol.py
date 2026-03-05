# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import convolve
from scipy.optimize import differential_evolution
from numba import jit

@jit(nopython=True)
def compute_convolution_fast(a):
    """Compute convolution of a with itself using optimized approach"""
    n = len(a)
    # Using the fact that convolution is symmetric
    b = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            b[i+j] += a[i] * a[j]
    return b

def compute_c1_value(a):
    """Compute C₁ value for given sequence a"""
    if len(a) == 0:
        return float('inf')
    
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')
        
    # Compute convolution b = a * a
    b = convolve(a, a, mode='full')  # This gives us the full convolution
    
    # Get the maximum value in convolution
    max_b = np.max(b)
    
    # Compute C₁ = 2n * max(b) / (sum(a))^2
    n = len(a)
    c1 = 2 * n * max_b / (sum_a ** 2)
    
    return c1

def compute_inv_c1(a):
    """Compute 1/C₁ value for given sequence a"""
    c1 = compute_c1_value(a)
    if c1 == float('inf'):
        return 0
    return 1.0 / c1

def create_optimal_step_function():
    """Create an optimal step function using a heuristic approach"""
    # Try different configurations to find good candidates
    best_inv_c1 = 0
    best_sequence = []
    
    # Try various lengths and configurations
    for n in range(10, 1000, 50):  # Test different lengths
        # Create some promising step function patterns
        # Pattern 1: Simple decreasing pattern
        pattern1 = np.array([1.0] + [1.0/(i+1) for i in range(1, n)])
        pattern1 = pattern1[:n]  # Ensure correct length
        
        # Pattern 2: Peak at beginning
        pattern2 = np.array([1.0] + [0.5] * (n-1))
        pattern2 = pattern2[:n]
        
        # Pattern 3: Gaussian-like decay
        x = np.linspace(0, 1, n)
        pattern3 = np.exp(-x**2 * 10) * 2.0
        
        # Pattern 4: Alternating pattern
        pattern4 = np.array([1.0 if i % 2 == 0 else 0.5 for i in range(n)])
        
        patterns = [pattern1, pattern2, pattern3, pattern4]
        
        for pattern in patterns:
            # Normalize to have reasonable sum
            pattern = pattern / np.sum(pattern) * 10
            inv_c1 = compute_inv_c1(pattern)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = pattern.copy()
    
    return best_sequence

def optimize_with_evolution():
    """Use evolutionary optimization to improve step functions"""
    # Start with a good initial guess
    initial_sequence = create_optimal_step_function()
    
    # Define objective function to minimize (negative of 1/C₁)
    def objective(x):
        # Clip values to valid range
        x = np.clip(x, 0, 1000)
        inv_c1 = compute_inv_c1(x)
        return -inv_c1  # Negative because we want to maximize
    
    # Use differential evolution for global optimization
    try:
        # Use a smaller subset for faster optimization
        n = min(len(initial_sequence), 200)
        bounds = [(0, 1000) for _ in range(n)]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if result.success:
            optimized_sequence = np.clip(result.x, 0, 1000)
            return optimized_sequence.tolist()
    except:
        pass
    
    return initial_sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Multiple attempts with different strategies
    best_inv_c1 = 0
    best_sequence = []
    
    # Strategy 1: Direct optimization
    sequence1 = optimize_with_evolution()
    inv_c1_1 = compute_inv_c1(sequence1)
    
    # Strategy 2: Random sampling with good heuristics
    best_random = []
    best_random_inv_c1 = 0
    
    for _ in range(50):
        # Generate random step function with some structure
        n = random.randint(50, 500)
        # Use exponential decay for more promising results
        decay_factor = random.uniform(0.8, 0.99)
        sequence = [1.0 * (decay_factor ** i) for i in range(n)]
        sequence = np.array(sequence)
        sequence = sequence / np.sum(sequence) * 10  # Normalize
        
        inv_c1 = compute_inv_c1(sequence)
        if inv_c1 > best_random_inv_c1:
            best_random_inv_c1 = inv_c1
            best_random = sequence.tolist()
    
    # Compare results
    if inv_c1_1 > best_random_inv_c1:
        return sequence1
    else:
        return best_random

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
