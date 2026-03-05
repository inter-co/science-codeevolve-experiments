# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import convolve
from scipy.optimize import differential_evolution, minimize
import time
from typing import List, Tuple

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute C₁ = 2n * max(convolution) / (sum(sequence))²
    Returns (C₁, 1/C₁) where we want to maximize 1/C₁
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure no negative values
    sequence = [max(0, x) for x in sequence]
    
    # Skip if sum is too small
    total_sum = sum(sequence)
    if total_sum < 0.01:
        return float('inf'), 0.0
    
    # Compute convolution (auto-correlation)
    # Using numpy's fft-based convolution for efficiency
    conv_result = convolve(sequence, sequence, mode='full')
    
    # The auto-correlation has 2n-1 elements, but we only care about the middle part
    # The maximum occurs at the center (k=0 in the full convolution)
    max_conv = max(conv_result)
    
    n = len(sequence)
    if max_conv == 0:
        return float('inf'), 0.0
    
    c1 = 2 * n * max_conv / (total_sum ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    
    return c1, inv_c1

def create_random_step_function(size: int) -> List[float]:
    """Create a random step function with specified size"""
    # Generate random heights between 0 and 1000
    return [random.uniform(0, 1000) for _ in range(size)]

def create_optimized_step_function() -> List[float]:
    """
    Create an optimized step function using evolutionary computation
    This explores different patterns that might give better results
    """
    # Try different construction strategies
    strategies = [
        lambda n: [1.0] * n,  # Uniform distribution
        lambda n: [1.0 if i == 0 else 0.0 for i in range(n)],  # Single spike
        lambda n: [1.0 if i % 2 == 0 else 0.0 for i in range(n)],  # Alternating pattern
        lambda n: [1.0 if i < n//2 else 0.0 for i in range(n)],  # Half-filled
        lambda n: [i+1 for i in range(n)],  # Linear increasing
        lambda n: [n-i for i in range(n)],  # Linear decreasing
        lambda n: [1.0] + [0.0] * (n-1) if n > 1 else [1.0],  # Single peak
    ]
    
    best_sequence = []
    best_inv_c1 = 0.0
    
    # Try different sizes
    sizes = [10, 20, 50, 100, 200, 500]
    
    for size in sizes:
        for strategy in strategies:
            try:
                seq = strategy(size)
                _, inv_c1 = compute_autocorrelation_constant(seq)
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = seq.copy()
            except:
                continue
    
    # If no good sequence found, create random one
    if best_inv_c1 == 0.0:
        best_sequence = create_random_step_function(random.randint(50, 500))
    
    return best_sequence

def optimize_with_differential_evolution(initial_seq: List[float]) -> List[float]:
    """
    Use differential evolution to optimize the sequence
    """
    n = len(initial_seq)
    
    def objective(x):
        # Normalize the sequence to prevent overflow
        # Convert to proper sequence format
        seq = [max(0.0, x[i]) for i in range(len(x))]
        
        # Add some regularization to avoid trivial solutions
        total_sum = sum(seq)
        if total_sum < 0.01:
            return 1e10
            
        _, inv_c1 = compute_autocorrelation_constant(seq)
        # We want to maximize 1/C₁, so minimize -1/C₁
        return -inv_c1 if inv_c1 > 0 else 1e10
    
    # Set bounds for each variable (0 to 1000)
    bounds = [(0.0, 1000.0)] * n
    
    try:
        # Use differential evolution with bounds
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
            optimized_seq = [max(0.0, x) for x in result.x]
            return optimized_seq
    except:
        pass
    
    return initial_seq

def search_for_best_sequence() -> List[float]:
    """Main search function using multiple strategies"""
    start_time = time.time()
    
    # Strategy 1: Try different optimized constructions
    best_sequence = create_optimized_step_function()
    _, best_inv_c1 = compute_autocorrelation_constant(best_sequence)
    
    # Strategy 2: Try evolutionary optimization on a few candidates
    candidate_sequences = [create_optimized_step_function() for _ in range(5)]
    
    for seq in candidate_sequences:
        _, inv_c1 = compute_autocorrelation_constant(seq)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = seq.copy()
    
    # Strategy 3: Use differential evolution on the best found so far
    if time.time() - start_time < 50:  # Leave room for other optimizations
        try:
            optimized = optimize_with_differential_evolution(best_sequence)
            _, inv_c1 = compute_autocorrelation_constant(optimized)
            if inv_c1 > best_inv_c1:
                best_sequence = optimized
                best_inv_c1 = inv_c1
        except:
            pass
    
    # Final validation
    _, final_inv_c1 = compute_autocorrelation_constant(best_sequence)
    if final_inv_c1 < 0.01:  # Very poor quality
        # Fall back to a known good construction
        best_sequence = [1.0] * 100  # Simple uniform sequence
        _, best_inv_c1 = compute_autocorrelation_constant(best_sequence)
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
