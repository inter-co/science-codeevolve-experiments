# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
from scipy.optimize import differential_evolution
import random
from typing import List

def compute_autocorrelation_constant(sequence: List[float]) -> float:
    """
    Compute the first autocorrelation inequality constant C₁ for a given sequence.
    C₁ = 2n * max(convolution) / (sum(sequence))^2
    We want to maximize 1/C₁, which means minimizing C₁.
    """
    if len(sequence) == 0:
        return float('inf')
    
    sum_seq = sum(sequence)
    if sum_seq < 0.01:
        return float('inf')  # Reject sequences with too small sum
    
    # Compute convolution (auto-correlation)
    # Using full convolution to get all correlation values
    conv = convolve(sequence, sequence, mode='full')
    
    # Take only the valid correlation values (middle part)
    # For a sequence of length n, the convolution gives 2n-1 elements
    # The peak correlation occurs at the center (index n-1)
    max_conv = np.max(conv)
    
    n = len(sequence)
    c1 = (2 * n * max_conv) / (sum_seq ** 2)
    return c1

def objective_function(sequence: List[float]) -> float:
    """
    Objective function to minimize: C₁
    We return C₁ directly since we want to minimize it.
    """
    c1 = compute_autocorrelation_constant(sequence)
    return c1 if not np.isinf(c1) else 1e10

def compute_inv_c1(sequence: List[float]) -> float:
    """
    Compute 1/C₁ for reporting purposes
    """
    c1 = compute_autocorrelation_constant(sequence)
    if np.isinf(c1) or c1 == 0:
        return 0.0
    return 1.0 / c1

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with non-negative values"""
    return [random.uniform(0.1, 100.0) for _ in range(length)]

def optimize_sequence() -> List[float]:
    """
    Optimizing approach using evolutionary algorithms with proper constraints.
    This is a completely different approach from the original LP-based method.
    """
    # Try different sequence lengths to find optimal
    best_sequence = None
    best_inv_c1 = 0.0
    best_c1 = float('inf')
    
    # Test various sequence lengths
    test_lengths = [10, 20, 50, 100, 200, 500]
    
    for length in test_lengths:
        # Generate initial population
        population_size = min(50, max(10, length))
        
        # Create initial population with random sequences
        initial_population = []
        for _ in range(population_size):
            seq = generate_random_sequence(length)
            # Ensure some minimum sum
            if sum(seq) < 0.01:
                seq = [x * 100.0 / sum(seq) if sum(seq) > 0 else 1.0 for x in seq]
            initial_population.append(seq)
        
        try:
            # Use differential evolution for global optimization
            bounds = [(0.01, 1000.0) for _ in range(length)]
            
            result = differential_evolution(
                lambda x: compute_autocorrelation_constant(x),
                bounds,
                maxiter=100,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            if result.success:
                optimized_seq = result.x.tolist()
                inv_c1 = compute_inv_c1(optimized_seq)
                
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = optimized_seq
                    best_c1 = result.fun
                    
        except Exception as e:
            continue  # Skip this length if optimization fails
    
    # If no good solution found, return a simple well-performing sequence
    if best_sequence is None:
        # Try a simple construction that often works well
        # Based on known good constructions for this problem
        best_sequence = [1.0] * 100  # Simple uniform sequence
        # Add some variation to make it more interesting
        for i in range(10):
            if i % 2 == 0:
                best_sequence[i] = 1.5
            else:
                best_sequence[i] = 0.5
    
    return best_sequence

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    # Use the more robust optimization approach
    sequence = optimize_sequence()
    
    # Final verification and adjustment
    inv_c1 = compute_inv_c1(sequence)
    if inv_c1 < 0.01:
        # If we didn't get a good result, try a different approach
        sequence = [1.0] * 50  # Simple uniform sequence as fallback
        sequence[0] = 2.0  # Make it slightly asymmetric
        sequence[-1] = 0.5
    
    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
