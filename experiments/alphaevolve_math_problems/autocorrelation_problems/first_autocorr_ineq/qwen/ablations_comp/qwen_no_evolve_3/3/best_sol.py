# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
from scipy.optimize import differential_evolution, minimize
import random
from typing import List, Tuple
import time

def compute_c1(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute C1 and 1/C1 for a given sequence.
    Returns (C1, 1/C1)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Ensure sequence has positive sum
    total_sum = sum(sequence)
    if total_sum < 0.01:
        return float('inf'), 0.0
    
    # Compute convolution (auto-correlation)
    # Using fast convolution for better performance
    conv_result = convolve(sequence, sequence, mode='full')
    
    # Maximum value in convolution (excluding trivial zero elements)
    max_conv = max(conv_result)
    
    # Compute C1 = 2n * max(b) / (sum(a))^2
    n = len(sequence)
    c1 = (2 * n * max_conv) / (total_sum ** 2)
    
    # Return both C1 and its reciprocal
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
    return c1, inv_c1

def objective_function(sequence: List[float]) -> float:
    """
    Objective function to minimize (negative of 1/C1).
    We want to maximize 1/C1, so we minimize -1/C1.
    """
    c1, inv_c1 = compute_c1(sequence)
    # Return negative of 1/C1 to convert maximization to minimization
    return -inv_c1 if inv_c1 > 0 else 1e10

def create_random_sequence(length: int) -> List[float]:
    """Create a random sequence with positive values."""
    # Generate random values in [0, 1000] range
    return [random.uniform(0.1, 1000.0) for _ in range(length)]

def create_step_function_from_params(params: List[float]) -> List[float]:
    """Convert parameter vector to step function with bounded heights."""
    # Clip parameters to [0, 1000] range
    clipped_params = [max(0.0, min(1000.0, p)) for p in params]
    return clipped_params

def optimize_with_evolutionary_search(max_time_seconds: float = 50.0) -> List[float]:
    """
    Use evolutionary algorithms to find good step function.
    """
    start_time = time.time()
    
    # Define bounds for each parameter (heights)
    # Using bounds [0, 1000] for each height
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Try different sequence lengths
    lengths_to_try = [10, 20, 50, 100, 200, 500]
    
    for length in lengths_to_try:
        if time.time() - start_time > max_time_seconds * 0.8:
            break
            
        # Create bounds for this length
        bounds = [(0.0, 1000.0)] * length
        
        # Use differential evolution for global optimization
        try:
            result = differential_evolution(
                lambda x: objective_function(create_step_function_from_params(x)),
                bounds,
                maxiter=100,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            if result.success:
                sequence = create_step_function_from_params(result.x)
                _, inv_c1 = compute_c1(sequence)
                
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = sequence
                    
        except Exception as e:
            continue
    
    # If no good solution found, return random sequence
    if best_sequence is None:
        best_sequence = create_random_sequence(100)
        
    return best_sequence

def optimize_with_gradient_free_methods(max_time_seconds: float = 50.0) -> List[float]:
    """
    Use gradient-free optimization methods to find good step function.
    """
    start_time = time.time()
    
    # Try different initial configurations
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Different initialization strategies
    strategies = [
        lambda: create_random_sequence(random.randint(50, 500)),
        lambda: [100.0] * random.randint(10, 100),  # Uniform heights
        lambda: [1000.0 if i % 2 == 0 else 10.0 for i in range(random.randint(50, 300))],  # Alternating pattern
    ]
    
    for strategy in strategies:
        if time.time() - start_time > max_time_seconds * 0.8:
            break
            
        try:
            # Try local optimization from this starting point
            initial_seq = strategy()
            bounds = [(0.0, 1000.0)] * len(initial_seq)
            
            # Use L-BFGS-B with bounds
            result = minimize(
                lambda x: objective_function(x),
                initial_seq,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50},
                tol=1e-6
            )
            
            if result.success:
                sequence = create_step_function_from_params(result.x)
                _, inv_c1 = compute_c1(sequence)
                
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = sequence
                    
        except Exception as e:
            continue
    
    # If no good solution found, return random sequence
    if best_sequence is None:
        best_sequence = create_random_sequence(100)
        
    return best_sequence

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence.
    Uses multiple optimization strategies to maximize 1/C1.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Try evolutionary approach first
    try:
        sequence = optimize_with_evolutionary_search(50.0)
        _, inv_c1 = compute_c1(sequence)
        
        # If we have a decent solution, return it
        if inv_c1 > 0.6:
            return sequence
            
    except Exception as e:
        pass
    
    # Fallback to other methods
    try:
        sequence = optimize_with_gradient_free_methods(50.0)
        return sequence
    except Exception as e:
        # Final fallback
        return create_random_sequence(200)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
