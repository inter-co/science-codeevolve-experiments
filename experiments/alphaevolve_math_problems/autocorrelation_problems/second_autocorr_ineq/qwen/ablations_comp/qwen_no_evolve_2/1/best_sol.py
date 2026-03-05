# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from typing import List
import random
import time

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    Uses fast convolution and proper integration.
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Create step function on [-1/4, 1/4] with appropriate spacing
    n_steps = len(f)
    dx = 0.5 / (n_steps - 1) if n_steps > 1 else 0.5
    
    # Perform autoconvolution using scipy's fft-based convolution
    # This gives us g = f * f (convolution)
    g = signal.convolve(f, f, mode='full')
    
    # Adjust indices for the actual domain
    # Original f is defined on [-1/4, 1/4], so convolution spans [-1/2, 1/2]
    # But we're interested in the overlap region [-1/4, 1/4]
    center_idx = len(g) // 2
    half_len = n_steps - 1
    
    # Extract the relevant portion ([-1/4, 1/4] in terms of convolution)
    g_relevant = g[center_idx - half_len:center_idx + half_len + 1]
    
    # Compute norms properly
    # ||g||₂² using trapezoidal integration (piecewise linear)
    g_squared = g_relevant ** 2
    g_abs = np.abs(g_relevant)
    
    # Trapezoidal rule for L2 norm squared
    # For piecewise linear segments, we use the formula for integral of quadratic
    # But since we're doing discrete integration, we'll use a more accurate approach:
    # Using Simpson's rule approximation for better accuracy
    if len(g_relevant) >= 3:
        # Use Simpson's rule for better integration
        # We need to compute ∫ g(x)² dx approximately
        # For simpson's rule, we need odd number of points
        if len(g_relevant) % 2 == 0:
            g_relevant = g_relevant[:-1]
            g_squared = g_squared[:-1]
        
        # Simpson's rule coefficients
        weights = np.ones_like(g_relevant)
        weights[1::2] = 4  # odd indices get weight 4
        weights[2::2] = 2  # even indices get weight 2 (except first and last)
        
        # Simpson's rule: h/3 * sum(weights * values)
        # Width is dx
        norm_2_squared = (dx / 3) * np.sum(weights * g_squared)
    else:
        # Fallback to simple trapezoidal rule
        norm_2_squared = dx * np.sum(g_squared[:-1] + g_squared[1:]) / 2
    
    # ||g||₁ (L1 norm)
    norm_1 = np.sum(g_abs) * dx
    
    # ||g||∞ (L-infinity norm)
    norm_inf = np.max(g_abs)
    
    return norm_2_squared, norm_1, norm_inf

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 value for given step function."""
    try:
        norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2
    except Exception:
        return 0.0

def generate_random_step_function(n_steps: int) -> List[float]:
    """Generate a random step function with non-negative values."""
    # Generate normally distributed values then clip to non-negative
    values = np.random.normal(0, 1, n_steps)
    values = np.maximum(values, 0)
    # Normalize to reasonable scale
    if np.sum(values) > 0:
        values = values / np.sum(values) * 10
    return values.tolist()

def mutate_step_function(f_values: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply mutation to a step function."""
    new_values = f_values.copy()
    
    for i in range(len(new_values)):
        if random.random() < mutation_rate:
            # Add Gaussian noise
            new_values[i] += np.random.normal(0, 0.1 * max(1, new_values[i]))
            # Ensure non-negative
            new_values[i] = max(0, new_values[i])
    
    return new_values

def crossover_step_functions(f1: List[float], f2: List[float]) -> List[float]:
    """Perform crossover between two step functions."""
    # Simple uniform crossover
    min_len = min(len(f1), len(f2))
    crossover_point = random.randint(1, min_len - 1)
    
    new_values = f1[:crossover_point] + f2[crossover_point:]
    
    # Ensure lengths match by padding or truncating
    if len(new_values) < len(f1):
        # Pad with zeros or last values
        new_values.extend([0] * (len(f1) - len(new_values)))
    elif len(new_values) > len(f1):
        new_values = new_values[:len(f1)]
        
    return new_values

def optimize_with_evolutionary_algorithm(max_time: float = 60.0) -> List[float]:
    """
    Evolutionary optimization approach for maximizing C2.
    Uses genetic algorithm with specialized operators.
    """
    start_time = time.time()
    
    # Initialize population
    population_size = 50
    max_generations = 1000
    population = []
    
    # Create initial population with varying sizes
    for _ in range(population_size):
        n_steps = random.randint(100, 1000)
        individual = generate_random_step_function(n_steps)
        population.append(individual)
    
    best_individual = None
    best_c2 = -1
    
    generation = 0
    while time.time() - start_time < max_time * 0.9 and generation < max_generations:
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            c2 = calculate_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_c2:
                best_c2 = c2
                best_individual = individual.copy()
        
        # Selection (tournament selection)
        selected_population = []
        tournament_size = 3
        
        for _ in range(population_size):
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected_population.append(population[winner_index].copy())
        
        # Crossover and mutation
        new_population = []
        
        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual)
        
        # Generate offspring
        while len(new_population) < population_size:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            
            # Crossover
            child = crossover_step_functions(parent1, parent2)
            
            # Mutation
            child = mutate_step_function(child, mutation_rate=0.1)
            
            # Occasionally change size
            if random.random() < 0.1:
                new_size = max(100, min(2000, len(child) + random.randint(-50, 50)))
                if new_size != len(child):
                    if new_size > len(child):
                        # Extend with zeros
                        child.extend([0] * (new_size - len(child)))
                    else:
                        # Truncate
                        child = child[:new_size]
            
            new_population.append(child)
        
        population = new_population[:population_size]
        generation += 1
    
    return best_individual if best_individual is not None else generate_random_step_function(500)

def construct_function() -> List[float]:
    """Main function to construct optimized step-function with high C2 value."""
    # Try several approaches to find better solutions
    best_result = None
    best_c2 = -1
    
    # Run multiple optimization attempts
    for attempt in range(3):
        try:
            # Use evolutionary algorithm
            result = optimize_with_evolutionary_algorithm(max_time=20.0)
            c2 = calculate_c2(result)
            
            if c2 > best_c2:
                best_c2 = c2
                best_result = result
                
        except Exception as e:
            continue
    
    # If no good result found, fallback to a well-tuned construction
    if best_result is None:
        # Create a carefully designed step function
        n_steps = 1000
        f_values = np.zeros(n_steps)
        
        # Create a pattern that balances smoothness and peakiness
        # Try to create something like a triangular or bell-shaped distribution
        x = np.linspace(-0.25, 0.25, n_steps)
        # Create a smooth bump in the center
        f_values = np.exp(-x**2 * 10) * 10
        
        # Normalize
        f_values = f_values / np.sum(f_values) * 10
        
        # Add some randomness to make it more interesting
        f_values = np.maximum(f_values + np.random.normal(0, 0.1, n_steps), 0)
        
        best_result = f_values.tolist()
        best_c2 = calculate_c2(best_result)
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
