# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import random
from scipy.signal import convolve
from scipy.optimize import differential_evolution
import time

def compute_autoconvolution(f: np.ndarray) -> np.ndarray:
    """Compute the autoconvolution g = f * f efficiently."""
    # Use scipy's convolution which handles the discrete case properly
    g = convolve(f, f, mode='full')
    # We only need the middle portion since we're working on [-1/4, 1/4]
    # The convolution will be of length 2*n - 1, centered at n-1
    center = len(g) // 2
    return g[center - len(f) + 1:center + len(f)]

def compute_c2_norms(g: np.ndarray) -> tuple[float, float, float]:
    """Compute the three norms needed for C2 calculation."""
    # L2 norm squared
    # Using trapezoidal rule approximation for integral of g^2
    # For piecewise linear segments, we use the formula for integral of quadratic
    if len(g) < 2:
        return 0.0, 0.0, 0.0
    
    # Compute L2 norm squared using trapezoidal rule for g^2
    g_squared = g * g
    # For piecewise linear integration, we approximate with trapezoidal rule
    # But since we're dealing with discrete values, we'll use a more accurate approach
    # For a trapezoid with heights y1, y2 and width h, area is h*(y1+y2)/2
    # But for integral of squares, we use Simpson's rule approximation or direct sum
    # Let's compute it as sum of trapezoids of g^2
    if len(g_squared) >= 2:
        # Trapezoidal rule for ∫g²dx
        # We assume uniform spacing
        h = 0.5 / (len(g) - 1)  # width of each interval
        # Sum of trapezoidal areas for g^2
        trap_sum = 0.5 * (g_squared[0] + g_squared[-1])
        trap_sum += np.sum(g_squared[1:-1])
        l2_norm_sq = h * trap_sum
    else:
        l2_norm_sq = g_squared[0] if len(g_squared) > 0 else 0.0
    
    # L1 norm
    l1_norm = np.sum(np.abs(g))
    
    # L-infinity norm  
    linf_norm = np.max(np.abs(g))
    
    return l2_norm_sq, l1_norm, linf_norm

def compute_c2(f_values: List[float]) -> tuple[float, float, float]:
    """Compute C2 value for given step function values."""
    if len(f_values) == 0:
        return 0.0, 0.0, 0.0
    
    f_array = np.array(f_values)
    g = compute_autoconvolution(f_array)
    l2_sq, l1, linf = compute_c2_norms(g)
    
    if l1 == 0 or linf == 0:
        return 0.0, l2_sq, l1
    
    c2 = l2_sq / (l1 * linf)
    return c2, l2_sq, l1

def evolutionary_search() -> List[float]:
    """Use evolutionary algorithm to find optimal step function."""
    # Parameters for evolutionary algorithm
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Initialize population with diverse step functions
    population = []
    for _ in range(population_size):
        # Create random step function with varying number of steps
        n_steps = random.randint(50, 500)
        individual = [random.uniform(0, 1) for _ in range(n_steps)]
        population.append(individual)
    
    best_individual = None
    best_c2 = -1
    start_time = time.time()
    
    for gen in range(generations):
        if time.time() - start_time > 55:  # Leave some buffer time
            break
            
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            c2, _, _ = compute_c2(individual)
            fitness_scores.append(c2)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_c2:
            best_c2 = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx][:]
        
        # Selection (tournament selection)
        selected = []
        tournament_size = 3
        for _ in range(population_size):
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx][:])
        
        # Crossover and mutation
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % population_size] if i + 1 < population_size else selected[0]
            
            # Crossover
            if random.random() < crossover_rate and len(parent1) > 1 and len(parent2) > 1:
                crossover_point = min(len(parent1), len(parent2)) // 2
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                child1, child2 = parent1[:], parent2[:]
            
            # Mutation
            for j in range(len(child1)):
                if random.random() < mutation_rate:
                    child1[j] = max(0, min(1, child1[j] + random.gauss(0, 0.1)))
            
            for j in range(len(child2)):
                if random.random() < mutation_rate:
                    child2[j] = max(0, min(1, child2[j] + random.gauss(0, 0.1)))
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
    
    return best_individual if best_individual is not None else []

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value using evolutionary search."""
    # Try different approaches and return the best
    best_result = []
    best_c2 = -1
    
    # Run evolutionary search
    try:
        result = evolutionary_search()
        if result:
            c2, _, _ = compute_c2(result)
            if c2 > best_c2:
                best_c2 = c2
                best_result = result
    except Exception as e:
        pass  # Fall back to simpler approach if evolutionary fails
    
    # Fallback to a better heuristic approach if nothing worked
    if not best_result:
        # Create a more structured approach - try to create something that balances
        # the autoconvolution profile to maximize C2
        n_steps = 200
        # Try creating a pattern that creates a flatter convolution
        # Start with a simple pattern like a bump function
        f_values = []
        half = n_steps // 2
        
        # Create a smooth, symmetric distribution
        for i in range(n_steps):
            # Create a bell-shaped curve for better autoconvolution properties
            x = (i - half) / half
            # Gaussian-like shape
            height = max(0, 1 - x*x)  # Simple parabolic shape
            f_values.append(height)
        
        # Normalize to make it more comparable
        total = sum(f_values)
        if total > 0:
            f_values = [v/total for v in f_values]
        
        best_result = f_values
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
