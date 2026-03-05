# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    f_values: step function heights
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    # Create step function on [-1/4, 1/4] with given heights
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Define domain
    x_domain = np.linspace(-0.25, 0.25, 2*n+1)  # Points for piecewise linear
    dx = x_domain[1] - x_domain[0]
    
    # Create step function - piecewise constant
    f = np.zeros(len(x_domain))
    step_width = 0.5 / n
    for i in range(n):
        start_idx = int(i * 2 + 1)  # Start index for this step
        end_idx = int((i + 1) * 2 + 1)  # End index for this step  
        f[start_idx:end_idx] = f_values[i]
    
    # Compute autoconvolution g = f * f
    # Using discrete convolution
    g = signal.convolve(f, f, mode='full')
    
    # Adjust indices to match [-0.5, 0.5] domain
    g_len = len(g)
    g_domain = np.linspace(-0.5, 0.5, g_len)
    
    # For our purposes, we only care about the middle portion [-0.25, 0.25]
    center_idx = g_len // 2
    half_len = n
    g_middle = g[center_idx-half_len:center_idx+half_len]
    
    # Compute norms
    # ||g||₂² = sum(g_i² * dx) - using trapezoidal rule approximation
    g_squared = g_middle ** 2
    # Trapezoidal rule for ∫g²dx: (dx/3) * (y₀² + y₀y₁ + y₁² + ... + yₙ₋₁yₙ + yₙ²)
    if len(g_middle) >= 2:
        # For piecewise linear integration with trapezoidal-like approach
        # We'll compute weighted sum of squares for adjacent pairs
        g_2_norm_sq = 0.0
        for i in range(len(g_middle)):
            if i == 0:
                g_2_norm_sq += g_middle[i]**2
            elif i == len(g_middle) - 1:
                g_2_norm_sq += g_middle[i]**2
            else:
                # Average of adjacent values times their product
                g_2_norm_sq += g_middle[i]**2
        g_2_norm_sq *= dx/3  # Simplified trapezoidal weight
    else:
        g_2_norm_sq = g_middle[0]**2 * dx if len(g_middle) > 0 else 0.0
    
    # ||g||₁ = sum(|g|) * dx
    g_1_norm = np.sum(np.abs(g_middle)) * dx
    
    # ||g||∞ = max(|g|)
    g_inf_norm = np.max(np.abs(g_middle))
    
    return g_2_norm_sq, g_1_norm, g_inf_norm

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using evolutionary approach."""
    
    # Initialize parameters
    n_steps = 100  # Number of steps
    max_iterations = 1000
    population_size = 50
    
    # Generate initial population of step functions
    population = []
    for _ in range(population_size):
        # Create random step heights, but ensure some structure
        heights = [random.uniform(0, 1) for _ in range(n_steps)]
        # Add some smoothing to avoid extreme peaks
        smoothed_heights = []
        for i in range(len(heights)):
            # Apply local averaging to create smoother steps
            total = heights[i]
            count = 1
            if i > 0:
                total += heights[i-1]
                count += 1
            if i < len(heights) - 1:
                total += heights[i+1]
                count += 1
            smoothed_heights.append(total / count)
        population.append(smoothed_heights)
    
    best_c2 = 0.0
    best_individual = None
    
    # Evolutionary search
    for generation in range(max_iterations):
        # Evaluate fitness (C2 values)
        fitness_scores = []
        for individual in population:
            try:
                g_2_sq, g_1, g_inf = compute_autoconvolution_norms(individual)
                if g_1 > 0 and g_inf > 0:
                    c2 = g_2_sq / (g_1 * g_inf)
                else:
                    c2 = 0.0
                fitness_scores.append(c2)
            except Exception:
                fitness_scores.append(0.0)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_c2:
            best_c2 = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx][:]
        
        # Selection (tournament selection)
        selected = []
        for _ in range(population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(population_size, tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx][:])
        
        # Crossover and mutation
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i+1) % population_size]
            
            # Single-point crossover
            crossover_point = random.randint(1, len(parent1)-1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            
            # Mutation (Gaussian noise)
            mutation_rate = 0.1
            for j in range(len(child1)):
                if random.random() < mutation_rate:
                    child1[j] = max(0, child1[j] + random.gauss(0, 0.1))
            for j in range(len(child2)):
                if random.random() < mutation_rate:
                    child2[j] = max(0, child2[j] + random.gauss(0, 0.1))
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
        
        # Occasionally introduce diversity
        if generation % 50 == 0 and generation > 0:
            for i in range(0, min(5, len(population)), 2):
                population[i] = [random.uniform(0, 1) for _ in range(n_steps)]
    
    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback to simple structured function
        return [1.0] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
