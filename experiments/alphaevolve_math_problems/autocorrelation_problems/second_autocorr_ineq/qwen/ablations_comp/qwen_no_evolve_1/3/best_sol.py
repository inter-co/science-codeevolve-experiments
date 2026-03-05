# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    f_values: step heights (non-negative)
    Returns: (||g||₂², ||g||₁, ||g||∞) where g = f*f
    """
    if not f_values:
        return 0.0, 0.0, 0.0
    
    # Normalize step function to have domain [-1/4, 1/4]
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Create step function with equal spacing on [-1/4, 1/4]
    dx = 0.5 / n  # width of each step
    x_f = np.linspace(-0.25 + dx/2, 0.25 - dx/2, n)  # centers of steps
    
    # Convert to piecewise constant function for convolution
    # We'll compute autoconvolution using discrete convolution
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using FFT for efficiency
    # Pad to appropriate size for proper convolution
    pad_size = 2 * n - 1
    f_padded = np.pad(f, (0, pad_size - n), mode='constant')
    
    # Use FFT-based convolution
    g_fft = np.fft.fft(f_padded) * np.conj(np.fft.fft(f_padded))
    g = np.fft.ifft(g_fft).real[:pad_size]
    
    # The autoconvolution is symmetric and centered, so we take the first half
    # But actually, let's compute it properly with correct indexing
    g = signal.convolve(f, f, mode='full')
    
    # Compute norms
    g_squared = g * g
    norm_g2_squared = np.sum(g_squared) * dx  # integrate over the full domain
    
    norm_g1 = np.sum(np.abs(g)) * dx  # integrate over the full domain
    norm_ginf = np.max(np.abs(g))
    
    return norm_g2_squared, norm_g1, norm_ginf

def evaluate_c2(f_values: list[float]) -> float:
    """Evaluate C2 for given step function values"""
    try:
        norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_g1 <= 1e-12 or norm_ginf <= 1e-12:
            return 0.0
            
        c2 = norm_g2_sq / (norm_g1 * norm_ginf)
        return c2
    except Exception:
        return 0.0

def construct_function() -> list[float]:
    """
    Use evolutionary optimization to find step function with high C2 value.
    This implements a novel approach using genetic algorithms and local refinement.
    """
    # Initialize parameters
    population_size = 50
    generations = 100
    num_steps = 200  # Fixed number of steps for consistency
    
    # Generate initial population
    population = []
    for _ in range(population_size):
        # Start with some structured patterns rather than pure randomness
        individual = []
        for i in range(num_steps):
            # Use a combination of random and structured approaches
            if i < num_steps // 4:
                # Early steps: higher values
                individual.append(random.uniform(0.5, 1.0))
            elif i < num_steps // 2:
                # Middle steps: moderate values  
                individual.append(random.uniform(0.2, 0.8))
            else:
                # Later steps: lower values
                individual.append(random.uniform(0.0, 0.5))
        population.append(individual)
    
    # Evolutionary process
    best_individual = None
    best_c2 = 0.0
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            # Ensure non-negativity
            individual = [max(0.0, val) for val in individual]
            c2 = evaluate_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_c2:
                best_c2 = c2
                best_individual = individual.copy()
        
        # Selection (tournament selection)
        selected = []
        for _ in range(population_size):
            tournament_indices = random.sample(range(population_size), 3)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        # Crossover and mutation
        next_generation = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % population_size]
            
            # Uniform crossover
            child1 = []
            child2 = []
            for j in range(len(parent1)):
                if random.random() < 0.5:
                    child1.append(parent1[j])
                    child2.append(parent2[j])
                else:
                    child1.append(parent2[j])
                    child2.append(parent1[j])
            
            # Mutation
            for j in range(len(child1)):
                if random.random() < 0.1:  # 10% mutation rate
                    child1[j] = max(0.0, child1[j] + random.gauss(0, 0.1))
                if random.random() < 0.1:
                    child2[j] = max(0.0, child2[j] + random.gauss(0, 0.1))
            
            next_generation.extend([child1, child2])
        
        population = next_generation[:population_size]
    
    # Final refinement with gradient-based approach if needed
    if best_individual is not None:
        # Apply local search around best solution
        refined = best_individual.copy()
        for _ in range(50):  # Local iterations
            # Small perturbations
            idx = random.randint(0, len(refined) - 1)
            delta = random.gauss(0, 0.05)
            refined[idx] = max(0.0, refined[idx] + delta)
            
            # Check improvement
            temp_c2 = evaluate_c2(refined)
            if temp_c2 > best_c2:
                best_c2 = temp_c2
            else:
                # Revert if no improvement
                refined[idx] = best_individual[idx]
        
        return refined
    else:
        # Fallback to simple structured approach
        return [1.0] * num_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
