# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List

def compute_autoconvolution_norms(f_values: List[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    f_values: list of step heights
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    # Create step function on [-1/4, 1/4] with equal spacing
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0
    
    # Define domain for step function
    x_domain = np.linspace(-0.25, 0.25, n_steps + 1)
    dx = x_domain[1] - x_domain[0]
    
    # Create piecewise constant function
    # We'll use the left endpoint values for the step function
    f = np.zeros(n_steps + 1)
    for i in range(n_steps):
        f[i] = f_values[i]
    f[n_steps] = f_values[-1]  # Last point
    
    # Compute autoconvolution g = f * f using discrete convolution
    # Using the fact that f is piecewise constant
    g = signal.convolve(f, f, mode='full')
    
    # Trim g to the appropriate size for our domain
    # The convolution result has length 2*n_steps - 1
    g = g[n_steps-1:2*n_steps-1]
    
    # Compute norms
    # ||g||₂² = sum of squares of g values times dx
    g_squared = g * g
    norm_g2_squared = np.sum(g_squared) * dx
    
    # ||g||₁ = sum of absolute values times dx
    norm_g1 = np.sum(np.abs(g)) * dx
    
    # ||g||∞ = maximum absolute value
    norm_ginf = np.max(np.abs(g))
    
    return norm_g2_squared, norm_g1, norm_ginf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C2 = ||g||₂² / (||g||₁ · ||g||∞)
    """
    norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
    
    # Handle edge cases
    if norm_g1 <= 1e-12 or norm_ginf <= 1e-12:
        return 0.0
    
    c2 = norm_g2_sq / (norm_g1 * norm_ginf)
    return c2

def construct_function() -> List[float]:
    """
    Advanced evolutionary approach to construct step-function with high C2 value.
    Uses a combination of genetic algorithm and local optimization techniques.
    """
    # Initialize parameters
    n_generations = 50
    population_size = 100
    n_steps = 200  # Fixed number of steps for consistency
    
    # Start with a diverse initial population
    population = []
    
    # Generate diverse initial solutions
    for _ in range(population_size):
        # Mix of different patterns: uniform, peaked, sawtooth, etc.
        pattern_type = random.choice(['uniform', 'peaked', 'sawtooth', 'random'])
        
        if pattern_type == 'uniform':
            # Uniform distribution
            individual = [0.5] * n_steps
        elif pattern_type == 'peaked':
            # Peak in center with gradual decay
            individual = []
            for i in range(n_steps):
                x = (i / (n_steps - 1) - 0.5) * 2  # Range from -1 to 1
                # Gaussian-like peak
                individual.append(max(0, 1 - abs(x) * 2))
        elif pattern_type == 'sawtooth':
            # Sawtooth pattern
            individual = []
            for i in range(n_steps):
                individual.append(i / (n_steps - 1))
        else:  # random
            # Random values between 0 and 1
            individual = [random.random() for _ in range(n_steps)]
        
        # Ensure non-negativity
        individual = [max(0, val) for val in individual]
        population.append(individual)
    
    # Evolutionary process
    best_individual = None
    best_c2 = 0.0
    
    for generation in range(n_generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            c2 = evaluate_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_c2:
                best_c2 = c2
                best_individual = individual.copy()
        
        # Selection: keep top 50% 
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        selected_indices = sorted_indices[:population_size // 2]
        selected_population = [population[i] for i in selected_indices]
        
        # Create new population through crossover and mutation
        new_population = selected_population.copy()
        
        # Elitism: keep the best individual
        if best_individual is not None:
            new_population.append(best_individual)
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = random.choice(selected_indices)
            parent2_idx = random.choice(selected_indices)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover: blend two parents
            child = []
            for i in range(n_steps):
                alpha = random.random()
                child_val = alpha * parent1[i] + (1 - alpha) * parent2[i]
                child.append(child_val)
            
            # Mutation: add small random perturbation
            mutation_rate = 0.1
            for i in range(n_steps):
                if random.random() < mutation_rate:
                    # Add small Gaussian noise
                    child[i] += random.gauss(0, 0.1)
                    # Keep non-negative
                    child[i] = max(0, child[i])
            
            new_population.append(child)
        
        population = new_population[:population_size]
    
    # Final optimization with local search around the best solution
    if best_individual is not None:
        # Try to refine the best solution further
        refined_solution = best_individual.copy()
        for _ in range(20):  # Small number of refinement iterations
            # Simple gradient-free local search
            test_solutions = []
            for i in range(n_steps):
                # Try small perturbations
                for delta in [-0.05, -0.01, 0.01, 0.05]:
                    candidate = refined_solution.copy()
                    candidate[i] = max(0, candidate[i] + delta)
                    test_solutions.append(candidate)
            
            # Evaluate and select the best
            best_test = None
            best_test_c2 = 0.0
            for sol in test_solutions:
                c2 = evaluate_c2(sol)
                if c2 > best_test_c2:
                    best_test_c2 = c2
                    best_test = sol
            
            if best_test is not None and best_test_c2 > evaluate_c2(refined_solution):
                refined_solution = best_test
            else:
                break
                
        return refined_solution
    
    # Fallback: return the best individual found
    if best_individual is not None:
        return best_individual
    else:
        # Return simple uniform distribution as fallback
        return [0.5] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
