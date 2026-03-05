# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation.
    f_values: list of step heights
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    # Convert to numpy array for easier handling
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f (using discrete convolution)
    g = signal.convolve(f, f, mode='full')
    
    # Trim to appropriate size (we only need the middle portion)
    # Since f has len(f) elements, g will have 2*len(f)-1 elements
    # We want the central part corresponding to valid convolutions
    center = len(g) // 2
    g_centered = g[center - len(f) + 1:center + len(f) - 1]
    
    # Compute the three norms
    # ||g||₂² = sum(g[i]²) for discrete case
    g_squared = g_centered ** 2
    norm_g_2_squared = np.sum(g_squared)
    
    # ||g||₁ = sum(|g[i]|) 
    norm_g_1 = np.sum(np.abs(g_centered))
    
    # ||g||∞ = max(|g[i]|)
    norm_g_inf = np.max(np.abs(g_centered))
    
    return norm_g_2_squared, norm_g_1, norm_g_inf

def calculate_c2(f_values):
    """Calculate C2 for given step function values"""
    try:
        norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_g_1 <= 1e-12 or norm_g_inf <= 1e-12:
            return 0.0
            
        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception:
        return 0.0

def evolve_step_function():
    """
    Evolve a step function using a hybrid approach combining:
    1. Genetic algorithm for global exploration
    2. Local optimization for refinement
    """
    # Parameters
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    
    # Initialize population with diverse step functions
    def create_individual(length):
        # Create a mix of peaks and valleys to encourage good convolution properties
        individual = []
        for _ in range(length):
            # Use a combination of different probability distributions
            r = random.random()
            if r < 0.3:
                # High peak
                individual.append(random.uniform(0.5, 1.0))
            elif r < 0.6:
                # Medium value
                individual.append(random.uniform(0.1, 0.5))
            else:
                # Low value or zero
                individual.append(random.uniform(0.0, 0.3))
        return individual
    
    # Fitness function (we want to maximize C2)
    def fitness(individual):
        c2 = calculate_c2(individual)
        return c2
    
    # Create initial population
    population = []
    for _ in range(population_size):
        length = random.randint(100, 500)  # Variable length
        individual = create_individual(length)
        population.append(individual)
    
    # Evolution loop
    best_individual = None
    best_fitness = -1
    
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = [fitness(ind) for ind in population]
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Selection (tournament selection)
        def tournament_selection():
            k = 3
            candidates = random.sample(list(zip(population, fitness_scores)), k)
            return max(candidates, key=lambda x: x[1])[0]
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Keep best individual (elitism)
        new_population.append(best_individual.copy())
        
        while len(new_population) < population_size:
            parent1 = tournament_selection()
            parent2 = tournament_selection()
            
            # Crossover (uniform)
            child = []
            for i in range(min(len(parent1), len(parent2))):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            
            # Mutation
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    # Add small random change
                    child[i] += random.gauss(0, 0.1)
                    child[i] = max(0, child[i])  # Ensure non-negative
            
            # Adjust length randomly
            if random.random() < 0.1:
                if len(child) > 50:
                    child = child[:len(child)//2]
                else:
                    # Add some elements
                    for _ in range(random.randint(1, 10)):
                        child.append(random.uniform(0, 1))
            
            new_population.append(child)
        
        population = new_population
    
    # Final refinement using local optimization
    if best_individual is not None:
        # Try to refine further with simple gradient-based approach
        refined = best_individual.copy()
        # Simple hill climbing: slightly adjust values
        for _ in range(50):
            idx = random.randint(0, len(refined) - 1)
            old_val = refined[idx]
            # Try small perturbations
            new_val = old_val + random.uniform(-0.1, 0.1)
            new_val = max(0, new_val)
            refined[idx] = new_val
            # Keep if improvement
            if calculate_c2(refined) > calculate_c2(best_individual):
                best_individual = refined.copy()
    
    return best_individual

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Use the evolved approach
    try:
        return evolve_step_function()
    except:
        # Fallback to a reasonable default
        return [0.5] * 200

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
