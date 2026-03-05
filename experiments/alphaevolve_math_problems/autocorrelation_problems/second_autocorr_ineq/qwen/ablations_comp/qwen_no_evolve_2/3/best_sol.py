# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Create step function on [-1/4, 1/4] with given heights
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0
    
    # Step width
    step_width = 0.5 / n_steps
    
    # Compute autoconvolution g = f * f using discrete convolution
    # f is represented as step function with n_steps
    # We'll use the fact that autoconvolution of step functions has specific structure
    
    # Create the step function as a piecewise constant function
    # For simplicity, we'll work with the discrete representation
    f_array = np.array(f_values)
    
    # Compute autoconvolution using discrete convolution
    # This gives us the convolution of the step function with itself
    g = np.convolve(f_array, f_array, mode='full')
    
    # Adjust indices to match the actual support [-1/4, 1/4] 
    # The convolution result spans from -2*step_width to 2*step_width
    # But we're interested in the central part [-1/4, 1/4]
    
    # The full convolution result has length 2*n_steps - 1
    # We want to extract the portion corresponding to [-1/4, 1/4]
    # Which is roughly the middle portion
    
    # Actually, let's reconsider: we're doing convolution of step functions
    # Let's use proper numerical integration approach
    
    # For accurate computation, let's compute the continuous convolution properly
    # But since we're working with discrete steps, we'll use the discrete approach
    # and apply proper normalization
    
    # Normalize g so that it corresponds to correct interval [-1/4, 1/4]
    # Width of domain is 0.5, so we multiply by step_width for proper scaling
    g = g * step_width
    
    # Compute norms
    g_squared = g**2
    norm_g2_squared = np.sum(g_squared) * step_width  # Approximate integral
    
    # L1 norm
    norm_g1 = np.sum(np.abs(g)) * step_width
    
    # L-infinity norm
    norm_g_infty = np.max(np.abs(g))
    
    return norm_g2_squared, norm_g1, norm_g_infty

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C2 = ||g||₂² / (||g||₁ · ||g||∞)
    """
    try:
        norm_g2_squared, norm_g1, norm_g_infty = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_g1 <= 1e-15 or norm_g_infty <= 1e-15:
            return 0.0
            
        c2 = norm_g2_squared / (norm_g1 * norm_g_infty)
        return c2
    except Exception:
        return 0.0

def construct_function() -> List[float]:
    """
    Improved evolutionary approach to construct step-function with high C2 value.
    Uses a hybrid of genetic algorithm and local search.
    """
    # Parameters
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    elite_size = 5
    
    # Initialize population with diverse step functions
    def create_individual():
        # Random number of steps between 100 and 500
        n_steps = random.randint(100, 500)
        # Create random heights, but bias towards higher values
        return [random.uniform(0, 1) for _ in range(n_steps)]
    
    # Create initial population
    population = [create_individual() for _ in range(population_size)]
    
    best_fitness = 0.0
    best_individual = []
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_c2(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Track best
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_individual = fitness_scores[0][1][:]  # Make copy
        
        # Selection: keep top individuals as elites
        elites = [ind for _, ind in fitness_scores[:elite_size]]
        
        # Create new population through crossover and mutation
        new_population = elites[:]
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(fitness_scores)
            parent2 = tournament_selection(fitness_scores)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Occasionally introduce some diversity
        if gen % 20 == 0 and gen > 0:
            for i in range(5):  # Add 5 new random individuals
                population[random.randint(0, population_size-1)] = create_individual()
    
    # Return the best individual found
    return best_individual

def tournament_selection(fitness_scores: List[Tuple[float, List[float]]], k: int = 3) -> List[float]:
    """Select an individual using tournament selection"""
    tournament = random.sample(fitness_scores, min(k, len(fitness_scores)))
    tournament.sort(key=lambda x: x[0], reverse=True)
    return tournament[0][1]

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Single-point crossover between two individuals"""
    if len(parent1) == 0 or len(parent2) == 0:
        return parent1 if len(parent1) > 0 else parent2
    
    # Choose crossover point
    min_len = min(len(parent1), len(parent2))
    if min_len == 0:
        return parent1 if len(parent1) > 0 else parent2
    
    crossover_point = random.randint(1, min_len - 1)
    
    # Create child
    child = parent1[:crossover_point] + parent2[crossover_point:]
    
    # Adjust length to match one of parents (with preference to longer)
    if len(child) != len(parent1) and len(child) != len(parent2):
        if len(parent1) >= len(parent2):
            child = child[:len(parent1)]
        else:
            child = child[:len(parent2)]
    
    return child

def mutate(individual: List[float], mutation_rate: float) -> List[float]:
    """Mutate an individual"""
    mutated = individual[:]
    
    # For each element, decide whether to mutate
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate by adding small random change
            mutated[i] += random.gauss(0, 0.1)
            # Ensure non-negative
            mutated[i] = max(0, mutated[i])
    
    # Occasionally change the number of steps
    if random.random() < 0.1 and len(mutated) > 10:
        # Slightly change the size
        new_size = max(10, int(len(mutated) * random.uniform(0.8, 1.2)))
        if new_size != len(mutated):
            if new_size > len(mutated):
                # Extend with zeros or last values
                mutated.extend([mutated[-1]] * (new_size - len(mutated)))
            else:
                # Truncate
                mutated = mutated[:new_size]
    
    return mutated

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
