# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List, Tuple
import time

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Create step function
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Define step positions (equally spaced on [-1/4, 1/4])
    step_width = 0.5 / n
    positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n)
    
    # Create piecewise constant function representation
    # Convolve with itself to get autoconvolution
    # Using discrete convolution with proper normalization
    g = signal.convolve(f_values, f_values, mode='full')
    
    # Adjust for proper spacing
    g = g / (step_width * 2)  # Normalize by step width
    
    # Trim to appropriate size (should be 2*n-1)
    g = g[n-1:2*n-1]
    
    # Compute norms
    g_squared = g * g
    norm_g2_squared = np.sum(g_squared) * step_width  # Approximate integral
    
    norm_g1 = np.sum(np.abs(g)) * step_width
    norm_ginf = np.max(np.abs(g))
    
    return norm_g2_squared, norm_g1, norm_ginf

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 value for given step function."""
    norm_g2_squared, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_g1 <= 1e-15 or norm_ginf <= 1e-15:
        return 0.0
    
    c2 = norm_g2_squared / (norm_g1 * norm_ginf)
    return c2

def create_individual(length: int) -> List[float]:
    """Create a random individual (step function)."""
    return [random.uniform(0, 1) for _ in range(length)]

def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate an individual with specialized operators."""
    mutated = individual.copy()
    
    # Apply Gaussian mutations with adaptive rate
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add normally distributed noise
            noise = random.gauss(0, 0.1)
            mutated[i] = max(0, mutated[i] + noise)  # Ensure non-negative
            
    return mutated

def crossover_individuals(parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
    """Perform specialized crossover for step functions."""
    # Uniform crossover with some smoothing
    child1, child2 = parent1.copy(), parent2.copy()
    
    # Randomly select crossover points
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    
    # Swap segments
    child1[crossover_point:] = parent2[crossover_point:]
    child2[crossover_point:] = parent1[crossover_point:]
    
    # Apply slight smoothing to maintain reasonable gradients
    for i in range(len(child1)):
        if i > 0 and i < len(child1) - 1:
            avg_neighbors = (child1[i-1] + child1[i+1]) / 2
            child1[i] = 0.7 * child1[i] + 0.3 * avg_neighbors
    
    return child1, child2

def evolve_step_function() -> List[float]:
    """Evolve step function using genetic algorithm with specialized operators."""
    # Parameters
    pop_size = 50
    generations = 100
    elite_size = 5
    mutation_rate = 0.1
    tournament_size = 3
    
    # Initialize population
    population = [create_individual(random.randint(50, 500)) for _ in range(pop_size)]
    
    best_fitness = 0
    best_individual = None
    
    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [(calculate_c2(ind), ind) for ind in population]
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Track best individual
        current_best_fitness, current_best = fitness_scores[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best.copy()
        
        # Selection (tournament selection)
        selected = []
        for _ in range(pop_size):
            tournament = random.sample(fitness_scores, tournament_size)
            winner = max(tournament, key=lambda x: x[0])
            selected.append(winner[1].copy())
        
        # Elitism - keep best individuals
        elite = [ind for _, ind in fitness_scores[:elite_size]]
        selected[:elite_size] = elite
        
        # Crossover and mutation
        new_population = elite.copy()
        
        while len(new_population) < pop_size:
            # Select parents
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Crossover
            child1, child2 = crossover_individuals(parent1, parent2)
            
            # Mutate
            child1 = mutate_individual(child1, mutation_rate)
            child2 = mutate_individual(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:pop_size]
    
    return best_individual if best_individual else create_individual(100)

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Try multiple evolution attempts to find better solutions
    best_result = None
    best_c2 = 0
    
    for attempt in range(3):
        try:
            individual = evolve_step_function()
            c2 = calculate_c2(individual)
            
            if c2 > best_c2:
                best_c2 = c2
                best_result = individual.copy()
                
        except Exception as e:
            continue
    
    # Return the best found solution
    if best_result is not None:
        return best_result
    else:
        # Fallback to simple approach if evolution fails
        return [random.uniform(0, 1) for _ in range(100)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
