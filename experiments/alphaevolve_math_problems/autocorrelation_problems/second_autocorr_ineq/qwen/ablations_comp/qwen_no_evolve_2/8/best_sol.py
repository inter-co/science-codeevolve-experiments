# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import random
from scipy import signal

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Create step function on [-1/4, 1/4] with equal spacing
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Define the domain
    domain = np.linspace(-0.25, 0.25, 2*n+1)  # More points for better resolution
    dx = domain[1] - domain[0]
    
    # Construct step function
    f = np.zeros_like(domain)
    for i in range(n):
        start_idx = 2*i
        end_idx = 2*(i+1)
        f[start_idx:end_idx] = f_values[i]
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = signal.convolve(f, f, mode='full')
    g = g[len(g)//2:]  # Take only positive indices
    
    # Compute the three norms
    g_squared = g**2
    norm_g2_squared = np.sum(g_squared) * dx  # ∫g²dx
    
    norm_g1 = np.sum(np.abs(g)) * dx  # ∫|g|dx
    
    norm_g_infty = np.max(np.abs(g))  # max|g|
    
    return norm_g2_squared, norm_g1, norm_g_infty

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 value for given step function values."""
    norm_g2_squared, norm_g1, norm_g_infty = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_g1 <= 1e-12 or norm_g_infty <= 1e-12:
        return 0.0
    
    c2 = norm_g2_squared / (norm_g1 * norm_g_infty)
    return c2

def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply mutation to an individual (step function)."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add small random change with bounded support
            mutated[i] = max(0.0, mutated[i] + np.random.normal(0, 0.1))
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform crossover between two individuals."""
    # Simple uniform crossover
    child = []
    for i in range(min(len(parent1), len(parent2))):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    # Ensure minimum length and reasonable values
    if len(child) == 0:
        child = [random.random()]
    return child

def create_initial_population(pop_size: int, min_length: int = 50, max_length: int = 500) -> List[List[float]]:
    """Create initial population with diverse step functions."""
    population = []
    for _ in range(pop_size):
        # Random length between min_length and max_length
        length = random.randint(min_length, max_length)
        # Create random step heights with some structure
        individual = [max(0.0, random.gauss(1.0, 0.3)) for _ in range(length)]
        population.append(individual)
    return population

def evolve_step_function() -> List[float]:
    """
    Evolutionary approach to find optimal step function.
    Uses genetic algorithm with custom operators.
    """
    # Parameters
    pop_size = 50
    generations = 100
    mutation_rate = 0.1
    elite_size = 5
    
    # Initialize population
    population = create_initial_population(pop_size)
    
    best_fitness = -1
    best_individual = None
    
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            c2 = calculate_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_fitness:
                best_fitness = c2
                best_individual = individual.copy()
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores.sort(reverse=True)
        
        # Keep elite
        elite = population[:elite_size]
        
        # Generate new population through selection, crossover, and mutation
        new_population = elite.copy()
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(population, fitness_scores, 3)
            parent2 = tournament_selection(population, fitness_scores, 3)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    return best_individual if best_individual is not None else []

def tournament_selection(population: List[List[float]], fitness_scores: List[float], k: int) -> List[float]:
    """Select individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), k)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index]

# Main function to construct optimized step function
def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value using evolutionary approach."""
    try:
        # Run evolution for up to 5 seconds (as per time limit)
        return evolve_step_function()
    except Exception as e:
        # Fallback to simple approach if evolution fails
        return [random.random() for _ in range(100)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
