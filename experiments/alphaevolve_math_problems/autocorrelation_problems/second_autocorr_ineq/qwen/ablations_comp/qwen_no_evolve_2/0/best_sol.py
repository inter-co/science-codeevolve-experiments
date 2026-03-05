# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import random
from scipy import signal
from scipy.optimize import differential_evolution
import time

def compute_autoconvolution_norms(f_values: List[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Create step function (assuming uniform spacing on [-1/4, 1/4])
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0
    
    # Step size
    dx = 0.5 / (n_steps - 1) if n_steps > 1 else 1.0
    
    # Compute autoconvolution using discrete convolution
    # We need to handle the fact that our function is defined on [-1/4, 1/4]
    # But convolution is typically done on symmetric intervals
    g = signal.convolve(f_values, f_values, mode='full')
    
    # Adjust indices for correct domain mapping
    # Original function spans [-1/4, 1/4], so autoconvolution spans [-1/2, 1/2]
    # We're interested in the middle portion [-1/4, 1/4] which corresponds to
    # the central part of the convolution
    mid_idx = len(g) // 2
    half_len = len(f_values)
    start_idx = mid_idx - half_len + 1
    end_idx = mid_idx + half_len
    
    g_values = g[start_idx:end_idx]
    
    # Compute norms
    # ||g||₂² = sum(g[i]² * dx) for piecewise linear integration
    # Using trapezoidal rule for integral approximation
    g_squared = np.array(g_values)**2
    g_abs = np.abs(np.array(g_values))
    
    # For L2 norm squared using trapezoidal integration
    # We approximate integral of g^2 using piecewise linear segments
    if len(g_values) <= 1:
        l2_norm_sq = 0.0
    else:
        # Trapezoidal integration for g^2
        # For piecewise linear segments, we compute integral as:
        # sum_{i=0}^{n-2} (dx/3) * (g[i]^2 + g[i]*g[i+1] + g[i+1]^2)
        l2_norm_sq = 0.0
        for i in range(len(g_values) - 1):
            l2_norm_sq += (dx/3) * (g_values[i]**2 + g_values[i]*g_values[i+1] + g_values[i+1]**2)
    
    # ||g||₁ = sum(|g|) * dx
    l1_norm = np.sum(g_abs) * dx
    
    # ||g||∞ = max(|g|)
    l_inf_norm = np.max(g_abs) if len(g_abs) > 0 else 0.0
    
    return l2_norm_sq, l1_norm, l_inf_norm

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞)"""
    l2_sq, l1, l_inf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if l1 <= 1e-15 or l_inf <= 1e-15:
        return 0.0
    
    return l2_sq / (l1 * l_inf)

def create_individual(n_steps: int) -> List[float]:
    """Create a random individual (step function)"""
    return [random.uniform(0, 1) for _ in range(n_steps)]

def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate an individual with specialized operators"""
    mutated = individual.copy()
    
    # Apply Gaussian mutations to some elements
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add normally distributed noise
            mutated[i] += random.gauss(0, 0.1)
            # Ensure non-negativity
            mutated[i] = max(0, mutated[i])
    
    # Occasionally apply a global scaling operation
    if random.random() < 0.1:
        scale_factor = random.uniform(0.5, 2.0)
        mutated = [max(0, x * scale_factor) for x in mutated]
    
    # Occasionally apply a smoothing operation
    if random.random() < 0.05 and len(mutated) > 2:
        # Simple moving average smoothing
        smoothed = []
        for i in range(len(mutated)):
            neighbors = []
            for j in range(max(0, i-1), min(len(mutated), i+2)):
                neighbors.append(mutated[j])
            smoothed.append(sum(neighbors)/len(neighbors))
        mutated = smoothed
    
    return mutated

def crossover_individuals(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform crossover between two individuals"""
    if len(parent1) != len(parent2):
        # If lengths differ, use the shorter one for crossover
        min_len = min(len(parent1), len(parent2))
        parent1 = parent1[:min_len]
        parent2 = parent2[:min_len]
    
    # Uniform crossover
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    return child

def evolve_step_function() -> List[float]:
    """Evolve a step function to maximize C2 using evolutionary algorithm"""
    # Parameters
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    elite_size = 5
    
    # Initialize population
    population = [create_individual(random.randint(50, 200)) for _ in range(population_size)]
    
    best_fitness = -float('inf')
    best_individual = None
    
    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = calculate_c2(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness
        fitness_scores.sort(reverse=True)
        
        # Track best individual
        current_best_fitness, current_best = fitness_scores[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best.copy()
        
        # Selection: keep top individuals
        elite = [ind for _, ind in fitness_scores[:elite_size]]
        
        # Generate new population
        new_population = elite.copy()
        
        # Fill rest with offspring
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament = random.sample(fitness_scores, tournament_size)
            winner = max(tournament, key=lambda x: x[0])[1]
            
            # Another tournament for second parent
            tournament = random.sample(fitness_scores, tournament_size)
            parent2 = max(tournament, key=lambda x: x[0])[1]
            
            # Crossover
            child = crossover_individuals(winner, parent2)
            
            # Mutation
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    return best_individual if best_individual is not None else []

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Use evolutionary algorithm to find optimal step function
    try:
        return evolve_step_function()
    except Exception as e:
        # Fallback to simple random construction if evolution fails
        return [random.random() for _ in range(random.randint(100, 500))]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
