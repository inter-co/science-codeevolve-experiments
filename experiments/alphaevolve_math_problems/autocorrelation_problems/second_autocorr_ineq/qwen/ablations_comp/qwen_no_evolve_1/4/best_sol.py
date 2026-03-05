# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from typing import List
import random

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f
    # Using convolution with proper normalization
    g = signal.convolve(f, f, mode='full')
    
    # Normalize for proper spacing
    # Since we're working on [-1/4, 1/4] with n points,
    # the spacing is 1/(2*n) for each step
    # But we need to consider the actual convolution domain
    
    # Take only the relevant part of convolution (centered around zero)
    center_idx = len(g) // 2
    g_centered = g[center_idx - len(f) + 1:center_idx + len(f) - 1]
    
    # Compute norms
    g_squared = g_centered ** 2
    g_abs = np.abs(g_centered)
    
    # ||g||₂² using trapezoidal rule approximation
    # For piecewise linear segments, we approximate integral of g²
    # We'll use a simple approach: sum of trapezoidal contributions
    # Each segment contributes (width/3)*(y1^2 + y1*y2 + y2^2)
    # Here we approximate using adjacent pairs
    if len(g_centered) < 2:
        norm_2_sq = 0.0
    else:
        norm_2_sq = 0.0
        # Simple trapezoidal rule for integral of g^2
        for i in range(len(g_centered)-1):
            h = 1.0 / (len(f) - 1)  # Approximate step size
            y1 = g_centered[i]
            y2 = g_centered[i+1]
            norm_2_sq += (h/3) * (y1**2 + y1*y2 + y2**2)
    
    # ||g||₁ = sum of absolute values divided by number of points
    norm_1 = np.sum(g_abs) / (len(g_centered) + 1)
    
    # ||g||∞ = maximum absolute value
    norm_inf = np.max(g_abs)
    
    return norm_2_sq, norm_1, norm_inf

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞)"""
    try:
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2
    except Exception:
        return 0.0

def evolve_step_function() -> List[float]:
    """
    Evolve a step function using a hybrid approach combining:
    1. Genetic algorithm principles (selection, crossover, mutation)
    2. Mathematical insights about optimal structures
    3. Adaptive parameter tuning
    """
    # Initialize parameters
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    elite_count = 5
    
    # Generate initial population
    population = []
    for _ in range(population_size):
        # Create diverse initial step functions
        n_steps = random.randint(100, 500)
        # Start with some structured patterns
        if random.random() < 0.5:
            # Gaussian-like pattern
            x = np.linspace(-0.25, 0.25, n_steps)
            sigma = 0.05 + random.random() * 0.05
            heights = np.exp(-x**2 / (2 * sigma**2)) * (0.5 + random.random() * 0.5)
        else:
            # Uniform or piecewise pattern
            heights = np.random.random(n_steps) * 0.5 + 0.25
        # Ensure non-negative
        heights = np.maximum(heights, 0)
        population.append(list(heights))
    
    best_fitness = 0.0
    best_individual = None
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = calculate_c2(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness
        fitness_scores.sort(reverse=True)
        
        # Track best individual
        current_best = fitness_scores[0][0]
        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = fitness_scores[0][1].copy()
        
        # Selection: keep top performers
        selected = [ind for _, ind in fitness_scores[:elite_count]]
        
        # Create new generation through crossover and mutation
        new_population = selected.copy()
        
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(fitness_scores, 3)
            parent2 = tournament_selection(fitness_scores, 3)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
        
        # Adaptive mutation rate
        if gen % 20 == 0 and gen > 0:
            mutation_rate = max(0.01, mutation_rate * 0.9)
    
    return best_individual if best_individual is not None else []

def tournament_selection(fitness_scores, k):
    """Select individual using tournament selection"""
    tournament = random.sample(fitness_scores, min(k, len(fitness_scores)))
    winner = max(tournament, key=lambda x: x[0])
    return winner[1]

def crossover(parent1, parent2):
    """Perform uniform crossover between two parents"""
    # Make sure they have same length
    min_len = min(len(parent1), len(parent2))
    parent1 = parent1[:min_len]
    parent2 = parent2[:min_len]
    
    # Uniform crossover
    child = []
    for i in range(min_len):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    return child

def mutate(individual, mutation_rate):
    """Apply mutation to an individual"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add small random perturbation
            delta = random.gauss(0, 0.1 * max(1e-6, abs(mutated[i])))
            mutated[i] = max(0, mutated[i] + delta)  # Ensure non-negative
    
    return mutated

def construct_function() -> List[float]:
    """Main function to construct step-function with high C2 value"""
    # Use evolutionary approach to find optimal step function
    return evolve_step_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
