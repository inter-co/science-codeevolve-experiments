# EVOLVE-BLOCK-START

import numpy as np
from typing import List, Tuple
import random
from scipy.signal import convolve
import time

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array for easier handling
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f
    # Using full convolution to get proper autoconvolution
    g = convolve(f, f, mode='full')
    
    # Take only the central portion (valid convolution)
    center = len(g) // 2
    half_len = len(f) - 1
    g_valid = g[center - half_len:center + half_len + 1]
    
    # Compute norms
    g_squared = g_valid ** 2
    g_abs = np.abs(g_valid)
    
    # ||g||₂² = sum of squares
    norm_2_sq = np.sum(g_squared)
    
    # ||g||₁ = sum of absolute values  
    norm_1 = np.sum(g_abs)
    
    # ||g||∞ = maximum absolute value
    norm_inf = np.max(g_abs)
    
    return norm_2_sq, norm_1, norm_inf

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 value for given step function"""
    try:
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2
    except Exception:
        return 0.0

def generate_individual(length: int) -> List[float]:
    """Generate a random individual (step function)"""
    # Use a more sophisticated approach than pure random
    # Start with a base pattern and add variation
    base_pattern = []
    for _ in range(length):
        # Generate values with some correlation to encourage better solutions
        if random.random() < 0.3:  # Some chance of higher values
            base_pattern.append(random.uniform(0.5, 1.0))
        else:
            base_pattern.append(random.uniform(0.0, 0.5))
    
    # Add some smoothing to make it more likely to produce good results
    smoothed = []
    for i in range(len(base_pattern)):
        # Apply local averaging to create smoother transitions
        window_size = min(3, len(base_pattern))
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(base_pattern), i + window_size//2 + 1)
        avg_val = sum(base_pattern[start_idx:end_idx]) / (end_idx - start_idx)
        smoothed.append(avg_val)
    
    return smoothed

def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate an individual with domain-specific knowledge"""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply different types of mutations based on position
            if i < len(mutated) // 4 or i > 3 * len(mutated) // 4:
                # Boundary regions: prefer larger values for better peaks
                mutated[i] = max(0.0, mutated[i] + random.gauss(0, 0.1))
            elif i < len(mutated) // 2:
                # Left side: encourage more structure
                mutated[i] = max(0.0, mutated[i] + random.gauss(0, 0.05))
            else:
                # Right side: similar to left but with different variance
                mutated[i] = max(0.0, mutated[i] + random.gauss(0, 0.05))
            
            # Ensure values don't go too high to maintain numerical stability
            mutated[i] = min(1.0, mutated[i])
    
    return mutated

def crossover_individuals(parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
    """Crossover operation that preserves structure"""
    if len(parent1) != len(parent2):
        # If lengths differ, truncate to shorter length
        min_len = min(len(parent1), len(parent2))
        parent1 = parent1[:min_len]
        parent2 = parent2[:min_len]
    
    # Create offspring using a blend crossover approach
    child1 = []
    child2 = []
    
    for i in range(len(parent1)):
        # Blend the two parents with some probability
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    
    return child1, child2

def evolve_step_function(max_time: float = 60.0) -> List[float]:
    """Main evolutionary algorithm to find optimal step function"""
    start_time = time.time()
    
    # Parameters for evolution
    pop_size = 50
    generations = 100
    elite_size = 5
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        # Use varying sizes to explore different configurations
        size = random.randint(100, 1000)
        individual = generate_individual(size)
        population.append(individual)
    
    best_fitness = 0.0
    best_individual = None
    
    # Evolution loop
    for gen in range(generations):
        if time.time() - start_time > max_time:
            break
            
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = calculate_c2(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness
        fitness_scores.sort(reverse=True)
        
        # Track best
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_individual = fitness_scores[0][1].copy()
        
        # Selection - keep top individuals
        elites = [ind for _, ind in fitness_scores[:elite_size]]
        
        # Create new population
        new_population = elites.copy()
        
        # Generate offspring through crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            tournament_size = 3
            tournament1 = random.sample(fitness_scores, tournament_size)
            tournament2 = random.sample(fitness_scores, tournament_size)
            
            parent1 = max(tournament1, key=lambda x: x[0])[1]
            parent2 = max(tournament2, key=lambda x: x[0])[1]
            
            # Crossover
            child1, child2 = crossover_individuals(parent1, parent2)
            
            # Mutation
            child1 = mutate_individual(child1)
            child2 = mutate_individual(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:pop_size]
    
    return best_individual if best_individual is not None else generate_individual(500)

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Use evolutionary algorithm to find optimal configuration
    return evolve_step_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
