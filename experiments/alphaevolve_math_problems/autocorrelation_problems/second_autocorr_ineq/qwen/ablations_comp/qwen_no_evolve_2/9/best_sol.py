# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from typing import List
import random
import time

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    f_values: step heights on [-1/4, 1/4]
    """
    if len(f_values) == 0:
        return 0, 0, 0
    
    # Create step function on [-1/4, 1/4] with equal spacing
    n_steps = len(f_values)
    x = np.linspace(-0.25, 0.25, n_steps + 1)
    dx = x[1] - x[0]
    
    # Convert to piecewise constant function
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using discrete convolution
    # We use full convolution then trim appropriately
    g = np.convolve(f, f, mode='full')
    
    # The convolution result has 2*n_steps - 1 elements
    # The valid range is from -n_steps+1 to n_steps-1
    # But we're interested in the central part where the convolution makes sense
    # Let's map back to our domain [-0.5, 0.5] with appropriate scaling
    
    # Actually, let's recompute properly using the correct domain mapping
    # Original f is defined on [-0.25, 0.25] with n_steps intervals
    # So step size is 0.5/n_steps = 0.5/n_steps
    step_size = 0.5 / n_steps
    
    # For autoconvolution, the result g will be defined on [-0.5, 0.5] 
    # with step size = 2 * step_size = 1/n_steps
    g_indices = np.arange(-(n_steps-1), n_steps)
    g_x = g_indices * step_size  # Positions where g is evaluated
    
    # Now compute the norms properly
    # g values at each point
    g_values = g
    
    # Compute ||g||₂² using trapezoidal-like integration
    # For piecewise linear approximation, we use the formula:
    # ∫ g² ≈ Σ (h/3)(y₁² + y₁y₂ + y₂²) for consecutive points
    g_norm_sq = 0.0
    for i in range(len(g_values)-1):
        h = step_size  # width of interval
        y1 = g_values[i]
        y2 = g_values[i+1]
        g_norm_sq += (h/3) * (y1*y1 + y1*y2 + y2*y2)
    
    # ||g||₁ = sum of absolute values divided by number of points
    g_norm_1 = np.sum(np.abs(g_values)) / len(g_values)
    
    # ||g||∞ = max absolute value
    g_norm_inf = np.max(np.abs(g_values))
    
    return g_norm_sq, g_norm_1, g_norm_inf

def evaluate_c2(f_values: List[float]) -> float:
    """Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
    try:
        g_norm_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
            return 0.0
            
        c2 = g_norm_sq / (g_norm_1 * g_norm_inf)
        return c2
    except Exception:
        return 0.0

def generate_initial_population(pop_size: int, min_steps: int = 100, max_steps: int = 500) -> List[List[float]]:
    """Generate initial population of step function configurations"""
    population = []
    for _ in range(pop_size):
        # Randomly determine number of steps
        n_steps = random.randint(min_steps, max_steps)
        # Generate random step heights (non-negative)
        individual = [random.uniform(0, 1) for _ in range(n_steps)]
        population.append(individual)
    return population

def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate an individual by randomly changing some step heights"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add small random change
            mutated[i] = max(0, mutated[i] + random.gauss(0, 0.1))
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Single-point crossover between two individuals"""
    if len(parent1) == 0 or len(parent2) == 0:
        return parent1 if len(parent1) > 0 else parent2
    
    # Determine crossover point
    min_len = min(len(parent1), len(parent2))
    if min_len == 0:
        return parent1 if len(parent1) > 0 else parent2
    
    crossover_point = random.randint(0, min_len - 1)
    
    # Create offspring
    offspring = parent1[:crossover_point] + parent2[crossover_point:]
    
    # Make sure offspring has reasonable length (not too long or short)
    if len(offspring) < 50:
        offspring.extend([random.uniform(0, 1)] * (50 - len(offspring)))
    elif len(offspring) > 1000:
        offspring = offspring[:1000]
    
    return offspring

def evolutionary_search(max_time_seconds: float = 60.0) -> tuple:
    """Main evolutionary algorithm to find optimal step function"""
    start_time = time.time()
    
    # Parameters
    pop_size = 50
    generations = 1000
    elite_size = 5
    tournament_size = 3
    
    # Initialize population
    population = generate_initial_population(pop_size)
    
    best_fitness = 0.0
    best_individual = None
    
    for generation in range(generations):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Evaluate fitness for all individuals
        fitness_scores = [(evaluate_c2(ind), ind) for ind in population]
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Track best individual
        current_best_fitness, current_best_individual = fitness_scores[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual
        
        # Print progress every 100 generations
        if generation % 100 == 0:
            print(f"Generation {generation}: Best C2 = {best_fitness:.6f}")
        
        # Selection and reproduction
        # Keep elite
        new_population = [ind for _, ind in fitness_scores[:elite_size]]
        
        # Generate offspring through tournament selection and crossover
        while len(new_population) < pop_size:
            # Tournament selection
            tournament = random.sample(fitness_scores, tournament_size)
            winner = max(tournament, key=lambda x: x[0])
            
            # Another random individual for crossover
            other = random.choice(fitness_scores)[1]
            
            # Create offspring
            offspring = crossover(winner[1], other)
            
            # Mutate offspring
            offspring = mutate_individual(offspring)
            
            new_population.append(offspring)
        
        population = new_population
    
    return best_individual, best_fitness

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using evolutionary search"""
    # Run evolutionary search for up to 60 seconds
    best_individual, best_fitness = evolutionary_search(max_time_seconds=55.0)
    
    # Return the best solution found
    return best_individual if best_individual is not None else [0.5] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
