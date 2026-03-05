# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    f_values: list of step heights
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    # Create step function and compute autoconvolution
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Step size for domain [-1/4, 1/4]
    step_size = 0.5 / (n - 1) if n > 1 else 0.5
    
    # Create the step function (piecewise constant)
    # We'll use the trapezoidal integration approach for ||g||₂²
    # First compute the autoconvolution g = f * f
    
    # For simplicity, let's compute convolution manually
    # Convolution of step function with itself
    g_values = []
    
    # We'll create a finer grid for accurate convolution computation
    # Using the fact that convolution of two step functions results in 
    # piecewise linear function
    g_len = 2 * n - 1
    g_values = np.zeros(g_len)
    
    # Compute convolution manually
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < g_len:
                g_values[idx] += f_values[i] * f_values[j]
    
    # Compute norms
    # ||g||₂² = sum(g_i² * Δx) where Δx is step size
    # But we're using trapezoidal rule for piecewise linear integration
    # For a piecewise linear function with values [y0, y1, ..., yn], 
    # the integral is sum(Δx_i * (y_i + y_{i+1}) / 2) for trapezoidal rule
    # But since we want sum(y_i^2 * Δx) for ||g||₂², we need to be careful
    
    # Actually, let's recompute properly using the correct trapezoidal formula
    # for the integral of g^2
    
    # For ||g||₂² we compute sum of (y_i^2 * Δx) terms
    # But since we're dealing with discrete values, we'll use the piecewise linear
    # approximation for integration
    g_squared = np.array(g_values)**2
    
    # For trapezoidal rule: ∫g² ≈ Σ(Δx_i * (g_i² + g_{i+1}²)/2)
    # But for ||g||₂² we actually want ∫g² = Σ(g_i² * Δx_i)
    # Let's compute it properly:
    total_area = 0.0
    for i in range(len(g_values)-1):
        dx = 2 * step_size  # Since we're convolving two functions of width 0.5
        area = (g_values[i]**2 + g_values[i+1]**2) * dx / 2
        total_area += area
    
    # If there's only one point, handle specially
    if len(g_values) == 1:
        total_area = g_values[0]**2 * 2 * step_size
    
    # ||g||₁ = sum of absolute values / number of intervals
    g_abs_sum = np.sum(np.abs(g_values))
    g_norm_1 = g_abs_sum / (len(g_values) - 1) if len(g_values) > 1 else g_abs_sum
    
    # ||g||∞ = max of absolute values
    g_norm_inf = np.max(np.abs(g_values))
    
    return total_area, g_norm_1, g_norm_inf

def evaluate_c2(f_values: List[float]) -> float:
    """Evaluate C2 value for given step function."""
    try:
        norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_g1 <= 1e-12 or norm_ginf <= 1e-12:
            return 0.0
            
        c2 = norm_g2_sq / (norm_g1 * norm_ginf)
        return c2
    except Exception:
        return 0.0

def generate_initial_population(pop_size: int, min_steps: int = 50, max_steps: int = 200) -> List[List[float]]:
    """Generate initial population of step functions."""
    population = []
    for _ in range(pop_size):
        n_steps = random.randint(min_steps, max_steps)
        # Generate step heights with some structure to encourage good solutions
        # Start with a simple pattern that has some symmetry
        heights = []
        for i in range(n_steps):
            # Use a combination of random and structured approach
            if i < n_steps // 2:
                # First half: decreasing pattern
                heights.append(max(0, 1.0 - 0.5 * i / (n_steps // 2)))
            else:
                # Second half: increasing pattern
                heights.append(max(0, 0.5 * (n_steps - i) / (n_steps // 2)))
        population.append(heights)
    return population

def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate an individual by randomly changing some heights."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add some noise or change the height
            change_factor = random.uniform(0.8, 1.2)
            mutated[i] = max(0, mutated[i] * change_factor)
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Crossover two individuals."""
    # Simple uniform crossover
    child = []
    min_len = min(len(parent1), len(parent2))
    for i in range(min_len):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    # Fill remaining elements from parent1 or parent2
    if len(parent1) > min_len:
        child.extend(parent1[min_len:])
    elif len(parent2) > min_len:
        child.extend(parent2[min_len:])
    
    return child

def evolve_step_function() -> List[float]:
    """
    Evolve a step function to maximize C2.
    Uses a hybrid approach combining local search with genetic algorithm.
    """
    # Parameters
    pop_size = 50
    generations = 100
    mutation_rate = 0.1
    
    # Initialize population
    population = generate_initial_population(pop_size)
    
    best_fitness = 0.0
    best_individual = None
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = evaluate_c2(individual)
            fitness_scores.append((fitness, individual))
            
        # Sort by fitness
        fitness_scores.sort(reverse=True)
        
        # Track best
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_individual = fitness_scores[0][1].copy()
        
        # Selection (tournament selection)
        selected = []
        for _ in range(pop_size):
            tournament_size = 3
            tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
            winner = max(tournament, key=lambda x: x[0])
            selected.append(winner[1])
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Elitism: keep best individual
        new_population.append(best_individual.copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < pop_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            child = crossover(parent1, parent2)
            child = mutate_individual(child, mutation_rate)
            
            # Ensure non-negative values
            child = [max(0, x) for x in child]
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    return best_individual if best_individual is not None else []

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Try several approaches to find better solution
    best_solution = []
    best_c2 = 0.0
    
    # Approach 1: Genetic Algorithm
    try:
        ga_solution = evolve_step_function()
        ga_c2 = evaluate_c2(ga_solution)
        if ga_c2 > best_c2:
            best_c2 = ga_c2
            best_solution = ga_solution
    except Exception:
        pass
    
    # Approach 2: Simple heuristic - try symmetric patterns
    try:
        # Try a symmetric triangle pattern
        n = 100
        heights = []
        for i in range(n):
            if i < n//2:
                heights.append(i / (n//2))
            else:
                heights.append((n - i) / (n//2))
        
        # Normalize to prevent extreme values
        heights = [h / max(heights) if max(heights) > 0 else h for h in heights]
        
        sym_c2 = evaluate_c2(heights)
        if sym_c2 > best_c2:
            best_c2 = sym_c2
            best_solution = heights
    except Exception:
        pass
    
    # Return the best solution found
    if not best_solution:
        # Fallback to simple random approach if nothing works
        return [random.random() for _ in range(50)]
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
