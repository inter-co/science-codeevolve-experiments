# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
import random
from typing import List
import math
from scipy.optimize import differential_evolution
from scipy.optimize import minimize_scalar
import time
from numba import jit
import copy
from collections import deque

@jit(nopython=True)
def compute_autoconvolution_norms_jit(f_values: List[float]) -> tuple:
    """
    Optimized computation of autoconvolution norms using numba for speed
    """
    n = len(f_values)
    # Autoconvolution using manual implementation for efficiency
    g = np.zeros(2*n - 1)
    
    # Manual convolution (f*f)
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    # Take central portion (this matches what the evaluator expects)
    g_centered = g[n-1:2*n-1]
    
    # Compute norms
    g_abs = np.abs(g_centered)
    
    # ||g||_1 = sum of absolute values
    norm_g1 = np.sum(g_abs)
    
    # ||g||_inf = max of absolute values  
    norm_ginf = np.max(g_abs)
    
    # ||g||_2^2 = more accurate piecewise linear integration
    # Using trapezoidal rule with triangular approximation for better accuracy
    if len(g_centered) <= 1:
        norm_g2_squared = 0.0
    else:
        # More precise integration: for each segment, integrate quadratic function
        # that matches the values at endpoints
        g_vals = g_centered
        norm_g2_squared = 0.0
        for i in range(len(g_vals) - 1):
            y1, y2 = g_vals[i], g_vals[i+1]
            # For trapezoidal integration of y^2, we integrate the quadratic that passes through (0,y1), (1,y2)
            # This gives us integral from 0 to 1 of ((y2-y1)*t + y1)^2 dt = (y1^2 + y1*y2 + y2^2)/3
            norm_g2_squared += (y1**2 + y1*y2 + y2**2) / 3.0
    
    return norm_g2_squared, norm_g1, norm_ginf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    """
    return compute_autoconvolution_norms_jit(f_values)

def evaluate_c2(f_values: List[float]) -> float:
    """Evaluate C2 for given step function values"""
    try:
        norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_g1 <= 1e-15 or norm_ginf <= 1e-15:
            return 0.0
            
        c2 = norm_g2_sq / (norm_g1 * norm_ginf)
        return c2
    except Exception as e:
        return 0.0

def create_structured_individual(length: int, pattern_type: str = "sine") -> List[float]:
    """Create structured individuals to improve convergence"""
    individual = []
    
    if pattern_type == "sine":
        # Create sine wave pattern
        for i in range(length):
            t = i / (length - 1) if length > 1 else 0
            value = 0.5 + 0.5 * math.sin(4 * math.pi * t)
            individual.append(max(0, value))
    elif pattern_type == "step":
        # Create step pattern with gradual transition
        mid = length // 2
        for i in range(length):
            if i < mid:
                individual.append(1.0)
            else:
                individual.append(0.0)
    elif pattern_type == "gaussian":
        # Create Gaussian-like pattern with sharper peak
        center = (length - 1) / 2
        sigma = length / 8
        for i in range(length):
            value = math.exp(-0.5 * ((i - center) / sigma)**2)
            individual.append(max(0, value))
    elif pattern_type == "bump":
        # Create bump function pattern
        center = (length - 1) / 2
        sigma = length / 10
        for i in range(length):
            value = math.exp(-0.5 * ((i - center) / sigma)**2) * 0.8
            individual.append(max(0, value))
    elif pattern_type == "double_bump":
        # Create double bump pattern
        center1 = length // 3
        center2 = 2 * length // 3
        sigma = length / 12
        for i in range(length):
            val1 = math.exp(-0.5 * ((i - center1) / sigma)**2) * 0.7
            val2 = math.exp(-0.5 * ((i - center2) / sigma)**2) * 0.7
            individual.append(max(0, val1 + val2))
    elif pattern_type == "ripple":
        # Create ripple pattern
        for i in range(length):
            t = i / (length - 1) if length > 1 else 0
            value = 0.5 + 0.5 * math.sin(8 * math.pi * t) * math.exp(-0.1 * length * abs(t - 0.5))
            individual.append(max(0, value))
    else:
        # Random pattern with some structure
        for i in range(length):
            if random.random() < 0.3:
                individual.append(random.uniform(0.8, 1.0))
            else:
                individual.append(random.uniform(0.0, 0.5))
    
    return individual

def mutate(individual: List[float], mutation_strength: float = 0.1) -> List[float]:
    """Enhanced mutation with adaptive strength and better distribution"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < 0.1:  # 10% chance to mutate each element
            # Use log-normal distribution for better control
            if mutated[i] > 0.01:
                # Apply multiplicative mutation for better scaling
                factor = random.gauss(1.0, mutation_strength)
                mutated[i] = max(0, mutated[i] * factor)
            else:
                # Additive for small values
                delta = random.gauss(0, mutation_strength)
                mutated[i] = max(0, mutated[i] + delta)
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Improved crossover with blend and uniform mixing"""
    if len(parent1) != len(parent2):
        # Make lengths compatible
        min_len = min(len(parent1), len(parent2))
        p1 = parent1[:min_len]
        p2 = parent2[:min_len]
    else:
        p1 = parent1
        p2 = parent2
    
    # Blend crossover with weighted averaging
    child = []
    alpha = random.random()  # Weight for blending
    for i in range(len(p1)):
        # Blend with some randomness
        if random.random() < 0.7:
            # Weighted average
            child.append(alpha * p1[i] + (1 - alpha) * p2[i])
        else:
            # Pick from either parent
            child.append(p1[i] if random.random() < 0.5 else p2[i])
    
    return child

def local_search_multi_start(individual: List[float], max_iterations: int = 100) -> List[float]:
    """Multi-start local search to avoid local optima"""
    current = individual.copy()
    current_c2 = evaluate_c2(current)
    
    # Keep track of best solution found
    best_solution = current.copy()
    best_c2 = current_c2
    
    # Multiple restarts with different perturbations
    for restart in range(5):
        # Start with slightly perturbed version
        test_individual = individual.copy()
        for i in range(len(test_individual)):
            if random.random() < 0.1:
                if test_individual[i] > 0.01:
                    delta = random.gauss(0, 0.05)
                    test_individual[i] = max(0, test_individual[i] * (1 + delta))
                else:
                    delta = random.gauss(0, 0.05)
                    test_individual[i] = max(0, test_individual[i] + delta)
        
        # Simulated annealing parameters
        temp = 1.0
        cooling_rate = 0.995
        
        for iteration in range(max_iterations // 5):
            # Try perturbations
            new_individual = test_individual.copy()
            idx = random.randint(0, len(new_individual) - 1)
            
            # Better perturbation with adaptive temperature
            if new_individual[idx] > 0.01:
                delta = random.gauss(0, 0.05 * temp)
                new_individual[idx] = max(0, new_individual[idx] * (1 + delta))
            else:
                delta = random.gauss(0, 0.05 * temp)
                new_individual[idx] = max(0, new_individual[idx] + delta)
            
            new_c2 = evaluate_c2(new_individual)
            
            # Accept with probability based on difference and temperature
            if new_c2 > current_c2 or random.random() < math.exp((new_c2 - current_c2) / (temp + 1e-10)):
                test_individual = new_individual
                current_c2 = new_c2
            
            # Cool down temperature
            temp *= cooling_rate
            
            # Update best solution
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_solution = test_individual.copy()
    
    return best_solution

def adaptive_evolutionary_strategy() -> List[float]:
    """
    Advanced evolutionary strategy with adaptive parameters and improved operators
    """
    # Parameters
    population_size = 200
    generations = 80
    elite_size = 20
    mutation_strength = 0.25
    
    # Create initial population with diverse patterns
    population = []
    pattern_types = ["sine", "step", "gaussian", "bump", "double_bump", "ripple"]
    
    for _ in range(population_size):
        length = random.randint(400, 1000)  # Larger range for better exploration
        pattern_type = random.choice(pattern_types)
        individual = create_structured_individual(length, pattern_type)
        population.append(individual)
    
    best_fitness = -1
    best_individual = None
    fitness_history = deque(maxlen=20)
    
    # Evolution loop
    for generation in range(generations):
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
        
        # Track history for adaptive parameters
        fitness_history.append(best_fitness)
        
        # Print progress every 10 generations
        if generation % 10 == 0:
            avg_fitness = sum([f for f, _ in fitness_scores[:20]]) / 20 if len(fitness_scores) >= 20 else 0
            print(f"Generation {generation}: Best C2 = {best_fitness:.6f}, Avg = {avg_fitness:.6f}")
        
        # Adaptive mutation strength based on convergence
        adaptive_mutation = mutation_strength
        if len(fitness_history) >= 10:
            recent_improvement = fitness_history[-1] - fitness_history[0] if fitness_history[0] > 0 else 0
            if recent_improvement < 1e-5:
                adaptive_mutation = min(0.5, mutation_strength * 1.2)  # Increase mutation if stuck
        
        # Selection and breeding
        new_population = []
        
        # Elitism - keep top individuals
        elites = [ind for _, ind in fitness_scores[:elite_size]]
        new_population.extend(elites)
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection with adaptive tournament size
            tournament_size = max(5, min(15, int(10 - generation / 10)))  # Decreasing tournament size
            tournament = random.sample(fitness_scores, tournament_size)
            parent1 = max(tournament, key=lambda x: x[0])[1]
            
            tournament = random.sample(fitness_scores, tournament_size)
            parent2 = max(tournament, key=lambda x: x[0])[1]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, adaptive_mutation)
            
            # Occasionally create completely new individuals
            if random.random() < 0.05:
                length = random.randint(400, 1000)
                pattern_type = random.choice(pattern_types)
                child = create_structured_individual(length, pattern_type)
            
            new_population.append(child)
        
        # Ensure correct population size
        population = new_population[:population_size]
    
    # Final refinement with enhanced local search
    if best_individual is not None:
        refined_individual = local_search_multi_start(best_individual, 500)
        final_c2 = evaluate_c2(refined_individual)
        print(f"Final C2 after refinement: {final_c2:.6f}")
        return refined_individual
    
    return best_individual if best_individual is not None else [0.5] * 100

def construct_function() -> List[float]:
    """Main function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Use advanced evolutionary approach to find good configuration
    try:
        result = adaptive_evolutionary_strategy()
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        return result
    except Exception as e:
        print(f"Error during evolution: {e}")
        # Fallback to simple approach if evolution fails
        return [1.0] * 200

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
