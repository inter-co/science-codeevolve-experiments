# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import random
from scipy import signal
from scipy.optimize import differential_evolution
import time
from numba import jit
import math
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: List[float]) -> tuple:
    """
    Fast computation of autoconvolution norms using Numba JIT compilation.
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Create step function on [-1/4, 1/4] with given heights
    step_width = 0.5 / n
    
    # Compute autoconvolution manually for better control and speed
    # The autoconvolution g[k] = sum_{i} f[i] * f[k-i] 
    g_length = 2 * n - 1
    g = np.zeros(g_length)
    
    # Direct computation of convolution
    for i in range(n):
        for j in range(n):
            k = i + j
            if 0 <= k < g_length:
                g[k] += f_values[i] * f_values[j]
    
    # Compute norms
    g_abs = np.abs(g)
    
    # L2 norm squared using trapezoidal-like integration
    norm_2_squared = 0.0
    for i in range(g_length - 1):
        y1, y2 = g[i], g[i+1]
        norm_2_squared += step_width/3 * (y1**2 + y1*y2 + y2**2)
    
    # L1 norm
    norm_1 = np.sum(g_abs) * step_width
    
    # L-infinity norm
    norm_inf = np.max(g_abs)
    
    return norm_2_squared, norm_1, norm_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    """
    return compute_autoconvolution_norms_fast(f_values)

def calculate_c2(f_values: List[float]) -> float:
    """
    Calculate C2 value for given step function.
    """
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    return norm_2_squared / (norm_1 * norm_inf)

def construct_function() -> List[float]:
    """
    Construct step-function with high C2 value using advanced optimization techniques.
    """
    start_time = time.time()
    
    # Start with a good heuristic pattern
    best_solution = None
    best_c2 = 0.0
    
    # Try different configurations with more sophisticated patterns
    configurations = [
        # Configuration 1: Multi-peak pattern that maximizes spread
        lambda n: [2.0 if i % (n//8) == 0 and i > 0 else 0.8 for i in range(n)],
        # Configuration 2: Sine wave pattern (more evenly distributed)
        lambda n: [1.5 + 0.5 * math.sin(2 * math.pi * i / (n//10)) for i in range(n)],
        # Configuration 3: Double peak with tapering
        lambda n: [1.0 + 0.5 * math.exp(-((i - n//3)**2)/(n//20)) + 
                   0.5 * math.exp(-((i - 2*n//3)**2)/(n//20)) for i in range(n)],
        # Configuration 4: Gaussian-like shape with peak at center
        lambda n: [max(0, 1.5 * math.exp(-((i - n//2)**2)/(n//15))) for i in range(n)],
        # Configuration 5: Optimized multi-peak pattern based on mathematical intuition
        lambda n: [1.0 + 0.5 * math.sin(2 * math.pi * i / (n//4)) + 
                   0.3 * math.sin(4 * math.pi * i / (n//4)) for i in range(n)]
    ]
    
    # Try different lengths and configurations
    lengths_to_try = [400, 500, 600, 700, 800]
    
    # First try heuristic approaches
    for length in lengths_to_try:
        for config_func in configurations:
            try:
                test_solution = config_func(length)
                c2 = calculate_c2(test_solution)
                if c2 > best_c2:
                    best_c2 = c2
                    best_solution = test_solution.copy()
            except Exception as e:
                continue
    
    # Enhanced evolutionary algorithm with better parameters and strategies
    if best_c2 < 0.92:
        # Use a more sophisticated evolutionary approach with adaptive parameters
        max_generations = 150
        pop_size = 100
        
        # Better initialization with more structured patterns
        initial_length = 600  # Increased length for better resolution
        
        # Create diverse initial population with better seeding
        population = []
        for _ in range(pop_size):
            individual = []
            # Create a more sophisticated pattern
            pattern_type = random.randint(0, 4)
            if pattern_type == 0:  # Multi-peak with careful spacing
                for i in range(initial_length):
                    if i % (initial_length // 10) == 0 and i > 0:
                        individual.append(random.uniform(2.0, 3.5))
                    elif i % (initial_length // 15) == 0:
                        individual.append(random.uniform(1.0, 2.5))
                    else:
                        individual.append(random.uniform(0.0, 1.5))
            elif pattern_type == 1:  # Smooth variation with exponential decay
                for i in range(initial_length):
                    # Create smooth transition from low to high values
                    t = i / (initial_length - 1)
                    individual.append(0.5 + 2.0 * (t**2))
            elif pattern_type == 2:  # Central spike with symmetric tapering
                for i in range(initial_length):
                    distance_from_center = abs(i - initial_length//2)
                    value = max(0.0, 3.0 - distance_from_center / (initial_length//12))
                    individual.append(value)
            elif pattern_type == 3:  # Multiple peaks with decreasing amplitude
                for i in range(initial_length):
                    value = 0.0
                    for j in range(5):  # 5 peaks
                        peak_pos = (j + 1) * initial_length // 6
                        amplitude = 2.0 / (j + 1)
                        value += amplitude * math.exp(-((i - peak_pos)**2)/(initial_length//20))
                    individual.append(max(0.0, value))
            else:  # Random with structure and adaptive scaling
                for i in range(initial_length):
                    if i % 120 == 0 or i % 121 == 0:
                        individual.append(random.uniform(2.0, 4.0))
                    elif i % 60 == 0:
                        individual.append(random.uniform(1.0, 2.5))
                    else:
                        individual.append(random.uniform(0.0, 1.8))
            population.append(individual)
        
        # Evolution loop with improved strategies
        stagnation_count = 0
        prev_best = 0.0
        
        for generation in range(max_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = calculate_c2(individual)
                fitness_scores.append((fitness, individual))
                
            # Sort by fitness
            fitness_scores.sort(reverse=True)
            
            # Track best individual
            current_best_fitness, current_best = fitness_scores[0]
            if current_best_fitness > best_c2:
                best_c2 = current_best_fitness
                best_solution = current_best.copy()
                stagnation_count = 0  # Reset stagnation counter
            else:
                stagnation_count += 1
                
            # Print progress every 20 generations
            if generation % 20 == 0:
                print(f"Generation {generation}: Best C2 = {best_c2:.6f}")
            
            # Early stopping if no improvement for 30 generations
            if stagnation_count >= 30:
                print(f"Early stopping at generation {generation}")
                break
            
            # Selection with better diversity preservation
            selected = []
            tournament_size = 8
            for _ in range(pop_size):
                tournament = random.sample(fitness_scores, tournament_size)
                winner = max(tournament, key=lambda x: x[0])
                selected.append(winner[1])
            
            # Create new population through crossover and mutation
            new_population = []
            for i in range(0, pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % pop_size]
                
                # Improved crossover with better probability distribution
                child1 = []
                child2 = []
                min_len = min(len(parent1), len(parent2))
                
                # Blend crossover with better probability distribution
                for j in range(min_len):
                    if random.random() < 0.7:  # Bias towards better parent
                        child1.append(parent1[j])
                        child2.append(parent2[j])
                    else:
                        child1.append(parent2[j])
                        child2.append(parent1[j])
                
                # Extend with remaining elements
                if len(parent1) > min_len:
                    child1.extend(parent1[min_len:])
                if len(parent2) > min_len:
                    child2.extend(parent2[min_len:])
                    
                new_population.extend([child1, child2])
            
            # Enhanced mutation with better strategies
            for i in range(len(new_population)):
                # Dynamic mutation rate that decreases over time
                mutation_rate = max(0.03, 0.3 * (1 - generation / max_generations))
                mutated = new_population[i].copy()
                for j in range(len(mutated)):
                    if random.random() < mutation_rate:
                        # Adaptive mutation with better scaling
                        if mutated[j] > 0:
                            base_mutation = 0.1 * mutated[j]
                        else:
                            base_mutation = 0.1
                        # Use larger jumps for early generations, smaller for later
                        delta = random.gauss(0, base_mutation * (1.0 - generation/max_generations + 0.2))
                        mutated[j] = max(0.0, mutated[j] + delta)
                new_population[i] = mutated
            
            # Stronger elitism
            if best_solution is not None:
                new_population[0] = best_solution.copy()
            population = new_population[:pop_size]
            
            # Check time limit
            if time.time() - start_time > 55:  # Leave some buffer
                break
    
    # If still no good solution, try a gradient-free optimization approach
    if best_c2 < 0.95 and best_solution is not None:
        try:
            # Try optimization with a few iterations of direct search
            # Use a more targeted approach with known good patterns
            refined_solution = best_solution.copy()
            current_c2 = best_c2
            
            # Simple local search around current best with better strategy
            for _ in range(100):
                # Make small adjustments to improve the solution
                candidate = refined_solution.copy()
                idx = random.randint(0, len(candidate) - 1)
                # Larger perturbation for exploration
                delta = random.uniform(-0.5, 0.5)
                candidate[idx] = max(0.0, candidate[idx] + delta)
                
                new_c2 = calculate_c2(candidate)
                if new_c2 > current_c2:
                    current_c2 = new_c2
                    refined_solution = candidate.copy()
                    if current_c2 > best_c2:
                        best_c2 = current_c2
                        best_solution = refined_solution.copy()
                        
        except Exception as e:
            pass
    
    # If still no good solution, return the best we found
    if best_solution is None:
        # Fallback to a carefully constructed pattern with mathematical insight
        n = 600
        # Create a pattern that balances peak height and distribution
        pattern = []
        for i in range(n):
            # Create a smooth distribution with higher values near center and tapering
            t = i / (n - 1)
            # Use sigmoid-like shape for smooth transition
            value = 1.5 + 0.8 * (1/(1 + math.exp(-10*(t - 0.5))))
            pattern.append(max(0.0, value))
        return pattern
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
