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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import itertools
from scipy.spatial.distance import cdist

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
    
    # Direct computation of convolution - optimized version
    for i in range(n):
        for j in range(n):
            k = i + j
            if 0 <= k < g_length:
                g[k] += f_values[i] * f_values[j]
    
    # Compute norms
    g_abs = np.abs(g)
    
    # L2 norm squared using trapezoidal-like integration (more accurate)
    norm_2_squared = 0.0
    for i in range(g_length - 1):
        y1, y2 = g[i], g[i+1]
        # Using Simpson's rule approximation for better accuracy
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
    
    # Avoid division by zero with stricter checks
    if norm_1 <= 1e-12 or norm_inf <= 1e-12:
        return 0.0
    
    return norm_2_squared / (norm_1 * norm_inf)

def generate_advanced_pattern(n: int, pattern_type: str) -> List[float]:
    """Generate more sophisticated patterns for better C2 values."""
    if pattern_type == "optimized_symmetric":
        # Create a highly optimized symmetric pattern with sharp peaks
        pattern = []
        center = n // 2
        for i in range(n):
            distance = abs(i - center)
            # Use a combination of Gaussian and polynomial decay for better shape
            value = 2.5 * math.exp(-distance**2 / (n//12)) + 0.5 * (1 - distance/(n//2))**3
            pattern.append(max(0.0, value))
        return pattern
    
    elif pattern_type == "multi_peak":
        # Create multiple peaks with strategic placement
        pattern = [0.0] * n
        num_peaks = 5
        peak_positions = [int(i * n / (num_peaks + 1)) for i in range(1, num_peaks + 1)]
        
        for i, pos in enumerate(peak_positions):
            # Each peak has a Gaussian-like shape
            for j in range(n):
                distance = abs(j - pos)
                # Different amplitudes for different peaks
                amplitude = 3.0 if i == 0 or i == num_peaks-1 else 2.5
                pattern[j] += amplitude * math.exp(-distance**2 / (n//15))
        
        # Normalize to avoid extreme values
        max_val = max(pattern) if pattern else 1.0
        return [max(0.0, val/max_val * 2.0) for val in pattern]
    
    elif pattern_type == "wavelet_like":
        # Create wavelet-like pattern with oscillating peaks
        pattern = []
        for i in range(n):
            # Combine sine and cosine components with different frequencies
            t = i / (n - 1)
            value = 1.5 + 0.8 * math.sin(8 * math.pi * t) + 0.5 * math.cos(12 * math.pi * t)
            pattern.append(max(0.0, value))
        return pattern
    
    elif pattern_type == "bimodal":
        # Create bimodal distribution with two distinct peaks
        pattern = []
        center1, center2 = n//3, 2*n//3
        for i in range(n):
            dist1 = abs(i - center1)
            dist2 = abs(i - center2)
            # Two peaks with different widths
            val1 = 2.0 * math.exp(-dist1**2 / (n//10))
            val2 = 1.5 * math.exp(-dist2**2 / (n//12))
            pattern.append(max(0.0, val1 + val2))
        return pattern
    
    elif pattern_type == "exponential_decay":
        # Create exponential decay pattern with asymmetric properties
        pattern = []
        for i in range(n):
            # Create an exponential decay from left to right
            t = i / (n - 1)
            value = 3.0 * math.exp(-t * 3) + 0.5
            pattern.append(max(0.0, value))
        return pattern
    
    else:
        # Default symmetric pattern
        return [max(0.0, 2.0 * (1.0 - abs(i - n//2) / (n//2))) for i in range(n)]

def construct_function() -> List[float]:
    """
    Construct step-function with high C2 value using advanced optimization techniques.
    """
    start_time = time.time()
    
    # Start with a good heuristic pattern
    best_solution = None
    best_c2 = 0.0
    
    # Try different advanced configurations
    pattern_types = [
        "optimized_symmetric", "multi_peak", "wavelet_like", 
        "bimodal", "exponential_decay"
    ]
    
    # Try different lengths and configurations
    lengths_to_try = [500, 600, 700, 800, 900]
    
    # First try advanced heuristic approaches
    for length in lengths_to_try:
        for pattern_type in pattern_types:
            try:
                test_solution = generate_advanced_pattern(length, pattern_type)
                c2 = calculate_c2(test_solution)
                if c2 > best_c2:
                    best_c2 = c2
                    best_solution = test_solution.copy()
            except Exception as e:
                continue
    
    # Use more sophisticated global optimization with better parameters
    if best_c2 < 0.92:
        try:
            # Use a hybrid approach with simulated annealing and gradient-free methods
            from scipy.optimize import differential_evolution, dual_annealing
            
            # Create a larger initial population with diverse patterns
            initial_solutions = []
            n = 700  # Standard size
            
            # Generate diverse solutions
            for i in range(100):
                # Mix different pattern types with randomness
                pattern_type = random.choice(["optimized_symmetric", "multi_peak", "bimodal"])
                solution = generate_advanced_pattern(n, pattern_type)
                
                # Add some noise to create diversity
                for j in range(len(solution)):
                    if random.random() < 0.1:  # 10% chance to mutate
                        solution[j] = max(0.0, solution[j] + random.gauss(0, 0.2))
                
                initial_solutions.append(solution)
            
            # Evaluate all initial solutions
            fitness_scores = []
            for solution in initial_solutions:
                fitness = calculate_c2(solution)
                fitness_scores.append((fitness, solution))
            
            # Sort by fitness and get best
            fitness_scores.sort(reverse=True)
            if fitness_scores:
                best_initial_c2, best_initial_solution = fitness_scores[0]
                if best_initial_c2 > best_c2:
                    best_c2 = best_initial_c2
                    best_solution = best_initial_solution.copy()
                    
        except Exception as e:
            pass
    
    # Enhanced evolutionary algorithm with better strategies
    if best_c2 < 0.95:
        # Use a more sophisticated evolutionary approach with adaptive parameters
        max_generations = 150
        pop_size = 100
        
        # Better initialization with more structured patterns
        initial_length = 700  # Increased length for better resolution
        
        # Create diverse initial population with better seeding
        population = []
        for _ in range(pop_size):
            individual = []
            # Use more sophisticated pattern generation
            pattern_type = random.choice([
                "optimized_symmetric", "multi_peak", "bimodal", 
                "wavelet_like", "exponential_decay"
            ])
            
            individual = generate_advanced_pattern(initial_length, pattern_type)
            
            # Add some controlled randomization to encourage exploration
            for i in range(len(individual)):
                if random.random() < 0.05:  # 5% mutation rate
                    individual[i] = max(0.0, individual[i] + random.gauss(0, individual[i] * 0.1 if individual[i] > 0 else 0.1))
            
            population.append(individual)
        
        # Evolution loop with improved strategies
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
            
            # Print progress every 20 generations
            if generation % 20 == 0:
                print(f"Generation {generation}: Best C2 = {best_c2:.6f}")
            
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
                # Adaptive mutation rate that decreases over time
                mutation_rate = max(0.03, 0.15 * (1 - generation / max_generations))
                mutated = new_population[i].copy()
                
                # Apply mutation to a subset of genes
                num_mutations = max(1, int(len(mutated) * mutation_rate * 0.3))
                mutation_indices = random.sample(range(len(mutated)), num_mutations)
                
                for j in mutation_indices:
                    if random.random() < 0.8:  # 80% chance of Gaussian mutation
                        # Adaptive mutation with better scaling
                        if mutated[j] > 0:
                            base_mutation = 0.2 * mutated[j]
                        else:
                            base_mutation = 0.1
                        delta = random.gauss(0, base_mutation)
                        mutated[j] = max(0.0, mutated[j] + delta)
                    else:  # 20% chance of uniform mutation
                        mutated[j] = max(0.0, mutated[j] + random.uniform(-0.5, 0.5))
                        
                new_population[i] = mutated
            
            # Stronger elitism - keep best solution
            if best_solution is not None:
                new_population[0] = best_solution.copy()
            population = new_population[:pop_size]
            
            # Check time limit
            if time.time() - start_time > 55:  # Leave some buffer
                break
    
    # Try local optimization for final improvement
    if best_c2 < 0.98 and best_solution is not None:
        try:
            # Try more aggressive hill climbing around current best solution
            refined_solution = best_solution.copy()
            current_c2 = best_c2
            
            # Perform more extensive local search
            for iter_num in range(500):  # More iterations
                # Make multiple small adjustments to improve the solution
                candidate = refined_solution.copy()
                
                # Mutate several positions at once
                num_changes = max(1, len(candidate) // 50)
                change_indices = random.sample(range(len(candidate)), num_changes)
                
                for idx in change_indices:
                    # Larger perturbation for exploration
                    delta = random.uniform(-0.8, 0.8)
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
        # Fallback to a carefully constructed pattern
        n = 700
        # Create a high-quality symmetric pattern based on mathematical insights
        return generate_advanced_pattern(n, "optimized_symmetric")
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
