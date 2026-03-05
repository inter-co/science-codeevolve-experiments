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
    
    # Direct computation of convolution - optimized version
    for i in range(n):
        f_i = f_values[i]
        for j in range(n):
            k = i + j
            if 0 <= k < g_length:
                g[k] += f_i * f_values[j]
    
    # Compute norms
    g_abs = np.abs(g)
    
    # L2 norm squared using trapezoidal-like integration - improved version
    norm_2_squared = 0.0
    for i in range(g_length - 1):
        y1, y2 = g[i], g[i+1]
        # Using Simpson's rule approximation for better accuracy
        norm_2_squared += step_width/3 * (y1**2 + y1*y2 + y2**2)
    
    # L1 norm - more accurate summation
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
    
    # Avoid division by zero with more robust handling
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    return norm_2_squared / (norm_1 * norm_inf)

def create_better_pattern(n: int) -> List[float]:
    """
    Create a much better optimized pattern using mathematical insights.
    """
    # Based on research that suggests patterns with flat tops and minimal peaks work well
    # Create a pattern with multiple peaks but controlled amplitudes to reduce sharp peaks
    pattern = []
    
    # Create a more sophisticated pattern that balances high values with low values
    # to create a flatter autoconvolution profile
    if n < 100:
        # For small n, use simpler approach
        for i in range(n):
            # Alternating pattern with more uniformity
            if i % 4 < 2:
                pattern.append(1.5)
            else:
                pattern.append(0.8)
    else:
        # For larger n, create a more complex pattern
        # Use a combination of Gaussian peaks and flat regions
        peak_positions = []
        num_peaks = min(8, n // 10)
        
        # Place peaks at regular intervals
        for i in range(num_peaks):
            pos = int((i + 1) * n / (num_peaks + 1))
            peak_positions.append(pos)
        
        # Generate pattern with peaks and valleys
        for i in range(n):
            # Distance to nearest peak
            min_dist = min(abs(i - pos) for pos in peak_positions)
            
            # Base value - low in valleys, high at peaks
            base_value = 0.5
            
            # Add peak contribution
            for pos in peak_positions:
                dist = abs(i - pos)
                # Gaussian peak with decreasing amplitude
                amplitude = 1.5 * math.exp(-dist**2 / (n//20))
                base_value += amplitude
                
            # Add some sinusoidal variation for smoother transitions
            sin_component = 0.2 * math.sin(2 * math.pi * i / (n//5))
            value = max(0.0, base_value + sin_component)
            
            # Add some randomness for exploration
            noise = 0.1 * random.random() - 0.05
            value = max(0.0, value + noise)
            
            pattern.append(value)
    
    # Normalize to make the total area reasonable
    total = sum(pattern)
    if total > 0:
        pattern = [x * 2.0 / total for x in pattern]
    
    return pattern

def create_focused_pattern(n: int) -> List[float]:
    """
    Create a focused pattern specifically designed to maximize C2.
    """
    # Inspired by mathematical analysis of optimal step functions
    # These patterns tend to create flatter autoconvolutions
    
    # Pattern based on symmetric peaks with controlled spacing
    pattern = []
    
    # Create a symmetric pattern with multiple peaks
    num_peaks = max(3, n // 20)
    
    # Determine peak positions
    peak_positions = []
    for i in range(num_peaks):
        # Distribute peaks more evenly
        pos = int((i + 1) * n / (num_peaks + 1))
        peak_positions.append(pos)
    
    # Create pattern
    for i in range(n):
        # Start with base level
        value = 0.5
        
        # Add contributions from peaks
        for j, peak_pos in enumerate(peak_positions):
            dist = abs(i - peak_pos)
            # Gaussian-like decay
            peak_height = 1.5 + 0.5 * math.sin(j * math.pi / num_peaks)  # Varying heights
            decay = math.exp(-dist**2 / (n // 15))
            value += peak_height * decay
            
        # Add some randomization to escape local optima
        noise_factor = 0.05
        noise = (random.random() - 0.5) * noise_factor
        value = max(0.0, value + noise)
        
        pattern.append(value)
    
    # Normalize
    total = sum(pattern)
    if total > 0:
        pattern = [x * 2.0 / total for x in pattern]
    
    return pattern

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
        lambda n: [2.0 if i % (n//6) == 0 and i > 0 else 0.8 for i in range(n)],
        # Configuration 2: Sine wave pattern (more evenly distributed)
        lambda n: [1.5 + 0.5 * math.sin(2 * math.pi * i / (n//10)) for i in range(n)],
        # Configuration 3: Double peak with tapering
        lambda n: [1.0 + 0.5 * math.exp(-((i - n//3)**2)/(n//20)) + 
                   0.5 * math.exp(-((i - 2*n//3)**2)/(n//20)) for i in range(n)],
        # Configuration 4: Gaussian-like shape with peak at center
        lambda n: [max(0, 1.5 * math.exp(-((i - n//2)**2)/(n//15))) for i in range(n)],
        # Configuration 5: Optimized pattern
        create_better_pattern,
        # Configuration 6: Focused pattern
        create_focused_pattern
    ]
    
    # Try different lengths and configurations
    lengths_to_try = [400, 500, 600, 700, 800, 900, 1000]
    
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
    if best_c2 < 0.95:
        # Use a more sophisticated evolutionary approach with adaptive parameters
        max_generations = 200
        pop_size = 120
        
        # Better initialization with more structured patterns
        initial_length = 800  # Increased length for better resolution
        
        # Create diverse initial population with better seeding
        population = []
        for _ in range(pop_size):
            individual = []
            # Create a more sophisticated pattern
            pattern_type = random.randint(0, 5)
            if pattern_type == 0:  # Multi-peak with strategic spacing
                for i in range(initial_length):
                    if i % (initial_length // 12) == 0 and i > 0:
                        individual.append(random.uniform(2.0, 3.0))
                    elif i % (initial_length // 15) == 0:
                        individual.append(random.uniform(1.0, 2.0))
                    else:
                        individual.append(random.uniform(0.0, 1.0))
            elif pattern_type == 1:  # Smooth variation with peaks
                for i in range(initial_length):
                    # Create smooth transition from low to high values
                    t = i / (initial_length - 1)
                    individual.append(0.5 + 1.5 * (t**2))
            elif pattern_type == 2:  # Central spike with tapering
                for i in range(initial_length):
                    distance_from_center = abs(i - initial_length//2)
                    value = max(0.0, 2.5 - distance_from_center / (initial_length//12))
                    individual.append(value)
            elif pattern_type == 3:  # Optimized pattern
                individual = create_better_pattern(initial_length)
            elif pattern_type == 4:  # Focused pattern
                individual = create_focused_pattern(initial_length)
            else:  # Random with structure and normalization
                for i in range(initial_length):
                    if i % 100 == 0 or i % 101 == 0:
                        individual.append(random.uniform(1.5, 3.5))
                    elif i % 50 == 0:
                        individual.append(random.uniform(1.0, 2.5))
                    else:
                        individual.append(random.uniform(0.0, 1.5))
                # Normalize to improve quality
                total = sum(individual)
                if total > 0:
                    individual = [x * 2.0 / total for x in individual]
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
            
            # Print progress every 25 generations
            if generation % 25 == 0:
                print(f"Generation {generation}: Best C2 = {best_c2:.6f}")
            
            # Selection with better diversity preservation
            selected = []
            tournament_size = 10
            for _ in range(pop_size):
                tournament = random.sample(fitness_scores, tournament_size)
                winner = max(tournament, key=lambda x: x[0])
                selected.append(winner[1])
            
            # Create new population through crossover and mutation
            new_population = []
            for i in range(0, pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % pop_size]
                
                # Improved crossover with more sophisticated recombination
                child1 = []
                child2 = []
                min_len = min(len(parent1), len(parent2))
                
                # Blend crossover with better probability distribution
                for j in range(min_len):
                    if random.random() < 0.6:  # Bias towards better parent
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
                mutation_rate = max(0.02, 0.2 * (1 - generation / max_generations))
                mutated = new_population[i].copy()
                for j in range(len(mutated)):
                    if random.random() < mutation_rate:
                        # Adaptive mutation with better scaling and bounds
                        if mutated[j] > 0:
                            base_mutation = 0.15 * mutated[j]
                        else:
                            base_mutation = 0.1
                        delta = random.gauss(0, base_mutation)
                        mutated[j] = max(0.0, mutated[j] + delta)
                new_population[i] = mutated
            
            # Stronger elitism - keep top 5 solutions
            if best_solution is not None:
                sorted_pop = sorted(fitness_scores, key=lambda x: x[0], reverse=True)
                new_population[:5] = [ind for _, ind in sorted_pop[:5]]
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
            
            # Simple local search around current best with more careful adjustment
            for _ in range(150):
                # Make small adjustments to improve the solution
                candidate = refined_solution.copy()
                idx = random.randint(0, len(candidate) - 1)
                # Small perturbation with adaptive size
                delta = random.uniform(-0.2, 0.2)
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
        n = 800
        return create_focused_pattern(n)
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
