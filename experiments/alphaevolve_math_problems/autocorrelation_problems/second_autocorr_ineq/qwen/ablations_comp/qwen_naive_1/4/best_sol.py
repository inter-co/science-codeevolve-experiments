# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from typing import List
import random
import time
from numba import jit
import math

@jit(nopython=True)
def compute_piecewise_integral(g_values: np.ndarray, dx: float) -> float:
    """Compute piecewise integral using trapezoidal-like formula - optimized version"""
    if len(g_values) < 2:
        return g_values[0]**2 if len(g_values) > 0 else 0.0
    
    integral = 0.0
    for i in range(len(g_values) - 1):
        integral += (dx/3) * (g_values[i]**2 + g_values[i]*g_values[i+1] + g_values[i+1]**2)
    
    # Add boundary terms correctly
    integral += (dx/6) * (g_values[0]**2 + g_values[-1]**2)
    return integral

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    f_values: step heights on [-1/4, 1/4] with equal spacing
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    if not f_values:
        return 0.0, 0.0, 0.0
    
    # Create step function on [-1/4, 1/4]
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Step width
    dx = 0.5 / n
    
    # Create the step function f
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using discrete convolution
    # Convolution of two functions on [-1/4, 1/4] produces a function on [-1/2, 1/2]
    g_full = np.convolve(f, f, mode='full')
    
    # Extract the central region [-1/4, 1/4] which corresponds to the meaningful part
    # The full convolution has 2*n-1 points spanning [-1/2, 1/2]
    # We want the central n points corresponding to [-1/4, 1/4]
    mid_start = (len(g_full) - n) // 2
    mid_end = mid_start + n
    g_middle = g_full[mid_start:mid_end]
    
    # Now compute the norms using proper piecewise integration
    g = g_middle
    
    # ||g||₂² using piecewise linear integration formula
    g2_norm_squared = compute_piecewise_integral(g, dx)
    
    # ||g||₁ = sum of absolute values  
    g1_norm = np.sum(np.abs(g))
    
    # ||g||∞ = maximum absolute value
    ginf_norm = np.max(np.abs(g))
    
    return g2_norm_squared, g1_norm, ginf_norm

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given step function values."""
    g2_sq, g1, ginf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if g1 <= 1e-15 or ginf <= 1e-15:
        return 0.0
    
    c2 = g2_sq / (g1 * ginf)
    return c2

def create_initial_population(size: int, min_steps: int = 100, max_steps: int = 1000) -> List[List[float]]:
    """Create initial population of step function configurations with improved strategies."""
    population = []
    for _ in range(size):
        n = random.randint(min_steps, max_steps)
        
        # Strategy 1: Optimized peak-centered pattern with mathematical foundation
        f_vals = []
        center = n // 2
        
        # Create a more sophisticated peak with better mathematical properties
        # Using Gaussian profile with optimized parameters
        sigma = n / 8.0  # Slightly narrower spread for sharper peaks
        for i in range(n):
            distance = (i - center) / sigma
            # Use Gaussian decay for sharper peak with smooth derivatives
            val = max(0, np.exp(-0.5 * distance**2)) 
            # Add controlled noise for diversity
            noise = random.uniform(-0.01, 0.01)
            val = max(0, val + noise)
            f_vals.append(val)
        
        # Strategy 2: Multi-peak with better distribution
        if random.random() < 0.35:
            f_vals = []
            # Create multiple peaks with decreasing amplitudes
            num_peaks = random.randint(2, 8)  # Increased number of peaks
            # Distribute peaks more evenly
            peak_positions = []
            for i in range(num_peaks):
                # Distribute peaks more evenly across the domain
                pos = (i + 1) * (n / (num_peaks + 1))
                peak_positions.append(int(pos))
            
            peak_heights = [random.uniform(0.7, 1.0) for _ in range(num_peaks)]
            
            for i in range(n):
                val = 0.0
                pos = i / n
                for j in range(num_peaks):
                    peak_pos = peak_positions[j] / n
                    distance = abs(pos - peak_pos)
                    # Gaussian peaks with decreasing amplitudes
                    val += peak_heights[j] * np.exp(-30 * distance**2)  # Sharper peaks
                val = max(0, val + random.uniform(-0.02, 0.02))
                f_vals.append(val)
        
        # Strategy 3: Sine wave with harmonics for complex patterns
        elif random.random() < 0.25:
            f_vals = []
            for i in range(n):
                pos = i / n
                # Combination of sine waves with different frequencies
                val = 0.4 + 0.3 * np.sin(10 * np.pi * pos) + 0.2 * np.sin(20 * np.pi * pos)
                # Add a central peak for better autoconvolution
                if abs(pos - 0.5) < 0.1:
                    val += 0.3
                val = max(0, val + random.uniform(-0.03, 0.03))
                f_vals.append(val)
                
        # Strategy 4: Sparse pattern with high peaks
        elif random.random() < 0.15:
            f_vals = [0.0] * n
            num_peaks = max(2, n // 20)  # More peaks for better exploration
            for _ in range(num_peaks):
                idx = random.randint(0, n-1)
                f_vals[idx] = random.uniform(0.8, 1.0)
                
        # Strategy 5: Flattened with subtle variations
        elif random.random() < 0.15:
            f_vals = []
            base_level = 0.5
            for i in range(n):
                # Add subtle sinusoidal variation
                pos = i / n
                variation = 0.1 * np.sin(8 * np.pi * pos) + 0.05 * np.cos(16 * np.pi * pos)
                val = max(0, base_level + variation + random.uniform(-0.02, 0.02))
                f_vals.append(val)
        
        # Strategy 6: Gradient pattern with peaks
        else:
            f_vals = []
            # Create a gradient with a central peak
            for i in range(n):
                pos = i / n
                # Create a gradient from low to high and back
                if pos < 0.5:
                    val = 0.3 + 0.7 * (pos * 2)
                else:
                    val = 0.3 + 0.7 * (1 - (pos - 0.5) * 2)
                # Add a central peak
                if abs(pos - 0.5) < 0.1:
                    val += 0.3 * (1 - abs(pos - 0.5) * 10)
                val = max(0, val + random.uniform(-0.03, 0.03))
                f_vals.append(val)
                
        population.append(f_vals)
    return population

def mutate_individual(f_values: List[float], mutation_rate: float = 0.1, generation: int = 0) -> List[float]:
    """Mutate a single individual with enhanced strategies."""
    new_values = f_values.copy()
    
    # Adaptive mutation rate that decreases over time
    adaptive_mutation_rate = mutation_rate * (1.0 - generation / 1000.0)
    adaptive_mutation_rate = max(0.005, adaptive_mutation_rate)
    
    # More aggressive early mutations, then fine-tuning
    if generation < 200:
        mutation_intensity = 2.0  # More aggressive early
    elif generation < 500:
        mutation_intensity = 1.0  # Moderate
    else:
        mutation_intensity = 0.5  # Fine-tuning
    
    for i in range(len(new_values)):
        if random.random() < adaptive_mutation_rate:
            current_val = new_values[i]
            
            # 60% chance of small Gaussian noise
            if random.random() < 0.6:
                # Add small Gaussian noise with adaptive scale
                noise_scale = 0.03 * mutation_intensity + 0.01 * (1.0 - current_val) 
                noise = random.gauss(0, noise_scale)
                new_values[i] += noise
                # Ensure non-negative
                new_values[i] = max(0, new_values[i])
            else:
                # 40% chance of significant change
                # Either random replacement or significant perturbation
                if random.random() < 0.7:
                    # Random replacement from better distribution
                    if current_val > 0.5:
                        new_values[i] = random.uniform(0, 0.8)
                    else:
                        new_values[i] = random.uniform(0, 1.0)
                else:
                    # Significant perturbation - use log-normal for better scaling
                    factor = random.uniform(0.7, 1.3)
                    new_values[i] = max(0, current_val * factor)
    
    return new_values

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform crossover with enhanced blending strategies."""
    size = min(len(parent1), len(parent2))
    child = []
    
    # 75% uniform crossover (more uniform)
    for i in range(size):
        if random.random() < 0.75:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    # 25% blend with weighted average - improved blending
    blend_indices = random.sample(range(size), int(0.25 * size))
    for i in blend_indices:
        alpha = random.random()
        child[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
    
    # If lengths differ, extend appropriately
    if len(parent1) > size:
        child.extend(parent1[size:])
    elif len(parent2) > size:
        child.extend(parent2[size:])
    
    return child

def optimize_step_function() -> List[float]:
    """Optimize step function to maximize C2 using enhanced evolutionary approach."""
    # Enhanced parameters for better performance
    population_size = 500  # Increased population size
    generations = 1200  # More generations
    elite_size = 50  # More elites
    mutation_rate = 0.15  # Higher mutation rate for better exploration
    
    # Initialize population with better starting points
    population = create_initial_population(population_size)
    
    best_c2 = 0.0
    best_solution = None
    stagnation_count = 0
    prev_best = 0.0
    
    # Evolutionary loop
    for generation in range(generations):
        # Evaluate fitness (C2 values)
        fitness_scores = []
        for individual in population:
            c2 = compute_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_c2:
                best_c2 = c2
                best_solution = individual.copy()
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Check for stagnation
        if abs(best_c2 - prev_best) < 1e-8:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best_c2
        
        # Print progress every 100 generations
        if generation % 100 == 0:
            print(f"Generation {generation}: Best C2 = {best_c2:.6f}")
        
        # Early stopping if no improvement for 400 generations
        if stagnation_count > 400:
            print(f"Early stopping at generation {generation}")
            break
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(elite_size):
            new_population.append(sorted_population[i].copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection with larger tournament size for better pressure
            parent1 = tournament_selection(sorted_population, sorted_fitness, 15)
            parent2 = tournament_selection(sorted_population, sorted_fitness, 15)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate_individual(child, mutation_rate, generation)
            
            new_population.append(child)
        
        population = new_population[:population_size]
    
    return best_solution if best_solution is not None else []

def tournament_selection(population: List[List[float]], fitness_scores: List[float], k: int) -> List[float]:
    """Select individual using tournament selection with better pressure."""
    # Use a larger tournament size for better selection pressure
    tournament_indices = random.sample(range(len(population)), min(k, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index].copy()

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run optimization
    start_time = time.time()
    result = optimize_step_function()
    end_time = time.time()
    
    # Verify and report results
    if result:
        c2 = compute_c2(result)
        print(f"Optimization completed in {end_time - start_time:.2f}s")
        print(f"Best C2 found: {c2:.6f}")
        if c2 > 0.962:
            print("SUCCESS: Beat AlphaEvolve's benchmark!")
        else:
            print("Did not beat AlphaEvolve's benchmark.")
    else:
        # Fallback to simple approach
        result = [0.5] * 1000  # Simple uniform distribution
        
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
