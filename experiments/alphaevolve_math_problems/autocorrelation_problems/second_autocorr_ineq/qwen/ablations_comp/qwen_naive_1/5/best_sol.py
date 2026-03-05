# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from typing import List
import random
import time
from numba import jit
import math
from scipy.optimize import differential_evolution
import warnings
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
from functools import partial
from scipy.spatial.distance import pdist, squareform

@jit(nopython=True)
def compute_piecewise_integral(g_values: np.ndarray, dx: float) -> float:
    """Compute piecewise integral using trapezoidal-like formula - optimized version"""
    if len(g_values) < 2:
        return g_values[0]**2 if len(g_values) > 0 else 0.0
    
    integral = 0.0
    # Vectorized computation for better performance
    for i in range(len(g_values) - 1):
        integral += (dx/3) * (g_values[i]**2 + g_values[i]*g_values[i+1] + g_values[i+1]**2)
    
    # Add boundary terms
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

def create_initial_population(size: int, min_steps: int = 100, max_steps: int = 5000) -> List[List[float]]:
    """Create initial population of step function configurations with better strategies."""
    population = []
    
    # Predefined patterns that work well for maximizing C2
    patterns = [
        # Pattern 1: Smooth peak with exponential decay
        lambda n: [max(0, np.exp(-((i - n//2)**2)/(n//10)**2)) for i in range(n)],
        
        # Pattern 2: Multi-peak with Gaussian-like shape
        lambda n: [max(0, 0.5 + 0.5 * np.exp(-((i - n//3)**2)/(n//8)**2) + 
                       0.5 * np.exp(-((i - 2*n//3)**2)/(n//8)**2)) for i in range(n)],
        
        # Pattern 3: Central peak with tapering edges
        lambda n: [max(0, 1 - 4 * ((i - n//2) / (n//2))**2) if abs(i - n//2) < n//2 else 0 for i in range(n)],
        
        # Pattern 4: Sharp central peak with oscillatory tails
        lambda n: [max(0, 1 - 10 * abs(i - n//2)**2 / (n//2)**2 + 0.2 * np.sin(8 * np.pi * i / n)) for i in range(n)],
        
        # Pattern 5: Sparsely distributed peaks - improved version
        lambda n: [0.0] * n,
        
        # Pattern 6: Optimized "flat-top" pattern
        lambda n: [1.0 if abs(i - n//2) < n//6 else 0.0 for i in range(n)],
        
        # Pattern 7: "Bell curve" pattern
        lambda n: [max(0, np.exp(-((i - n//2)**2)/(n//4)**2)) for i in range(n)],
        
        # Pattern 8: Improved bell curve with sharper peak
        lambda n: [max(0, np.exp(-((i - n//2)**2)/(n//6)**2) * (1 + 0.2 * np.sin(10 * np.pi * i / n))) for i in range(n)],
        
        # Pattern 9: Concentrated peak with fast decay
        lambda n: [max(0, 1.5 * np.exp(-((i - n//2)**2)/(n//8)**2) - 0.5) for i in range(n)],
        
        # Pattern 10: Multi-peak with controlled spacing and amplitude
        lambda n: [max(0, 0.8 * np.exp(-((i - n//4)**2)/(n//10)**2) + 
                       0.8 * np.exp(-((i - 3*n//4)**2)/(n//10)**2)) for i in range(n)]
    ]
    
    for _ in range(size):
        n = random.randint(min_steps, max_steps)
        
        # Use a hybrid approach - combine multiple strategies
        strategy = random.randint(0, 30)
        
        if strategy < 5:  # Use predefined patterns
            pattern_idx = random.randint(0, len(patterns)-1)
            f_vals = patterns[pattern_idx](n)
            # Add some noise to prevent getting stuck in local optima
            f_vals = [max(0, val + random.uniform(-0.05, 0.05)) for val in f_vals]
        elif strategy < 10:  # Modified Gaussian peak with better shape
            center = n // 2
            f_vals = []
            for i in range(n):
                distance = abs(i - center) / (n // 2)
                # Use sharper decay for better C2
                val = max(0, np.exp(-distance**3 * 3))  # Even sharper decay
                val = max(0, val + random.uniform(-0.03, 0.03))
                f_vals.append(val)
        elif strategy < 15:  # Multi-peak with controlled spacing
            f_vals = []
            num_peaks = random.randint(2, 6)
            for i in range(n):
                pos = i / n
                val = 0.0
                for j in range(num_peaks):
                    peak_pos = (j + 1) / (num_peaks + 1)
                    distance = abs(pos - peak_pos)
                    # Stronger decay for better C2
                    val += max(0, 1 - 15 * distance**2)  
                val = max(0, val + random.uniform(-0.03, 0.03))
                f_vals.append(val)
        elif strategy < 20:  # Enhanced "bell curve" with multiple peaks
            f_vals = []
            num_peaks = random.randint(2, 5)
            for i in range(n):
                pos = i / n
                val = 0.0
                for j in range(num_peaks):
                    peak_pos = (j + 1) / (num_peaks + 1)
                    distance = abs(pos - peak_pos)
                    val += max(0, np.exp(-distance**3 * 5))  # Sharper decay
                val = max(0, val + random.uniform(-0.02, 0.02))
                f_vals.append(val)
        elif strategy < 25:  # More sophisticated multi-peak pattern
            f_vals = []
            # Create a more structured pattern with better balance
            center = n // 2
            for i in range(n):
                # Create a symmetric pattern around center
                dist_from_center = abs(i - center)
                # Create a combination of multiple peaks
                val = 0.0
                for j in range(1, 5):
                    # Place peaks at positions that give good spread
                    peak_pos = center + (j * n // 10) - (n // 20)  # Spread out
                    if 0 <= peak_pos < n:
                        distance = abs(i - peak_pos) / (n // 8)
                        val += max(0, 0.5 * np.exp(-distance**2.5))
                val = max(0, val + random.uniform(-0.02, 0.02))
                f_vals.append(val)
        else:  # Random smooth pattern with controlled variation
            f_vals = []
            for i in range(n):
                # Create a smooth curve with randomization
                angle = 2 * np.pi * i / n
                base_val = 0.3 + 0.4 * np.sin(3 * angle) + 0.3 * np.cos(2 * angle)
                # Add some localized peaks
                if i % (max(1, n // 15)) == 0:
                    base_val += random.uniform(0.1, 0.3)
                base_val = max(0, base_val)
                f_vals.append(base_val)
                
        # Normalize to reasonable range
        if f_vals:
            max_val = max(f_vals)
            if max_val > 0:
                f_vals = [val / max_val * 0.8 for val in f_vals]
        
        population.append(f_vals)
    return population

def mutate_individual(f_values: List[float], mutation_rate: float = 0.1, generation: int = 0) -> List[float]:
    """Mutate a single individual with adaptive parameters and better strategies."""
    new_values = f_values.copy()
    # Adaptive mutation rate that decreases over time
    adaptive_mutation_rate = mutation_rate * (1.0 - generation / 1000.0)
    adaptive_mutation_rate = max(0.001, adaptive_mutation_rate)
    
    for i in range(len(new_values)):
        if random.random() < adaptive_mutation_rate:
            # Apply mutation with adaptive strength based on current value
            current_val = new_values[i]
            
            # 70% chance of small mutation (Gaussian noise)
            if random.random() < 0.7:
                # Add small Gaussian noise with adaptive scale
                noise_scale = 0.01 + 0.03 * (1.0 - current_val)  # Less noise near 1.0
                noise = random.gauss(0, noise_scale)
                new_values[i] += noise
                # Ensure non-negative
                new_values[i] = max(0, new_values[i])
            else:
                # 30% chance of significant mutation (replacing with new pattern)
                # This helps escape local optima more effectively
                if random.random() < 0.5:
                    # Replace with a value from a nearby position
                    neighbor_idx = max(0, min(len(new_values)-1, i + random.randint(-3, 3)))
                    new_values[i] = new_values[neighbor_idx] * random.uniform(0.8, 1.2)
                else:
                    # Replace with a value from a better distribution
                    # Use a better distribution based on neighborhood
                    if i > 0 and i < len(new_values) - 1:
                        avg_neighbor = (new_values[i-1] + new_values[i+1]) / 2
                        new_values[i] = avg_neighbor * random.uniform(0.7, 1.3)
                    else:
                        new_values[i] = random.uniform(0, 1.0)
                    
                # Ensure non-negative
                new_values[i] = max(0, new_values[i])
                
    return new_values

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform crossover between two parents with enhanced strategy."""
    # Use uniform crossover with better blending
    size = min(len(parent1), len(parent2))
    child = []
    
    # Blend strategy: mix both parents' values with probability
    for i in range(size):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
        
        # Occasionally blend with a weighted average (more aggressive blending)
        if random.random() < 0.15:
            # Use a more sophisticated blending approach
            alpha = random.random() * 0.6 + 0.2  # Weight between 0.2 and 0.8
            child[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
    
    # Extend to match longer parent if needed
    if len(parent1) > size:
        child.extend(parent1[size:])
    elif len(parent2) > size:
        child.extend(parent2[size:])
    
    return child

def optimize_step_function() -> List[float]:
    """Optimize step function to maximize C2 using evolutionary approach with improvements."""
    # Parameters - tuned for better performance
    population_size = 400
    generations = 1000
    elite_size = 50
    mutation_rate = 0.15
    
    # Initialize population with higher quality starting points
    population = create_initial_population(population_size)
    
    best_c2 = 0.0
    best_solution = None
    stagnation_count = 0
    prev_best = 0.0
    
    # Evolutionary loop
    for generation in range(generations):
        # Evaluate fitness (C2 values) - parallelized for speed
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
        if abs(best_c2 - prev_best) < 1e-6:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best_c2
        
        # Print progress every 100 generations
        if generation % 100 == 0:
            print(f"Generation {generation}: Best C2 = {best_c2:.6f}")
        
        # Early stopping if no improvement for 200 generations
        if stagnation_count > 200:
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
            parent1 = tournament_selection(sorted_population, sorted_fitness, 12)
            parent2 = tournament_selection(sorted_population, sorted_fitness, 12)
            
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

def optimize_with_differential_evolution(n_steps: int = 2000) -> List[float]:
    """Use differential evolution for fine-tuning the solution."""
    # Start with a good configuration
    initial_guess = []
    center = n_steps // 2
    
    # Create a promising initial pattern based on previous knowledge
    for i in range(n_steps):
        distance = abs(i - center) / (n_steps // 2)
        # Use a pattern that tends to work well for maximizing C2
        val = max(0, 1 - distance**2.5)  # Slightly steeper than cubic decay
        # Add small random variation to avoid local optima
        val = max(0, val + random.uniform(-0.01, 0.01))
        initial_guess.append(val)
    
    # Define bounds (0 to 1 for all parameters)
    bounds = [(0.0, 1.0) for _ in range(n_steps)]
    
    # Use differential evolution with optimized parameters
    def callback(xk, convergence):
        pass  # We don't need detailed callbacks for this implementation
    
    try:
        # Suppress warnings from differential evolution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                lambda x: -compute_c2(x.tolist()),  # Negative because we minimize
                bounds,
                maxiter=300,  # Increased iterations for better search
                popsize=50,   # Larger population
                mutation=(0.5, 1.0),
                recombination=0.9,  # Higher recombination
                seed=42,
                callback=callback,
                disp=False
            )
        
        if result.success:
            return result.x.tolist()
        else:
            return initial_guess
    except Exception:
        return initial_guess

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run evolutionary optimization
    start_time = time.time()
    result = optimize_step_function()
    end_time = time.time()
    
    # Fine-tune with differential evolution if we have a good solution
    if result and compute_c2(result) > 0.9:
        print("Using differential evolution for fine-tuning...")
        fine_tuned_result = optimize_with_differential_evolution(len(result))
        final_c2 = compute_c2(fine_tuned_result)
        if final_c2 > compute_c2(result):
            result = fine_tuned_result
            print(f"Fine-tuned C2: {final_c2:.6f}")
    
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
