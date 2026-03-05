# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution, minimize
import random
from typing import List
import time
import warnings
import math
import numba
from numba import jit
from scipy.interpolate import interp1d
import multiprocessing as mp
from functools import partial
import nevergrad as ng
import optuna
warnings.filterwarnings('ignore')

# Global constants for optimization
MAX_STEPS = 3000  # Reduced to manage memory better
RESOLUTION_FACTOR = 200  # Higher resolution for better accuracy  
POP_SIZE = 100
GENERATIONS = 200
TIME_LIMIT = 55  # seconds

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: List[float], n_steps: int, points_per_step: int) -> tuple:
    """
    Fast computation of autoconvolution norms using Numba JIT compilation
    """
    if n_steps == 0:
        return 0.0, 0.0, 0.0
    
    # Create domain
    total_points = n_steps * points_per_step + 1
    dx = 0.5 / (total_points - 1)
    
    # Create step function efficiently
    f = np.zeros(total_points)
    for i in range(n_steps):
        start_idx = i * points_per_step
        end_idx = (i + 1) * points_per_step
        if end_idx > len(f):
            end_idx = len(f)
        f[start_idx:end_idx] = f_values[i]
    
    # Compute autoconvolution more accurately using manual convolution
    # This avoids issues with scipy convolution for our specific case
    g = np.zeros(total_points)
    
    # Manual convolution implementation for better control - optimized version
    # Using proper discrete convolution with correct indexing
    for i in range(n_steps):
        for j in range(n_steps):
            # For discrete convolution, we map indices appropriately
            pos = i + j
            if 0 <= pos < n_steps:
                # Simple rectangular convolution approximation
                g[pos] += f_values[i] * f_values[j] * dx
    
    # Normalize and compute norms using proper trapezoidal integration
    # For ||g||₂² using trapezoidal rule (more accurate than simple sum)
    g_squared = g * g
    # Trapezoidal integration for g^2 - using the correct trapezoidal formula
    g_norm_2_squared = np.sum((g_squared[:-1] + g_squared[1:]) * dx / 2)
    
    # ||g||₁ (L1 norm)
    g_norm_1 = np.sum(np.abs(g)) * dx
    
    # ||g||∞ (infinity norm)
    g_norm_inf = np.max(np.abs(g))
    
    return g_norm_2_squared, g_norm_1, g_norm_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Create step function on [-1/4, 1/4]
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Limit maximum size to prevent memory issues
    if n > MAX_STEPS:
        n = MAX_STEPS
        f_values = f_values[:n]
    
    # Define domain - use a reasonable resolution
    domain_size = 0.5  # from -0.25 to 0.25
    points_per_step = max(100, RESOLUTION_FACTOR)  # higher resolution for better accuracy
    total_points = n * points_per_step + 1
    
    # Create evenly spaced points
    x = np.linspace(-0.25, 0.25, total_points)
    dx = x[1] - x[0]
    
    # Create step function efficiently using vectorized operations
    f = np.zeros_like(x)
    
    # Vectorized assignment of step values - much more efficient
    for i in range(n):
        start_idx = i * points_per_step
        end_idx = (i + 1) * points_per_step
        if end_idx > len(f):  # Handle boundary case
            end_idx = len(f)
        f[start_idx:end_idx] = f_values[i]
    
    # Compute autoconvolution g = f * f using optimized convolution
    try:
        # Use fast convolution with proper handling
        g = signal.convolve(f, f, mode='full')
        
        # Extract the valid portion (center part) - more precise indexing
        center_idx = len(f) - 1
        g = g[center_idx:]  # Keep only the valid portion
        
        # Trim g to match the domain properly
        g = g[:len(x)]
        
        # Compute norms using more accurate trapezoidal integration
        # For ||g||₂² using trapezoidal approximation (piecewise linear)
        # Use correct trapezoidal rule for integration of g^2
        g_squared = g * g
        # Proper trapezoidal integration for g^2 - more accurate version
        g_norm_2_squared = np.sum((g_squared[:-1] + g_squared[1:]) * dx / 2)
        
        # ||g||₁ (L1 norm) - corrected for exact integration
        g_norm_1 = np.sum(np.abs(g)) * dx
        
        # ||g||∞ (infinity norm)
        g_norm_inf = np.max(np.abs(g))
        
        return g_norm_2_squared, g_norm_1, g_norm_inf
    except Exception as e:
        # Fallback for computational errors
        return 0.0, 0.0, 0.0

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C2 = ||g||₂² / (||g||₁ · ||g||∞)
    """
    try:
        g_norm_2_squared, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero with a more robust check
        if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
            return 0.0
            
        c2 = g_norm_2_squared / (g_norm_1 * g_norm_inf)
        return c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0
    except Exception:
        return 0.0

def create_initial_population(pop_size: int, min_steps: int = 500, max_steps: int = 2000) -> List[List[float]]:
    """Create initial population with diverse and promising step functions"""
    population = []
    for _ in range(pop_size):
        # Use a more informed approach to step count
        n_steps = random.randint(min_steps, min(max_steps, 2000))
        
        # Focus on mathematical patterns that tend to produce high C2
        pattern_type = random.choice([
            'optimized_multipeak', 
            'balanced_bimodal', 
            'sine_wave_modulated',
            'spike_distribution',
            'gaussian_like',
            'wavelet_like',
            'harmonic_superposition',
            'multifractal_pattern',
            'logarithmic_decay',
            'inverse_power_pattern',
            'fractal_like',
            'golden_ratio_pattern',
            'butterfly_pattern'
        ])
        
        if pattern_type == 'optimized_multipeak':
            # Create optimized multi-peak pattern with mathematical balance
            heights = []
            for i in range(n_steps):
                # Create peaks with strategic spacing and varying heights
                phase = 2 * np.pi * i / n_steps
                # Combine sine and cosine components for complex pattern
                peak_height = 1.0 + 0.5 * np.sin(phase * 3) + 0.3 * np.cos(phase * 5)
                # Add some randomness but keep it reasonable
                random_factor = 0.8 + 0.4 * random.random()
                heights.append(max(0.0, peak_height * random_factor))
        elif pattern_type == 'balanced_bimodal':
            # Two dominant modes with careful balance
            heights = []
            for i in range(n_steps):
                # Alternate between two distinct patterns
                if i % 12 < 6:
                    heights.append(1.5 + 0.8 * np.sin(2 * np.pi * i / 15))
                else:
                    heights.append(0.8 + 0.4 * np.sin(2 * np.pi * i / 12))
        elif pattern_type == 'sine_wave_modulated':
            # Sine wave pattern with modulation for complexity
            heights = []
            for i in range(n_steps):
                base = 1.2 + 0.6 * np.sin(2 * np.pi * i / max(15, n_steps))
                # Add modulation to create interesting structure
                modulation = 0.3 * np.sin(4 * np.pi * i / max(10, n_steps))
                heights.append(max(0.0, base + modulation + random.uniform(-0.1, 0.1)))
        elif pattern_type == 'spike_distribution':
            # Distributed spikes with controlled heights
            heights = []
            for i in range(n_steps):
                # Create distributed high peaks with valleys
                if i % 18 < 3:  # Sharp spikes
                    heights.append(2.0 + 1.0 * random.random())
                elif i % 18 < 10:  # Medium peaks
                    heights.append(1.0 + 0.5 * random.random())
                else:  # Low valleys
                    heights.append(0.3 + 0.3 * random.random())
        elif pattern_type == 'gaussian_like':
            # Gaussian-like pattern with peak in center and decay
            heights = []
            center = n_steps // 2
            for i in range(n_steps):
                # Gaussian-like shape with some randomness
                dist = abs(i - center) / (n_steps / 2)
                gaussian_val = np.exp(-dist**2 * 2)
                heights.append(max(0.0, gaussian_val + 0.2 * random.random()))
        elif pattern_type == 'wavelet_like':
            # Wavelet-like pattern for structured variation
            heights = []
            for i in range(n_steps):
                # Create wavelet-like oscillation with decay
                freq = 2 * np.pi * i / n_steps
                wavelet_component = np.sin(freq * 5) * np.exp(-freq**2 * 0.5)
                heights.append(max(0.0, 1.0 + 0.8 * wavelet_component + 0.2 * random.random()))
        elif pattern_type == 'harmonic_superposition':
            # Superposition of harmonics
            heights = []
            for i in range(n_steps):
                # Combine multiple sine waves with different frequencies
                h1 = 0.8 + 0.4 * np.sin(2 * np.pi * i / 20)
                h2 = 0.6 + 0.3 * np.sin(4 * np.pi * i / 25)
                h3 = 0.4 + 0.2 * np.sin(6 * np.pi * i / 30)
                h4 = 0.3 + 0.1 * np.sin(8 * np.pi * i / 35)
                heights.append(max(0.0, h1 + h2 + h3 + h4 + 0.1 * random.random()))
        elif pattern_type == 'multifractal_pattern':
            # Multifractal-inspired pattern
            heights = []
            for i in range(n_steps):
                # Create multifractal-like structure with varying exponents
                power = 1.5 + 0.5 * np.sin(2 * np.pi * i / 50)
                fractal_component = np.power(1.0 + 0.5 * np.sin(2 * np.pi * i / 30), power)
                heights.append(max(0.0, 1.0 + 0.8 * fractal_component + 0.2 * random.random()))
        elif pattern_type == 'logarithmic_decay':
            # Logarithmic decay pattern
            heights = []
            for i in range(n_steps):
                # Logarithmic decay with oscillation
                log_component = 1.0 / (1.0 + 0.1 * i)
                oscillation = 0.3 * np.sin(2 * np.pi * i / 20)
                heights.append(max(0.0, log_component + oscillation + 0.1 * random.random()))
        elif pattern_type == 'inverse_power_pattern':
            # Inverse power law pattern
            heights = []
            for i in range(n_steps):
                # Inverse power pattern with noise
                power = 2.0 + 0.5 * np.sin(2 * np.pi * i / 40)
                inverse_component = 1.0 / (1.0 + 0.05 * i**power)
                heights.append(max(0.0, inverse_component + 0.2 * random.random()))
        elif pattern_type == 'fractal_like':
            # Fractal-like self-similar pattern
            heights = []
            for i in range(n_steps):
                # Create fractal-like pattern with recursive properties
                fractal_val = 1.0 + 0.5 * np.sin(2 * np.pi * i / 25) * np.cos(4 * np.pi * i / 30)
                heights.append(max(0.0, fractal_val + 0.3 * random.random()))
        elif pattern_type == 'golden_ratio_pattern':
            # Golden ratio inspired pattern
            heights = []
            phi = (1 + np.sqrt(5)) / 2
            for i in range(n_steps):
                # Golden ratio pattern with periodicity
                period = 20
                phase = 2 * np.pi * i / period
                # Create peaks at golden ratio positions
                golden_peak = 1.0 + 0.7 * np.sin(phase * phi)
                heights.append(max(0.0, golden_peak + 0.2 * random.random()))
        elif pattern_type == 'butterfly_pattern':
            # Butterfly-inspired pattern with symmetric peaks
            heights = []
            for i in range(n_steps):
                # Create butterfly wing like structure
                x = 2 * i / n_steps - 1  # Scale to [-1, 1]
                butterfly = 1.0 + 0.5 * np.exp(-x**2 * 10) * np.cos(3 * np.pi * x)
                heights.append(max(0.0, butterfly + 0.1 * random.random()))
        else:  # uniform
            # Uniform distribution with some variation
            heights = [1.0 + 0.3 * np.sin(2 * np.pi * i / n_steps) for i in range(n_steps)]
        
        # Ensure non-negative and add some randomness for robustness
        heights = [max(0.0, h + random.uniform(-0.1, 0.1)) for h in heights]
        population.append(heights)
    
    return population

def mutate_individual(individual: List[float], generation: int, max_generations: int) -> List[float]:
    """Apply mutation to individual with adaptive rate and better strategy"""
    mutated = individual.copy()
    
    # Adaptive mutation rate based on generation (decrease over time)
    mutation_rate = 0.4 + 0.2 * (1 - generation/max_generations)
    
    # Apply mutations
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Use more sophisticated perturbations
            if random.random() < 0.3:  # Small change
                delta = random.gauss(0, 0.01)
            elif random.random() < 0.6:  # Medium change
                delta = random.gauss(0, 0.05)
            else:  # Large change for exploration
                delta = random.gauss(0, 0.15)
            mutated[i] = max(0.0, mutated[i] + delta)
    
    # Occasionally adjust number of steps with better control
    if len(mutated) > 100 and random.random() < 0.01:
        # Remove a random element
        idx = random.randint(0, len(mutated) - 1)
        mutated.pop(idx)
    elif len(mutated) < 2000 and random.random() < 0.01:
        # Add a random element
        idx = random.randint(0, len(mutated))
        mutated.insert(idx, max(0.0, random.gauss(1.0, 0.3)))
    
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Create offspring through crossover with better handling"""
    # Single point crossover with size adjustment
    max_len = max(len(parent1), len(parent2))
    
    # Determine crossover point
    cross_point = random.randint(0, min(len(parent1), len(parent2)))
    
    # Create child
    child = parent1[:cross_point] + parent2[cross_point:]
    
    # Ensure reasonable size
    if len(child) > MAX_STEPS:
        child = child[:MAX_STEPS]
    elif len(child) < 50:
        # Pad with average values if too short
        avg_val = np.mean([p for p in parent1 + parent2 if p > 0]) if parent1 or parent2 else 1.0
        while len(child) < 50:
            child.append(avg_val)
    
    return child

def adaptive_evolutionary_search(max_evaluations: int = 1000) -> List[float]:
    """Improved evolutionary algorithm with adaptive parameters and better selection"""
    # Time tracking
    start_time = time.time()
    
    # Initialize population with better starting conditions
    population = create_initial_population(POP_SIZE, min_steps=500, max_steps=1800)
    
    best_individual = None
    best_c2 = 0.0
    
    # Evolutionary loop
    for gen in range(GENERATIONS):
        # Early termination check
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Evaluate fitness with parallel processing for efficiency
        fitness_scores = []
        for individual in population:
            c2 = evaluate_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_c2:
                best_c2 = c2
                best_individual = individual.copy()
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep top performers
        elite_count = max(15, POP_SIZE // 4)
        selected = sorted_population[:elite_count]
        
        # Generate new population from elites
        new_population = selected.copy()
        
        # Fill rest with offspring
        while len(new_population) < POP_SIZE:
            # Tournament selection for parents (larger tournament size)
            parent1 = sorted_population[random.randint(0, elite_count-1)]
            parent2 = sorted_population[random.randint(0, elite_count-1)]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate_individual(child, gen, GENERATIONS)
            
            new_population.append(child)
        
        population = new_population[:POP_SIZE]
        
        # Periodic diversity maintenance
        if gen % 20 == 0 and gen > 0:
            # Introduce some fresh individuals occasionally
            fresh_individuals = create_initial_population(15, min_steps=500, max_steps=1800)
            population = population[:POP_SIZE-15] + fresh_individuals
    
    return best_individual if best_individual is not None else []

def bayesian_optimization_approach(initial_solution: List[float]) -> List[float]:
    """Use Bayesian optimization for fine-tuning the best solution"""
    def objective(trial):
        # Create a slightly modified version of the input
        params = []
        for i, val in enumerate(initial_solution):
            # Modify each parameter with trial-based adjustments
            if i < len(initial_solution):
                # Adjust by ±20% of the original value
                factor = trial.suggest_float(f'factor_{i}', 0.8, 1.2)
                params.append(max(0.0, val * factor))
            else:
                params.append(val)
        return -evaluate_c2(params)  # Negative because we minimize
    
    # Use Optuna for Bayesian optimization
    study = optuna.create_study(direction='minimize')
    study.enqueue_trial({'factor_0': 1.0})  # Initial point
    
    # Run optimization with limited trials to save time
    try:
        study.optimize(objective, n_trials=50, timeout=10)
        best_params = study.best_params
        result = []
        for i, val in enumerate(initial_solution):
            if f'factor_{i}' in best_params:
                factor = best_params[f'factor_{i}']
                result.append(max(0.0, val * factor))
            else:
                result.append(val)
        return result
    except Exception:
        return initial_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using hybrid optimization"""
    # Time limit constraint
    start_time = time.time()
    
    try:
        # First run evolutionary search for global optimization
        result = adaptive_evolutionary_search(max_evaluations=1000)
        
        # Quick local refinement with enhanced hill climbing
        if time.time() - start_time < TIME_LIMIT - 5:
            try:
                # Convert to numpy array for easier manipulation
                if len(result) > 0:
                    current_best = result.copy()
                    current_c2 = evaluate_c2(current_best)
                    
                    # Enhanced hill climbing with systematic search
                    for _ in range(1000):  # More iterations for better refinement
                        if time.time() - start_time > TIME_LIMIT - 2:
                            break
                            
                        candidate = current_best.copy()
                        # Modify multiple random elements with varied changes
                        num_changes = min(50, len(candidate) // 3)
                        for _ in range(num_changes):
                            idx = random.randint(0, len(candidate) - 1)
                            # Mix of small and large changes with more strategic approach
                            if random.random() < 0.5:
                                delta = random.gauss(0, 0.03)
                            else:
                                delta = random.gauss(0, 0.10)
                            candidate[idx] = max(0.0, candidate[idx] + delta)
                        
                        candidate_c2 = evaluate_c2(candidate)
                        if candidate_c2 > current_c2:
                            current_best = candidate
                            current_c2 = candidate_c2
                    
                    # If improvement found, use it
                    if evaluate_c2(current_best) > evaluate_c2(result):
                        result = current_best
                        
            except Exception:
                pass
                
        # Final refinement using Bayesian optimization if time allows
        if time.time() - start_time < TIME_LIMIT - 3:
            try:
                # Try Bayesian optimization on promising results
                if len(result) > 50:
                    result = bayesian_optimization_approach(result)
            except Exception:
                pass
                
        return result
    except Exception:
        # Fallback to a proven good pattern with mathematical sophistication
        n = 1500
        # Create an optimized pattern based on mathematical insights for high C2
        heights = []
        # Create a more sophisticated pattern with multiple peaks and valleys
        for i in range(n):
            # Create a pattern that maximizes the ratio of ||g||₂² to (||g||₁ · ||g||∞)
            # This uses the principle that balanced, smooth variations work well
            # Use more mathematical approach to create optimal structure
            if i % 25 < 4:  # Sharp peaks
                heights.append(2.0 + 0.8 * np.sin(2 * np.pi * i / 30))
            elif i % 25 < 12:  # Moderate peaks
                heights.append(1.5 + 0.6 * np.sin(2 * np.pi * i / 20))
            elif i % 25 < 20:  # Valleys
                heights.append(0.8 + 0.4 * np.sin(2 * np.pi * i / 15))
            else:  # Flat regions
                heights.append(0.5 + 0.2 * np.sin(2 * np.pi * i / 10))
        
        # Ensure all non-negative and add some randomization for robustness
        heights = [max(0.0, h + random.uniform(-0.05, 0.05)) for h in heights]
        return heights

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
