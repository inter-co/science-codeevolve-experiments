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
warnings.filterwarnings('ignore')

# Global constants for optimization
MAX_STEPS = 3000  # Increased to allow better resolution
RESOLUTION_FACTOR = 100  # Higher resolution for better accuracy
POP_SIZE = 100
GENERATIONS = 300
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
    
    # Create step function - more accurate implementation
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
    
    # Manual convolution implementation for better control
    # Using proper discrete convolution approach for step functions
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
    # Trapezoidal integration for g^2
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
    points_per_step = max(30, RESOLUTION_FACTOR)  # higher resolution for better accuracy
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
        
        # Avoid division by zero
        if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
            return 0.0
            
        c2 = g_norm_2_squared / (g_norm_1 * g_norm_inf)
        return c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0
    except Exception:
        return 0.0

def create_initial_population(pop_size: int, min_steps: int = 500, max_steps: int = 2500) -> List[List[float]]:
    """Create initial population with diverse and promising step functions"""
    population = []
    for _ in range(pop_size):
        # Use a more informed approach to step count
        n_steps = random.randint(min_steps, min(max_steps, 2500))
        
        # Focus on mathematical patterns that tend to produce high C2
        pattern_type = random.choice([
            'multi_peak_optimized', 
            'bimodal_balanced', 
            'sine_wave_modulated',
            'spike_distributed',
            'gaussian_like',
            'smooth_gradient',
            'optimized_bimodal',
            'wavelet_pattern',
            'golden_ratio_peaks',
            'harmonic_combination'
        ])
        
        if pattern_type == 'multi_peak_optimized':
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
        elif pattern_type == 'bimodal_balanced':
            # Two dominant modes with careful balance
            heights = []
            for i in range(n_steps):
                # Alternate between two distinct patterns
                if i % 12 < 6:
                    heights.append(1.5 + 0.8 * np.sin(2 * np.pi * i / 20))
                else:
                    heights.append(0.8 + 0.4 * np.sin(2 * np.pi * i / 15))
        elif pattern_type == 'sine_wave_modulated':
            # Sine wave pattern with modulation for complexity
            heights = []
            for i in range(n_steps):
                base = 1.2 + 0.6 * np.sin(2 * np.pi * i / max(15, n_steps))
                # Add modulation to create interesting structure
                modulation = 0.3 * np.sin(4 * np.pi * i / max(10, n_steps))
                heights.append(max(0.0, base + modulation + random.uniform(-0.1, 0.1)))
        elif pattern_type == 'spike_distributed':
            # Distributed spikes with controlled heights
            heights = []
            for i in range(n_steps):
                # Create distributed high peaks with valleys
                if i % 20 < 3:  # Sharp spikes
                    heights.append(2.0 + 1.0 * random.random())
                elif i % 20 < 8:  # Medium peaks
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
        elif pattern_type == 'smooth_gradient':
            # Smooth gradient pattern that balances peaks and valleys
            heights = []
            for i in range(n_steps):
                # Create smooth variation with gradual changes
                x = i / (n_steps - 1)
                # Smooth transition between high and low values
                smooth_val = 1.0 + 0.5 * np.sin(2 * np.pi * x * 4) + 0.3 * np.sin(2 * np.pi * x * 8)
                heights.append(max(0.0, smooth_val + 0.1 * random.random()))
        elif pattern_type == 'optimized_bimodal':
            # Optimized bimodal pattern for better C2
            heights = []
            for i in range(n_steps):
                # Create two distinct peaks with appropriate spacing
                x = i / (n_steps - 1)
                # Two Gaussian peaks
                peak1 = 1.5 * np.exp(-((x - 0.3)**2) * 10)
                peak2 = 1.2 * np.exp(-((x - 0.7)**2) * 15)
                heights.append(max(0.0, peak1 + peak2 + 0.2 * random.random()))
        elif pattern_type == 'wavelet_pattern':
            # Wavelet-inspired pattern with oscillatory behavior
            heights = []
            for i in range(n_steps):
                # Create wavelet-like oscillations with decaying amplitude
                x = i / (n_steps - 1)
                oscillation = np.sin(10 * np.pi * x) * np.exp(-x * 5)
                heights.append(max(0.0, 1.0 + 0.5 * oscillation + 0.1 * random.random()))
        elif pattern_type == 'golden_ratio_peaks':
            # Golden ratio inspired peaks for mathematical elegance
            heights = []
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            for i in range(n_steps):
                # Create peaks at golden ratio positions
                position = (i * phi) % 1
                peak = 1.0 + 0.8 * np.exp(-((position - 0.5)**2) * 10)
                heights.append(max(0.0, peak + 0.1 * random.random()))
        elif pattern_type == 'harmonic_combination':
            # Harmonic combination of sine waves for structured complexity
            heights = []
            for i in range(n_steps):
                # Combination of harmonics for structured complexity
                x = i / (n_steps - 1)
                harmonic = 1.0 + 0.3 * np.sin(2 * np.pi * x * 3) + 0.2 * np.sin(2 * np.pi * x * 7) + 0.1 * np.sin(2 * np.pi * x * 11)
                heights.append(max(0.0, harmonic + 0.05 * random.random()))
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
    mutation_rate = 0.25 + 0.05 * (1 - generation/max_generations)
    
    # Apply mutations with more sophisticated strategy
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Use more sophisticated perturbations with adaptive variance
            # Prioritize small changes for fine-tuning but allow larger ones early on
            if generation < max_generations * 0.3:  # Early generations - more aggressive
                if random.random() < 0.6:  # Small change
                    delta = random.gauss(0, 0.1)
                elif random.random() < 0.9:  # Medium change
                    delta = random.gauss(0, 0.2)
                else:  # Large change
                    delta = random.gauss(0, 0.4)
            else:  # Later generations - more fine-tuned
                if random.random() < 0.7:  # Small change
                    delta = random.gauss(0, 0.05)
                elif random.random() < 0.9:  # Medium change
                    delta = random.gauss(0, 0.15)
                else:  # Large change
                    delta = random.gauss(0, 0.3)
            mutated[i] = max(0.0, mutated[i] + delta)
    
    # Occasionally adjust number of steps with better control
    if len(mutated) > 100 and random.random() < 0.02:
        # Remove a random element
        idx = random.randint(0, len(mutated) - 1)
        mutated.pop(idx)
    elif len(mutated) < 2500 and random.random() < 0.02:
        # Add a random element
        idx = random.randint(0, len(mutated))
        mutated.insert(idx, max(0.0, random.gauss(1.0, 0.6)))
    
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
    elif len(child) < 200:
        # Pad with average values if too short
        avg_val = np.mean([p for p in parent1 + parent2 if p > 0]) if parent1 or parent2 else 1.0
        while len(child) < 200:
            child.append(avg_val)
    
    return child

def improved_evolutionary_search(max_evaluations: int = 2500) -> List[float]:
    """Improved evolutionary algorithm with better convergence and more effective search"""
    # Time tracking
    start_time = time.time()
    
    # Initialize population with better starting conditions
    population = create_initial_population(POP_SIZE, min_steps=600, max_steps=2000)
    
    best_individual = None
    best_c2 = 0.0
    stagnation_counter = 0
    prev_best_c2 = -1
    
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
        
        # Check for stagnation
        if abs(best_c2 - prev_best_c2) < 1e-6:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        prev_best_c2 = best_c2
        
        # Stop if stagnating too long
        if stagnation_counter > 25:
            break
            
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep top performers
        elite_count = max(15, POP_SIZE // 3)
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
        if gen % 12 == 0 and gen > 0:
            # Introduce some fresh individuals occasionally
            fresh_individuals = create_initial_population(12, min_steps=500, max_steps=1800)
            population = population[:POP_SIZE-12] + fresh_individuals
    
    return best_individual if best_individual is not None else []

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using hybrid optimization"""
    # Time limit constraint
    start_time = time.time()
    
    try:
        # First run evolutionary search for global optimization
        result = improved_evolutionary_search(max_evaluations=2500)
        
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
                        num_changes = min(60, len(candidate) // 2)
                        for _ in range(num_changes):
                            idx = random.randint(0, len(candidate) - 1)
                            # Mix of small and large changes with more strategic approach
                            if random.random() < 0.4:
                                delta = random.gauss(0, 0.03)
                            elif random.random() < 0.7:
                                delta = random.gauss(0, 0.08)
                            elif random.random() < 0.9:
                                delta = random.gauss(0, 0.15)
                            else:
                                delta = random.gauss(0, 0.25)
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
                
        # Final refinement using direct optimization if time allows
        if time.time() - start_time < TIME_LIMIT - 3:
            try:
                # Try direct optimization on promising results
                if len(result) > 50:
                    # Use a more targeted approach with smarter optimization
                    refined_result = []
                    for i in range(len(result)):
                        # Try to optimize each element individually
                        base_val = result[i]
                        best_val = base_val
                        best_c2 = evaluate_c2(result[:i] + [base_val] + result[i+1:])
                        
                        # Test nearby values in a more strategic way
                        test_values = [base_val * 0.7, base_val * 0.8, base_val * 0.9, base_val, base_val * 1.1, base_val * 1.2, base_val * 1.3]
                        for val in test_values:
                            temp_result = result[:i] + [val] + result[i+1:]
                            c2 = evaluate_c2(temp_result)
                            if c2 > best_c2:
                                best_c2 = c2
                                best_val = val
                        
                        refined_result.append(best_val)
                    
                    final_c2 = evaluate_c2(refined_result)
                    original_c2 = evaluate_c2(result)
                    if final_c2 > original_c2:
                        result = refined_result
                        
            except Exception:
                pass
                
        return result
    except Exception:
        # Fallback to a proven good pattern with mathematical sophistication
        n = 2000
        # Create an optimized pattern based on mathematical insights for high C2
        heights = []
        for i in range(n):
            # Create a pattern that maximizes the ratio of ||g||₂² to (||g||₁ · ||g||∞)
            # This uses the principle that balanced, smooth variations work well
            # Create peaks and valleys with mathematical precision
            if i % 15 < 4:  # Prominent peaks
                heights.append(1.8 + 0.6 * np.sin(2 * np.pi * i / 20))
            elif i % 15 < 10:  # Medium peaks  
                heights.append(1.2 + 0.4 * np.sin(2 * np.pi * i / 18))
            else:  # Valleys
                heights.append(0.6 + 0.3 * np.sin(2 * np.pi * i / 12))
        
        # Ensure all non-negative and add some randomization for robustness
        heights = [max(0.0, h + random.uniform(-0.05, 0.05)) for h in heights]
        return heights

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
