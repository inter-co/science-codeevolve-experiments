# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List
import time

def compute_autoconvolution_norms(f_values: List[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)
    
    # Create the step function on [-1/4, 1/4] with appropriate spacing
    n_steps = len(f)
    if n_steps == 0:
        return 0.0, 0.0, 0.0
    
    # Step size
    dx = 0.5 / n_steps  # Total range is 1/2, so step size is 0.5/n_steps
    
    # Compute autoconvolution g = f * f using proper discrete convolution
    g = signal.convolve(f, f, mode='full')
    
    # Trim to appropriate size (should be 2*n-1)
    g = g[n_steps-1:2*n_steps-1]
    
    # Compute norms properly
    g_abs = np.abs(g)
    
    # ||g||₂² (L2 norm squared) - exact piecewise linear integration
    # For trapezoidal-like piecewise linear integration over segments:
    # contribution = (dx/3)(y1² + y1*y2 + y2²) for consecutive points
    if len(g_abs) >= 2:
        g_l2_squared = 0.0
        for i in range(len(g_abs) - 1):
            g_l2_squared += (g_abs[i]**2 + g_abs[i]*g_abs[i+1] + g_abs[i+1]**2) / 3.0
        g_l2_squared *= dx
    else:
        g_l2_squared = 0.0
    
    # ||g||₁ (L1 norm) - sum of absolute values times step size
    norm_1 = np.sum(g_abs) * dx
    
    # ||g||∞ (infinity norm)
    norm_inf = np.max(g_abs)
    
    return g_l2_squared, norm_1, norm_inf

def evaluate_c2(f_values: List[float]) -> float:
    """Evaluate C2 value for given step function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def construct_function() -> List[float]:
    """
    Optimized evolutionary algorithm that focuses on the most effective peak configurations
    """
    # Parameters optimized for performance within 60-second limit
    n_steps = 450  # Slightly increased for better resolution
    max_generations = 200  # More generations for better convergence
    population_size = 120  # Larger population for better exploration
    
    # Start timing for performance monitoring
    start_time = time.time()
    
    # Initialize population with highly effective strategies
    population = []
    
    # Strategy 1: Very focused symmetric peak pattern (based on analysis of best performers)
    for _ in range(population_size // 3):
        f_values = []
        half = n_steps // 2
        # Create very sharp central peak with gradual tapering
        for i in range(half):
            # Central peak region - very high values
            if abs(i - half//2) < half//10:
                val = random.uniform(0.95, 1.0)
            elif abs(i - half//2) < half//5:
                val = random.uniform(0.8, 0.95)
            else:
                val = random.uniform(0.2, 0.6)
            f_values.append(val)
        # Make symmetric
        f_values.extend(reversed(f_values[:half]))
        if n_steps % 2 == 1:
            f_values.insert(half, random.uniform(0.7, 0.95))
        population.append(f_values)
    
    # Strategy 2: Smooth structured random (for diversity)
    for _ in range(population_size // 3):
        f_values = []
        for i in range(n_steps):
            if i == 0:
                val = random.uniform(0.3, 0.7)
            else:
                # Correlated with previous value but with controlled variation
                prev_val = f_values[-1]
                delta = (random.random() - 0.5) * 0.25
                val = max(0, min(1, prev_val + delta))
            f_values.append(val)
        population.append(f_values)
    
    # Strategy 3: Multi-peak with strategic spacing (inspired by high-performing patterns)
    for _ in range(population_size // 3):
        f_values = [0.0] * n_steps
        # Place multiple peaks strategically
        num_peaks = 3
        peak_positions = [i * n_steps // (num_peaks + 1) for i in range(1, num_peaks + 1)]
        for pos in peak_positions:
            if pos < n_steps:
                f_values[pos] = random.uniform(0.8, 1.0)
        # Smooth the peaks
        for i in range(n_steps):
            if f_values[i] > 0:
                # Apply smoothing with less noise than before
                smoothed = 0.0
                count = 0
                for j in range(max(0, i-2), min(n_steps, i+3)):
                    if f_values[j] > 0:
                        smoothed += f_values[j]
                        count += 1
                if count > 0:
                    f_values[i] = max(0, smoothed / count + (random.random() - 0.5) * 0.1)
        population.append(f_values)
    
    # Evolutionary optimization with refined operators
    best_fitness = -1.0
    best_solution = None
    
    # Enhanced evolutionary loop with time management
    for generation in range(max_generations):
        # Check time limit to prevent exceeding 60 seconds
        if time.time() - start_time > 55:  # Leave 5 seconds buffer
            break
            
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            c2 = evaluate_c2(individual)
            fitness_scores.append(c2)
            
        # Selection: keep top 50% (good balance of exploration/exploitation)
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        top_indices = sorted_indices[:population_size // 2]
        
        # Create new population through better crossover and mutation
        new_population = []
        
        # Keep best individuals
        for idx in top_indices:
            new_population.append(population[idx][:])
        
        # Generate offspring with improved operators
        while len(new_population) < population_size:
            # Tournament selection (size 3 for good selection pressure)
            parent1_idx = random.choice(top_indices)
            parent2_idx = random.choice(top_indices)
            
            # Arithmetic crossover with better blending
            child = []
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Use arithmetic crossover with better blending parameter
            # Try to be more aggressive with blending to get better combinations
            alpha = 0.5 + (random.random() - 0.5) * 0.3
            for i in range(n_steps):
                blended = alpha * parent1[i] + (1 - alpha) * parent2[i]
                # Add controlled noise for diversity
                noise = (random.random() - 0.5) * 0.08
                child.append(max(0, min(1, blended + noise)))
            
            # Mutation with adaptive rate
            mutation_rate = max(0.02, 0.15 - (generation / max_generations) * 0.12)
            
            # Apply Gaussian mutation
            for i in range(n_steps):
                if random.random() < mutation_rate:
                    # Slightly smaller variance for more precise fine-tuning
                    variance = 0.03 * (1 - generation / max_generations)
                    noise = random.gauss(0, variance)
                    child[i] = max(0, min(1, child[i] + noise))
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Track best
        current_best_idx = sorted_indices[0]
        current_best_fitness = fitness_scores[current_best_idx]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx][:]
    
    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to simple construction
        return [random.uniform(0, 1) for _ in range(n_steps)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
