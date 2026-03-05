# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List
import time

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation using proper numerical integration.
    f_values: step heights of the function on [-1/4, 1/4]
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    # Convert to numpy array
    f = np.array(f_values)
    n = len(f)
    
    if n == 0:
        return 0.0, 1.0, 0.0
    
    # Step size for the interval [-0.25, 0.25]
    dx = 0.5 / n
    
    # Compute autoconvolution g = f * f using scipy's convolution
    g = signal.convolve(f, f, mode='full')
    
    # Compute ||g||₂² using proper trapezoidal integration
    if len(g) <= 1:
        norm_g2_sq = 0.0
    else:
        # Use Simpson's rule for better accuracy: integrate g^2
        norm_g2_sq = 0.0
        # For simpson's rule, we need an odd number of intervals
        # We'll integrate using trapezoidal-like approach for simplicity but more accurate
        for i in range(len(g) - 1):
            # Trapezoidal rule for g^2
            y1, y2 = g[i]**2, g[i+1]**2
            norm_g2_sq += (dx/2) * (y1 + y2)
    
    # ||g||₁ = sum of absolute values times dx (trapezoidal rule)
    norm_g1 = np.sum(np.abs(g)) * dx
    
    # ||g||∞ = max absolute value
    norm_ginf = np.max(np.abs(g))
    
    return norm_g2_sq, norm_g1, norm_ginf

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 = ||g||₂² / (||g||₁ · ||g||∞)"""
    norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_g1 <= 1e-12 or norm_ginf <= 1e-12:
        return 0.0
    
    c2 = norm_g2_sq / (norm_g1 * norm_ginf)
    return c2

def construct_function() -> List[float]:
    """
    Enhanced function construction using a hybrid approach combining:
    1. Multiple mathematical initialization strategies (like INSPIRATION 1)
    2. Sophisticated evolutionary optimization (like INSPIRATION 1)
    3. Proper numerical integration (like INSPIRATION 2)
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Use higher resolution for better optimization (as in successful inspirations)
    n_steps = 2000  # Much higher resolution for better optimization
    
    # Strategy 1: Initialize with multiple sophisticated mathematical structures
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # 1. Multi-scale Gaussian mixture (inspired by INSPIRATION 1)
    gaussian_mixture = np.zeros(n_steps)
    scales = [0.02, 0.04, 0.08]
    weights = [0.4, 0.4, 0.2]
    for i, (scale, weight) in enumerate(zip(scales, weights)):
        gaussian_mixture += weight * np.exp(-x**2 / (2*scale**2))
    
    # 2. Sinusoidal modulation on Gaussian (inspired by INSPIRATION 1)
    sinusoidal = np.exp(-x**2 / (2*0.05**2)) * (1 + 0.3 * np.sin(8 * np.pi * x))
    
    # 3. Peak with plateau (inspired by INSPIRATION 1)
    plateau = np.zeros(n_steps)
    center = n_steps // 2
    for i in range(n_steps):
        distance = abs(i - center) / (n_steps/6)
        plateau[i] = max(0, 1.0 - distance**2) * (1.0 - 0.5 * (distance > 0.2))
    
    # 4. Smooth bell curve (inspired by INSPIRATION 2)
    bell_curve = np.exp(-x**2 / (2*0.03**2))
    
    # 5. Piecewise linear with multiple segments (inspired by INSPIRATION 2)
    piecewise = np.zeros(n_steps)
    segments = [n_steps//4, n_steps//2, 3*n_steps//4]
    for i in range(n_steps):
        distances = [abs(i - seg) for seg in segments]
        min_dist = min(distances)
        piecewise[i] = max(0, 1.0 - min_dist/(n_steps/8))
    
    # 6. Double peak structure (inspired by INSPIRATION 2)
    double_peak = np.zeros(n_steps)
    center1 = n_steps // 3
    center2 = 2 * n_steps // 3
    for i in range(n_steps):
        dist1 = abs(i - center1) / (n_steps/6)
        dist2 = abs(i - center2) / (n_steps/6)
        double_peak[i] = max(0, 1.0 - dist1**2) + max(0, 1.0 - dist2**2)
    
    # 7. Exponential decay (inspired by INSPIRATION 2)
    t = np.linspace(0, 1, n_steps)
    exp_decay = np.exp(-t * 3)
    
    # 8. Uniform distribution (inspired by INSPIRATION 2)
    uniform_dist = np.ones(n_steps)
    
    # Evaluate all strategies and pick the best starting point
    strategies = [
        gaussian_mixture,
        sinusoidal,
        plateau,
        bell_curve,
        piecewise,
        double_peak,
        exp_decay,
        uniform_dist
    ]
    
    best_strategy = strategies[0]
    best_c2 = compute_c2(best_strategy.tolist())
    
    for strategy in strategies[1:]:
        c2 = compute_c2(strategy.tolist())
        if c2 > best_c2:
            best_c2 = c2
            best_strategy = strategy
    
    # Now apply a more aggressive evolutionary approach
    current_solution = best_strategy.tolist()
    
    # Evolutionary parameters tuned for better performance
    population_size = 50  # Larger population
    generations = 30      # More generations
    elite_size = 5        # Keep top individuals
    
    # Initialize population with diverse strategies
    population = [current_solution.copy()]
    for _ in range(population_size - 1):
        # Create variations with different intensities (inspired by INSPIRATION 1)
        variation_type = random.randint(0, 3)
        mutated = current_solution.copy()
        
        if variation_type == 0:  # Small Gaussian noise
            mutated = [max(0, x + random.gauss(0, 0.03)) for x in mutated]
        elif variation_type == 1:  # Medium noise
            mutated = [max(0, x + random.gauss(0, 0.08)) for x in mutated]
        elif variation_type == 2:  # Large noise
            mutated = [max(0, x + random.gauss(0, 0.15)) for x in mutated]
        else:  # Combination of mutations
            mutated = [max(0, x + random.gauss(0, 0.05) + 0.02 * random.random()) for x in mutated]
            
        # Normalize to keep reasonable scale
        total = sum(mutated)
        if total > 0:
            mutated = [x * sum(current_solution) / total for x in mutated]
        population.append(mutated)
    
    # Evolutionary loop with more sophisticated selection (inspired by INSPIRATION 1)
    for gen in range(generations):
        # Evaluate population
        fitness_scores = []
        for individual in population:
            try:
                c2 = compute_c2(individual)
                fitness_scores.append(c2)
            except:
                fitness_scores.append(0.0)
        
        # Find best individual
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_c2:
            best_c2 = fitness_scores[best_idx]
            current_solution = population[best_idx].copy()
        
        # Early stopping if we've reached the benchmark
        if best_c2 > 0.962:
            break
            
        # Create new generation using tournament selection (inspired by INSPIRATION 1)
        new_population = []
        
        # Keep elite individuals
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 5
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            parent = population[winner_idx]
            
            # Mutation with adaptive rate (inspired by INSPIRATION 1)
            child = [max(0, x + random.gauss(0, 0.05 * (1 + 0.1 * gen/generations))) for x in parent]
            
            # Add crossover with some probability
            if random.random() < 0.4:
                # Select another parent
                parent2_idx = random.choice(elite_indices)
                parent2 = population[parent2_idx]
                # Blend crossover (inspired by INSPIRATION 1)
                alpha = random.random()
                child = [alpha * x1 + (1-alpha) * x2 for x1, x2 in zip(child, parent2)]
            
            # Normalize
            total = sum(child)
            if total > 0:
                child = [x * sum(parent) / total for x in child]
            
            new_population.append(child)
        
        population = new_population[:population_size]
    
    return current_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
