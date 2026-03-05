# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List
import time

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
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
    
    # Compute ||g||₂² using the correct trapezoidal-like integration
    # As specified in prompt: for interval with heights y1, y2 and width h, 
    # contribution is (h/3)(y1² + y1y2 + y2²)
    if len(g) <= 1:
        norm_g2_sq = 0.0
    else:
        norm_g2_sq = 0.0
        # Each segment has width dx, connecting consecutive points in g
        for i in range(len(g) - 1):
            y1, y2 = g[i], g[i+1]
            norm_g2_sq += (dx/3) * (y1*y1 + y1*y2 + y2*y2)
    
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
    Enhanced function construction using a robust evolutionary approach with 
    carefully designed initializations and adaptive strategies.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Use larger resolution for better optimization
    n_steps = 2000
    
    # Strategy 1: Initialize with multiple good mathematical structures
    # Gaussian-like shape
    x = np.linspace(-0.25, 0.25, n_steps)
    sigma = 0.05
    gaussian_init = np.exp(-x**2 / (2*sigma**2))
    gaussian_init = np.maximum(gaussian_init, 0)
    
    # Double peak structure
    double_peak = np.zeros(n_steps)
    center1 = n_steps // 3
    center2 = 2 * n_steps // 3
    for i in range(n_steps):
        dist1 = abs(i - center1) / (n_steps/6)
        dist2 = abs(i - center2) / (n_steps/6)
        double_peak[i] = max(0, 1.0 - dist1**2) + max(0, 1.0 - dist2**2)
    
    # Sparse structure with high peaks
    sparse_structure = np.zeros(n_steps)
    peaks = [n_steps//4, n_steps//2, 3*n_steps//4]
    for peak_pos in peaks:
        for i in range(max(0, peak_pos-10), min(n_steps, peak_pos+10)):
            distance = abs(i - peak_pos)
            sparse_structure[i] += max(0, 1.0 - distance/10.0)
    
    # Exponential decay
    t = np.linspace(0, 1, n_steps)
    exp_decay = np.exp(-t * 3)
    
    # Uniform distribution
    uniform_dist = np.ones(n_steps)
    
    # Additional mathematical structure: sinc-like pattern
    sinc_like = np.sinc(x * 4) 
    sinc_like = np.maximum(sinc_like, 0)
    
    # Triangular pattern
    triangle = np.zeros(n_steps)
    for i in range(n_steps):
        distance_from_center = abs(i - n_steps//2) / (n_steps/2)
        triangle[i] = max(0, 1.0 - distance_from_center)
    
    # Evaluate all strategies and pick the best starting point
    strategies = [
        gaussian_init,
        double_peak,
        sparse_structure,
        exp_decay,
        uniform_dist,
        sinc_like,
        triangle
    ]
    
    best_strategy = strategies[0]
    best_c2 = compute_c2(best_strategy.tolist())
    
    for strategy in strategies[1:]:
        c2 = compute_c2(strategy.tolist())
        if c2 > best_c2:
            best_c2 = c2
            best_strategy = strategy
    
    # Now apply a robust evolutionary approach
    # Start with the best initialization
    current_solution = best_strategy.tolist()
    
    # Evolutionary parameters optimized for better performance
    population_size = 30  # Slightly larger population for better exploration
    generations = 25      # Moderate number of generations for time constraints
    mutation_rate = 0.15  # Balanced mutation rate
    
    # Initialize population with diverse strategies
    population = [current_solution.copy()]
    for _ in range(population_size - 1):
        # Create variations of the best solution
        mutated = [max(0, x + random.gauss(0, 0.1)) for x in current_solution]
        # Normalize to keep reasonable scale
        total = sum(mutated)
        if total > 0:
            mutated = [x * sum(current_solution) / total for x in mutated]
        population.append(mutated)
    
    # Evolutionary loop with early stopping
    start_time = time.time()
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
        
        # Check if we've beaten the benchmark quickly
        if best_c2 > 0.962:
            break
            
        # Create new generation
        new_population = [population[best_idx].copy()]  # Keep best
        
        # Add mutated versions of top performers
        top_indices = np.argsort(fitness_scores)[-6:]  # Top 6 for diversity
        for _ in range(population_size - 1):
            # Select parent from top performers
            parent_idx = random.choice(top_indices)
            parent = population[parent_idx]
            
            # Create child through mutation
            child = [max(0, x + random.gauss(0, 0.08)) for x in parent]
            
            # Add crossover with higher probability
            if random.random() < 0.4:  # Moderate crossover probability
                # Crossover with another top performer
                other_parent_idx = random.choice(top_indices)
                other_parent = population[other_parent_idx]
                alpha = random.random()
                child = [alpha * x1 + (1-alpha) * x2 for x1, x2 in zip(child, other_parent)]
            
            # Normalize
            total = sum(child)
            if total > 0:
                child = [x * sum(parent) / total for x in child]
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Time check to ensure we don't exceed 60 seconds
        if time.time() - start_time > 55:
            break
    
    # Final local refinement with small perturbations
    final_f = current_solution.copy()
    final_c2 = best_c2
    
    # Try several random small perturbations
    for _ in range(25):  # Moderate number for speed
        mutated_f = final_f.copy()
        
        # Randomly modify some elements
        for i in range(len(mutated_f)):
            if random.random() < 0.1:  # 10% chance to mutate each element
                # Add small random change
                change = random.gauss(0, 0.05)
                mutated_f[i] = max(0, mutated_f[i] + change)
        
        # Normalize
        if sum(mutated_f) > 0:
            mutated_f = [x * sum(final_f) / sum(mutated_f) for x in mutated_f]
        
        mutated_c2 = compute_c2(mutated_f)
        if mutated_c2 > final_c2:
            final_f = mutated_f
            final_c2 = mutated_c2
    
    return final_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
