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
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import copy
from scipy.optimize import differential_evolution, minimize
import itertools
from scipy.signal import convolve

@jit(nopython=True)
def compute_piecewise_integral(g_values: np.ndarray, dx: float) -> float:
    """Compute piecewise integral using trapezoidal-like formula"""
    if len(g_values) < 2:
        return g_values[0]**2 if len(g_values) > 0 else 0.0
    
    integral = 0.0
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
    g_full = convolve(f, f, mode='full')
    
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
    
    # Focus on better mathematical patterns that should produce high C2 values
    strategies = [
        # Sharp central peak (good for creating high peaks in convolution)
        lambda n: [max(0, 1.0 - 4*(abs(i - n//2)/(n//2))**2) for i in range(n)],
        
        # Multi-peak with specific spacing that creates interference patterns
        lambda n: [max(0, 1 - 4*((i - n//3)/(n//3))**2) + max(0, 1 - 4*((i - 2*n//3)/(n//3))**2) for i in range(n)],
        
        # Gaussian-like with optimized width for convolution
        lambda n: [np.exp(-0.5*((i - n//2)/(n//6))**2) for i in range(n)],
        
        # Optimized bell curve with better balance
        lambda n: [np.exp(-0.2*((i - n//2)/(n//4))**2) for i in range(n)],
        
        # Two symmetric peaks with controlled shape
        lambda n: [0.5 + 0.3*np.sin(4*np.pi*i/n) for i in range(n)],
        
        # Square wave pattern with sharp transitions
        lambda n: [1.0 if i < n//2 else 0.0 for i in range(n)],
        
        # Modified square wave with gradual transition
        lambda n: [0.5 + 0.5*np.tanh(4*(i - n//2)/n) for i in range(n)],
        
        # Asymmetric pattern that might produce beneficial convolution
        lambda n: [max(0, 1.0 - 2*(abs(i - n//4)/(n//4))**1.5) for i in range(n)],
        
        # High-frequency oscillation pattern
        lambda n: [0.5 + 0.3*np.sin(12*np.pi*i/n) + 0.2*np.cos(18*np.pi*i/n) for i in range(n)],
        
        # Optimized smooth pattern with peak concentration
        lambda n: [max(0, 1 - 0.5*((i - n//2)/(n//3))**2) for i in range(n)],
        
        # Concentrated peak pattern - optimized for better C2
        lambda n: [0.8 if abs(i - n//2) < n//8 else 0.2 for i in range(n)],
        
        # Double peak with specific spacing
        lambda n: [0.6 * (1 + np.cos(4*np.pi*i/n)) if i < n//2 else 0.6 * (1 + np.cos(4*np.pi*(i-n//2)/n)) for i in range(n)],
        
        # Sine wave with optimized amplitude and phase
        lambda n: [0.5 + 0.3*np.sin(6*np.pi*i/n + np.pi/4) for i in range(n)],
    ]
    
    # Create diverse initial population
    for i in range(size):
        n = random.randint(min_steps, max_steps)
        
        # Select strategy based on index
        strategy_idx = i % len(strategies)
        if strategy_idx < len(strategies):
            try:
                f_vals = strategies[strategy_idx](n)
                # Add some noise to make solutions unique
                for j in range(len(f_vals)):
                    if random.random() < 0.15:
                        noise = random.gauss(0, 0.03)
                        f_vals[j] = max(0, min(1.0, f_vals[j] + noise))
                population.append(f_vals)
            except Exception:
                # Fallback to uniform distribution
                population.append([0.5] * n)
        else:
            # Fallback to uniform distribution
            population.append([0.5] * n)
    
    return population

def mutate_individual(f_values: List[float], mutation_rate: float = 0.1, generation: int = 0) -> List[float]:
    """Mutate a single individual with enhanced parameters and strategies."""
    new_values = f_values.copy()
    # Adaptive mutation rate that decreases over time
    adaptive_mutation_rate = mutation_rate * (1.0 - generation / 500.0)
    adaptive_mutation_rate = max(0.005, adaptive_mutation_rate)
    
    for i in range(len(new_values)):
        if random.random() < adaptive_mutation_rate:
            # Apply mutation with enhanced strategies
            current_val = new_values[i]
            if random.random() < 0.7:  # 70% chance of small mutation
                # Add small Gaussian noise with adaptive scale
                noise_scale = 0.01 + 0.02 * (1.0 - current_val)  # Less noise near 1.0
                noise = random.gauss(0, noise_scale)
                new_values[i] += noise
                # Ensure non-negative
                new_values[i] = max(0, new_values[i])
            else:
                # More significant mutation for exploration
                if random.random() < 0.5:
                    # Large perturbation
                    delta = random.uniform(-0.1, 0.1)
                    new_values[i] = max(0, min(1.0, new_values[i] + delta))
                else:
                    # Random replacement with better distribution
                    if current_val > 0.7:
                        new_values[i] = random.uniform(0.8, 1.0)
                    elif current_val < 0.3:
                        new_values[i] = random.uniform(0, 0.3)
                    else:
                        new_values[i] = random.uniform(0, 1.0)
    return new_values

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform crossover with enhanced blending strategies."""
    # Use uniform crossover with more sophisticated blending
    size = min(len(parent1), len(parent2))
    child = []
    
    # Blend strategy: mix both parents' values with probability
    for i in range(size):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
        
        # Occasionally blend with a weighted average
        if random.random() < 0.15:  # Reduced frequency but more effective
            alpha = random.random()
            # Use a more sophisticated blending that considers neighbors
            if i > 0 and i < size - 1:
                # Blend based on local context
                neighbor_avg = (parent1[i-1] + parent1[i+1] + parent2[i-1] + parent2[i+1]) / 4
                child[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
                # Bias towards local average for smoother transitions
                child[i] = 0.4 * child[i] + 0.6 * neighbor_avg
            else:
                child[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
    
    # Extend to match longer parent if needed
    if len(parent1) > size:
        child.extend(parent1[size:])
    elif len(parent2) > size:
        child.extend(parent2[size:])
    
    return child

def optimize_step_function() -> List[float]:
    """Optimize step function to maximize C2 using improved evolutionary approach."""
    # Parameters - tuned for better performance and faster convergence
    population_size = 300  # Reduced population size for faster convergence
    generations = 400      # Reduced generations to meet time constraint
    elite_size = 30        # Reduced elite size to focus on exploration
    mutation_rate = 0.15   # Slightly higher mutation rate for exploration
    
    # Initialize population
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
        if abs(best_c2 - prev_best) < 1e-6:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best_c2
        
        # Print progress every 50 generations
        if generation % 50 == 0:
            print(f"Generation {generation}: Best C2 = {best_c2:.6f}")
        
        # Early stopping if no improvement for 50 generations
        if stagnation_count > 50:
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
            parent1 = tournament_selection(sorted_population, sorted_fitness, 10)
            parent2 = tournament_selection(sorted_population, sorted_fitness, 10)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate_individual(child, mutation_rate, generation)
            
            new_population.append(child)
        
        population = new_population[:population_size]
    
    return best_solution if best_solution is not None else []

def tournament_selection(population: List[List[float]], fitness_scores: List[float], k: int) -> List[float]:
    """Select individual using tournament selection with enhanced pressure."""
    # Use a larger tournament size for better selection pressure
    tournament_indices = random.sample(range(len(population)), min(k, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index].copy()

def local_refinement(f_values: List[float], iterations: int = 50) -> List[float]:
    """Perform enhanced local refinement of a solution."""
    current_solution = f_values.copy()
    current_c2 = compute_c2(current_solution)
    
    # Try multiple refinement approaches
    for _ in range(iterations):
        # Strategy 1: Random small perturbations with enhanced control
        candidate = current_solution.copy()
        for i in range(len(candidate)):
            if random.random() < 0.2:  # Increased chance to modify (more aggressive)
                # Add small random change with better scaling
                delta = random.uniform(-0.08, 0.08)
                candidate[i] = max(0, candidate[i] + delta)
        
        new_c2 = compute_c2(candidate)
        if new_c2 > current_c2:
            current_solution = candidate
            current_c2 = new_c2
            continue
            
        # Strategy 2: Gradient-based perturbation with better smoothing
        candidate = current_solution.copy()
        for i in range(len(candidate)):
            if random.random() < 0.15:  # Increased chance for directional modification
                # Look at neighbors for gradient approximation
                neighbor_sum = 0
                count = 0
                if i > 0:
                    neighbor_sum += candidate[i-1]
                    count += 1
                if i < len(candidate) - 1:
                    neighbor_sum += candidate[i+1]
                    count += 1
                
                if count > 0:
                    avg_neighbor = neighbor_sum / count
                    # Increase if below average, decrease if above
                    if candidate[i] < avg_neighbor:
                        candidate[i] = min(1.0, candidate[i] + 0.03)
                    else:
                        candidate[i] = max(0, candidate[i] - 0.03)
        
        new_c2 = compute_c2(candidate)
        if new_c2 > current_c2:
            current_solution = candidate
            current_c2 = new_c2
    
    return current_solution

def advanced_local_search(initial_solution: List[float], max_iterations: int = 100) -> List[float]:
    """Enhanced local search with more aggressive improvement strategies."""
    current_solution = initial_solution.copy()
    current_c2 = compute_c2(current_solution)
    
    # Try different strategies for local improvement
    for iteration in range(max_iterations):
        # Strategy 1: Random sampling around current solution with better diversity
        candidate = current_solution.copy()
        # Perturb a few elements
        num_perturbations = max(3, len(candidate) // 10)  # Perturb about 10% of elements
        for _ in range(num_perturbations):
            i = random.randint(0, len(candidate) - 1)
            # Add a small random change
            delta = random.uniform(-0.08, 0.08)
            candidate[i] = max(0, min(1.0, candidate[i] + delta))
        
        new_c2 = compute_c2(candidate)
        if new_c2 > current_c2:
            current_solution = candidate
            current_c2 = new_c2
            continue
            
        # Strategy 2: Simulated annealing-inspired approach with better temperature schedule
        if random.random() < 0.25:  # Increased chance of trying simulated annealing
            candidate = current_solution.copy()
            # Make a larger perturbation
            i = random.randint(0, len(candidate) - 1)
            delta = random.uniform(-0.12, 0.12)
            candidate[i] = max(0, min(1.0, candidate[i] + delta))
            
            new_c2 = compute_c2(candidate)
            # Accept worse solutions with some probability based on temperature
            if new_c2 > current_c2 or random.random() < 0.08:  # Higher acceptance probability
                current_solution = candidate
                current_c2 = new_c2
    
    return current_solution

def direct_optimization_approach() -> List[float]:
    """Try direct optimization approach with improved strategies."""
    # Since we're dealing with discrete step functions, let's try a hybrid approach
    # First get a good starting point from evolution
    print("Starting evolutionary optimization...")
    evol_result = optimize_step_function()
    
    if evol_result is None or len(evol_result) < 50:
        # If evolution failed, use a simple pattern
        return [0.5] * 1000
    
    # Then refine using advanced local search around the best solution
    print("Refining with advanced local search...")
    
    # Try different patterns around the best solution
    best_c2 = compute_c2(evol_result)
    best_solution = evol_result.copy()
    
    # Advanced local refinement
    refined_solution = advanced_local_search(best_solution, 50)
    refined_c2 = compute_c2(refined_solution)
    
    if refined_c2 > best_c2:
        best_c2 = refined_c2
        best_solution = refined_solution.copy()
    
    # Additional refinement attempts with more aggressive strategies
    for attempt in range(15):
        # Small perturbations around the best solution
        refined = best_solution.copy()
        for i in range(len(refined)):
            if random.random() < 0.08:  # 8% chance to modify each element
                # Add small random change with larger range
                delta = random.uniform(-0.05, 0.05)
                refined[i] = max(0, refined[i] + delta)
        
        new_c2 = compute_c2(refined)
        if new_c2 > best_c2:
            best_c2 = new_c2
            best_solution = refined.copy()
    
    # Final intensive local search
    final_solution = advanced_local_search(best_solution, 80)
    final_c2 = compute_c2(final_solution)
    
    if final_c2 > best_c2:
        best_c2 = final_c2
        best_solution = final_solution.copy()
    
    return best_solution

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run optimization
    start_time = time.time()
    result = direct_optimization_approach()
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
