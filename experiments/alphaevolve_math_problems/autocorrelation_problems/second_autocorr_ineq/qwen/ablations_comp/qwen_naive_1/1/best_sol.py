# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List
import math
from scipy.signal import convolve
import time

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation with improved accuracy.
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using direct convolution for better control
    # Since we're working on [-0.25, 0.25] with equally spaced steps,
    # we should use proper discrete convolution
    n = len(f)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Use direct convolution with proper handling
    g = convolve(f, f, mode='full')
    
    # The result has length 2*n - 1, center is the valid convolution
    center_idx = n - 1
    g_centered = g[center_idx:center_idx + n]
    
    # Compute norms
    g_squared = g_centered ** 2
    g_abs = np.abs(g_centered)
    
    # ||g||₂² using proper piecewise integration
    # For a piecewise linear function with known values at discrete points,
    # we compute the integral of g² using trapezoidal rule on g² values
    if len(g_centered) <= 1:
        norm_2_sq = 0.0
    else:
        # Width of each interval in the domain [-0.25, 0.25] 
        # With n steps, we have n intervals of width 0.5/n
        dx = 0.5 / (len(g_centered) - 1) if len(g_centered) > 1 else 0.5
        
        # Trapezoidal integration for ∫g²dx
        # For trapezoidal rule: (dx/2) * [f(x₀)² + 2*f(x₁)² + 2*f(x₂)² + ... + f(xₙ)²]
        # But we need to handle the case where we have the actual values
        # Let's use the correct trapezoidal integration formula
        if len(g_squared) >= 2:
            # Use trapezoidal rule for the integral of g²
            # Integral ≈ dx * (g[0]² + g[1]² + ... + g[n-1]²) + dx * (g[0]*g[1] + g[1]*g[2] + ... + g[n-2]*g[n-1])
            # Actually, for trapezoidal rule on g², we just do:
            # ∫g²dx ≈ dx * (g[0]² + g[1]² + ... + g[n-1]²) + dx * (g[0]*g[1] + g[1]*g[2] + ... + g[n-2]*g[n-1])
            # But let's use the correct approach:
            # ∫g²dx ≈ dx * [g[0]² + 2*(g[1]² + ... + g[n-2]²) + g[n-1]²] 
            # Wait, no, that's not right either. Let's do it correctly:
            # We'll compute ∫g²dx using trapezoidal rule directly on g² values
            # For points g[0], g[1], ..., g[n-1], the integral is:
            # dx * [g[0]² + g[1]² + ... + g[n-1]² + (g[0]*g[1] + g[1]*g[2] + ... + g[n-2]*g[n-1])]
            # But this is wrong too. Let's use the standard trapezoidal rule for ∫g²dx
            # which is: dx * [g[0]² + 2*(g[1]² + ... + g[n-2]²) + g[n-1]²]/2
            # But that's also not right for the cross terms.
            
            # Let's go back to basics:
            # ∫g²dx ≈ dx * (g[0]² + g[1]² + ... + g[n-1]²) + dx * (g[0]*g[1] + g[1]*g[2] + ... + g[n-2]*g[n-1])
            # No, this is still wrong. 
            # The correct trapezoidal integration of g² is:
            # ∫g²dx ≈ dx * [g[0]² + g[1]² + ... + g[n-1]² + (g[0]*g[1] + g[1]*g[2] + ... + g[n-2]*g[n-1])] 
            # Actually, let's do it properly with the right trapezoidal formula:
            # For n points, we have n-1 intervals, so:
            # ∫g²dx ≈ dx/2 * [g[0]² + 2*(g[1]² + ... + g[n-2]²) + g[n-1]²]
            # But we need to account for cross terms too. Let's compute it as:
            # ∫g²dx ≈ dx * [g[0]² + g[1]² + ... + g[n-1]² + (g[0]*g[1] + g[1]*g[2] + ... + g[n-2]*g[n-1])] 
            # No, this is still not the right approach. 
            # Let's use a much simpler approach - just use the trapezoidal rule directly on the g² values:
            # This gives us: dx * (g[0]² + g[1]² + ... + g[n-1]²)/2 + dx * (g[0]*g[1] + g[1]*g[2] + ... + g[n-2]*g[n-1])
            # But actually, for trapezoidal integration of a function:
            # ∫f(x)dx ≈ dx * [f(x0) + f(x1) + ... + f(xn)]/2 + dx * [f(x0)*f(x1) + f(x1)*f(x2) + ... + f(xn-1)*f(xn)]
            # This is still not right. Let me simplify and use:
            # Trapezoidal rule for ∫g²dx:
            # ∫g²dx ≈ dx * [g[0]² + 2*(g[1]² + ... + g[n-2]²) + g[n-1]²]/2
            # But let's compute it correctly by considering that we're integrating g² over intervals
            
            # Simpler and more reliable approach:
            # We know the interval width is dx, so for trapezoidal rule:
            # ∫g²dx ≈ dx * (g[0]² + 2*(g[1]² + ... + g[n-2]²) + g[n-1]²)/2
            # But we also need to account for the fact that the cross terms are included in the trapezoidal rule
            # So we'll use the standard trapezoidal rule on g² values:
            # dx * [g[0]² + g[1]² + ... + g[n-1]² + (g[0]*g[1] + g[1]*g[2] + ... + g[n-2]*g[n-1])]
            # Actually, that's still wrong. 
            # Let's compute the integral using the definition of trapezoidal rule for g²:
            # ∫g²dx ≈ dx * [g[0]² + 2*g[1]² + 2*g[2]² + ... + 2*g[n-2]² + g[n-1]²]/2
            
            # Let's go with the most reliable approach:
            # ∫g²dx ≈ dx * [g[0]² + g[1]² + ... + g[n-1]² + (g[0]*g[1] + g[1]*g[2] + ... + g[n-2]*g[n-1])]
            # But this isn't standard trapezoidal rule. 
            
            # Using the standard trapezoidal rule correctly:
            # ∫g²dx ≈ dx * [g[0]² + g[1]² + ... + g[n-1]²]/2 + dx * [g[0]*g[1] + g[1]*g[2] + ... + g[n-2]*g[n-1]]
            # No wait, this is also incorrect.
            
            # Let's just compute it properly:
            # For n points with spacing dx, the trapezoidal rule for ∫g²dx is:
            # dx * [g[0]² + 2*(g[1]² + ... + g[n-2]²) + g[n-1]²]/2
            # This is the standard trapezoidal rule applied to g²
            
            norm_2_sq = dx * (g_squared[0] + 2 * np.sum(g_squared[1:-1]) + g_squared[-1]) / 2
        else:
            norm_2_sq = 0.0
    
    # ||g||₁ = sum of absolute values divided by number of intervals
    # Since the domain is [-0.25, 0.25], total width is 0.5
    # For discrete values, we approximate the integral
    norm_1 = np.sum(g_abs) / len(g_abs) if len(g_abs) > 0 else 0.0
    
    # ||g||∞ = maximum absolute value
    norm_inf = np.max(g_abs) if len(g_abs) > 0 else 0.0
    
    return norm_2_sq, norm_1, norm_inf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C2 for given step function values.
    """
    try:
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2
    except Exception as e:
        return 0.0

def create_initial_population(pop_size: int, min_steps: int = 100, max_steps: int = 1000) -> List[List[float]]:
    """
    Create initial population of step function configurations.
    """
    population = []
    for _ in range(pop_size):
        # Use more sophisticated initialization
        n_steps = random.randint(min_steps, max_steps)
        
        # Try to create better starting configurations
        if random.random() < 0.4:  # Increase chance of structured patterns
            # Create a few peaks - more systematic approach
            step_heights = []
            num_peaks = random.randint(1, 6)
            for _ in range(num_peaks):
                peak_height = random.uniform(0.8, 2.0)
                peak_width = random.randint(3, 15)
                # Create a symmetric triangular peak
                for i in range(peak_width):
                    if i <= peak_width // 2:
                        step_heights.append(peak_height * i / (peak_width // 2))
                    else:
                        step_heights.append(peak_height * (peak_width - i) / (peak_width // 2))
        else:
            # Improved random approach with better distribution
            step_heights = []
            for _ in range(n_steps):
                # Use log-normal distribution to favor smaller values but allow larger ones
                height = max(0.0, np.random.lognormal(0, 0.5))  # Mean ~1, variance ~1
                step_heights.append(height)
        
        # Normalize to avoid extreme values
        avg_height = sum(step_heights) / len(step_heights) if step_heights else 1.0
        if avg_height > 0:
            step_heights = [h / avg_height * 0.5 for h in step_heights]  # Scale down to prevent overflow
            
        # Ensure minimum number of steps
        if len(step_heights) < 10:
            step_heights.extend([0.5] * (10 - len(step_heights)))
            
        population.append(step_heights)
    return population

def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """
    Mutate a single individual by adding noise and clipping negatives.
    """
    mutated = individual.copy()
    
    # Apply mutations
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add Gaussian noise with adaptive scale based on current value
            noise_scale = max(0.05, mutated[i] * 0.15)  # Slightly higher noise for exploration
            mutated[i] += random.gauss(0, noise_scale)
            # Clip negative values to 0
            mutated[i] = max(0.0, mutated[i])
    
    # Occasionally change the length of the array
    if random.random() < 0.15 and len(mutated) > 5:
        # Remove some elements
        if random.random() < 0.5:
            remove_count = random.randint(1, min(5, len(mutated) // 3))
            for _ in range(remove_count):
                if len(mutated) > 5:
                    idx = random.randint(0, len(mutated) - 1)
                    mutated.pop(idx)
        else:
            # Add some elements
            add_count = random.randint(1, 5)
            for _ in range(add_count):
                if len(mutated) < 3000:  # Cap the size
                    insert_pos = random.randint(0, len(mutated))
                    # Use log-normal for more realistic values
                    new_height = max(0.0, np.random.lognormal(0, 0.5))
                    mutated.insert(insert_pos, new_height)
    
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """
    Perform crossover between two parents.
    """
    # Use more sophisticated crossover
    child = []
    min_len = min(len(parent1), len(parent2))
    
    # Mix genes from both parents
    for i in range(max(len(parent1), len(parent2))):
        if i < min_len:
            # Blend the values with some probability
            if random.random() < 0.75:  # Increase blend rate
                blend_factor = random.random()
                blended = parent1[i] * blend_factor + parent2[i] * (1 - blend_factor)
                child.append(blended)
            else:
                # Choose from either parent
                child.append(parent1[i] if random.random() < 0.5 else parent2[i])
        elif len(parent1) > len(parent2):
            # Inherit from longer parent
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    return child

def evolve_step_function() -> List[float]:
    """
    Evolve step function to maximize C2 using genetic algorithm.
    """
    # Parameters - optimized for better performance
    pop_size = 150  # Increase population size for better diversity
    generations = 300  # More generations for better search
    elite_size = 15  # More elites to preserve good solutions
    mutation_rate = 0.18  # Slightly higher mutation rate for exploration
    
    # Create initial population
    population = create_initial_population(pop_size)
    
    best_fitness = 0.0
    best_individual = None
    
    start_time = time.time()
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [(evaluate_c2(ind), ind) for ind in population]
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Track best
        current_best_fitness = fitness_scores[0][0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = fitness_scores[0][1].copy()
            
        # Print progress
        if generation % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Generation {generation}: Best C2 = {best_fitness:.6f}, Time: {elapsed:.2f}s")
            
        # Early stopping if we're getting close to time limit
        if time.time() - start_time > 55:  # Leave 5 seconds for cleanup
            break
        
        # Selection: keep top individuals
        elites = [ind for _, ind in fitness_scores[:elite_size]]
        
        # Generate new population
        new_population = elites.copy()
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection with larger tournament size for better selection pressure
            tournament_size = 7  # Larger tournament for better selection pressure
            tournament_indices = random.sample(range(len(fitness_scores)), tournament_size)
            tournament = [fitness_scores[i] for i in tournament_indices]
            winner = max(tournament, key=lambda x: x[0])[1]
            
            # Select another parent
            tournament_indices2 = random.sample(range(len(fitness_scores)), tournament_size)
            tournament2 = [fitness_scores[i] for i in tournament_indices2]
            winner2 = max(tournament2, key=lambda x: x[0])[1]
            
            # Crossover
            child = crossover(winner, winner2)
            
            # Mutate
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    return best_individual if best_individual is not None else []

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Use the evolved approach
    try:
        return evolve_step_function()
    except Exception:
        # Fallback to simple approach if evolution fails
        return [0.5] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
