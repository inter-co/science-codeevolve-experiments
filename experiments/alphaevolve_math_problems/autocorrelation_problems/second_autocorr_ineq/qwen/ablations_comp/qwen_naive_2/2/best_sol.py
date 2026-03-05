# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from typing import List
import random
import time
from numba import jit
import optuna
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
import warnings

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: np.ndarray) -> tuple:
    """
    Fast computation of autoconvolution norms using Numba JIT compilation.
    Uses piecewise linear integration method as specified in problem description.
    f_values: step heights on equally spaced intervals
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    if len(f_values) == 0:
        return 0.0, 0.0, 0.0
    
    n = len(f_values)
    dx = 0.5 / n  # spacing on [-0.25, 0.25]
    
    # Compute autoconvolution g = f * f using discrete convolution
    # This gives us a function on [-0.5, 0.5] with 2*n-1 points
    g = np.zeros(2 * n - 1)
    
    # Manual convolution for better control - optimized version
    for i in range(n):
        for j in range(n):
            idx = i + j
            g[idx] += f_values[i] * f_values[j]
    
    # Apply proper scaling for integration - the key insight is that 
    # we're doing discrete integration with dx spacing
    g_scaled = g * dx
    
    # Compute norms using the piecewise linear integration method as described
    # For ||g||₂², use trapezoidal-like piecewise integration:
    # For consecutive points with heights y1, y2 and width dx, contribution is (dx/3)(y1² + y1*y2 + y2²)
    g_squared_sum = 0.0
    for i in range(len(g_scaled) - 1):
        y1 = g_scaled[i]
        y2 = g_scaled[i + 1]
        g_squared_sum += (dx / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)
    
    # ||g||₁ is approximated as sum(|g|) / (len(g) + 1) as per specification
    g_abs_sum = np.sum(np.abs(g_scaled))
    g_abs_sum = g_abs_sum / (len(g_scaled) + 1) if len(g_scaled) + 1 > 0 else 0.0
    
    # ||g||∞ is max(|g|)
    g_max = np.max(np.abs(g_scaled))
    
    return g_squared_sum, g_abs_sum, g_max

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    f_values: step heights on equally spaced intervals
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    if len(f_values) == 0:
        return 0.0, 0.0, 0.0
    
    f_array = np.array(f_values)
    return compute_autoconvolution_norms_fast(f_array)

def evaluate_c2(f_values: List[float]) -> float:
    """Evaluate C2 for given step function values."""
    try:
        g_squared_sum, g_abs_sum, g_max = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero with a more robust check
        if g_abs_sum <= 1e-12 or g_max <= 1e-12:
            return 0.0
            
        c2 = g_squared_sum / (g_abs_sum * g_max)
        return c2
    except Exception as e:
        # Log error for debugging but return 0.0 to avoid crashes
        return 0.0

def create_initial_population(pop_size: int, individual_size: int) -> List[List[float]]:
    """Create initial population of step function configurations with better initialization."""
    population = []
    
    # Strategy 1: Mix of different initialization approaches for better diversity
    for i in range(pop_size):
        individual = []
        if i < pop_size // 4:  # Flat distributions
            # Flat with some variation
            base_val = np.random.uniform(0.1, 0.8)
            for _ in range(individual_size):
                val = max(0.0, base_val + np.random.normal(0, 0.1))
                individual.append(val)
        elif i < pop_size // 2:  # Peaky distributions  
            # Peaks at center with decay - more structured approach
            for j in range(individual_size):
                # Create a bell-shaped pattern with sharper peak
                center = individual_size // 2
                distance = abs(j - center)
                # Gaussian-like decay from center with sharper peak
                val = max(0.0, 1.5 * np.exp(-distance * distance / (individual_size / 3)))
                # Add some noise
                val = max(0.0, val + np.random.normal(0, 0.05))
                individual.append(val)
        elif i < 3 * pop_size // 4:  # Multi-peak patterns
            # Multiple peaks for variety
            for j in range(individual_size):
                # Create multi-peak pattern
                pos1 = individual_size // 3
                pos2 = 2 * individual_size // 3
                val1 = 1.0 * np.exp(-((j - pos1)**2) / (individual_size / 4))
                val2 = 0.8 * np.exp(-((j - pos2)**2) / (individual_size / 4))
                val = max(0.0, val1 + val2)
                val = max(0.0, val + np.random.normal(0, 0.05))
                individual.append(val)
        else:  # Random initialization with higher variance
            # Sample from a distribution that favors values around 0.5
            for _ in range(individual_size):
                # Use log-normal distribution for positive values with good spread
                val = max(0.0, np.random.lognormal(0, 0.5))  
                individual.append(val)
                
        population.append(individual)
    return population

def mutate_individual(individual: List[float], mutation_rate: float = 0.1, generation: int = 0) -> List[float]:
    """Enhanced mutation with adaptive parameters and better strategies."""
    mutated = individual.copy()
    
    # Adaptive mutation rate that decreases over generations
    adaptive_rate = mutation_rate * (1.0 - generation / 1000.0)
    adaptive_rate = max(0.01, adaptive_rate)
    
    for i in range(len(mutated)):
        if random.random() < adaptive_rate:
            current_val = mutated[i]
            # Different mutation strategies based on value magnitude
            if current_val < 0.1:  # Very small values
                # Larger jumps for small values
                mutated[i] = max(0.0, mutated[i] + np.random.normal(0, 0.2))
            elif current_val > 0.8:  # Large values
                # Smaller jumps for large values to prevent overshoot
                mutated[i] = max(0.0, mutated[i] + np.random.normal(0, 0.05))
            else:  # Medium values
                # Standard mutation with wider range for exploration
                mutated[i] = max(0.0, mutated[i] + np.random.normal(0, 0.15))
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Improved crossover with blending and better handling of length differences."""
    if len(parent1) != len(parent2):
        # If lengths differ, pad shorter one with average values
        min_len = min(len(parent1), len(parent2))
        max_len = max(len(parent1), len(parent2))
        parent1 = parent1[:min_len] + [np.mean(parent1)] * (max_len - min_len)
        parent2 = parent2[:min_len] + [np.mean(parent2)] * (max_len - min_len)
        
    # Blend crossover with better mixing
    point = random.randint(1, len(parent1) - 1)
    child = []
    
    # Take first part from parent1, second part from parent2
    child.extend(parent1[:point])
    child.extend(parent2[point:])
    
    # Add some blending to smooth transitions
    for i in range(len(child)):
        if random.random() < 0.03:  # Reduced blending probability
            if i > 0 and i < len(child) - 1:
                # Average with neighbors for smoothing - but more conservative
                avg_val = (child[i-1] + child[i] + child[i+1]) / 3.0
                child[i] = avg_val
                
    return child

def select_parents(population: List[List[float]], fitnesses: List[float], 
                   tournament_size: int = 3) -> List[List[float]]:
    """Enhanced selection with better diversity maintenance."""
    selected = []
    
    # Sort by fitness to get top performers
    sorted_indices = np.argsort(fitnesses)[::-1]
    
    # Elitism: keep top 40% (more than before)
    elite_count = max(1, len(population) // 2)
    elite_indices = sorted_indices[:elite_count]
    for idx in elite_indices:
        selected.append(population[idx].copy())
    
    # Tournament selection for remaining slots with diversity consideration
    remaining_slots = len(population) - len(elite_indices)
    
    # Use fitness proportionate selection for some diversity
    if len(population) > 10:
        # Calculate selection probabilities based on fitness
        fitness_probs = np.array(fitnesses) - np.min(fitnesses) + 1e-10  # Ensure positive
        fitness_probs = fitness_probs / np.sum(fitness_probs)
        
        for _ in range(remaining_slots):
            # With some probability, do tournament selection
            if random.random() < 0.6:
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
                selected.append(population[winner_idx].copy())
            else:
                # Otherwise, sample with probability proportional to fitness
                selected_idx = np.random.choice(len(population), p=fitness_probs)
                selected.append(population[selected_idx].copy())
    else:
        # For smaller populations, stick with pure tournament selection
        for _ in range(remaining_slots):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx].copy())
    
    return selected

def optimize_step_function_direct(max_time: float = 60.0) -> tuple:
    """
    Direct optimization approach using gradient-free optimization with enhanced strategy.
    """
    start_time = time.time()
    
    # Better parameter ranges for optimization
    individual_size = 500  # Fixed size for efficiency
    pop_size = 150         # Population size
    mutation_rate = 0.15   # Mutation rate
    num_generations = 2000 # Generations
    
    # Start with a good baseline
    best_individual = [1.0] * individual_size
    best_c2 = evaluate_c2(best_individual)
    
    # Enhanced evolutionary approach
    population = create_initial_population(pop_size, individual_size)
    
    for generation in range(num_generations):
        if time.time() - start_time > max_time:
            break
            
        # Evaluate fitness
        fitnesses = [evaluate_c2(ind) for ind in population]
        
        # Track best
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_c2:
            best_c2 = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
            
        # Print progress every 200 generations
        if generation % 200 == 0:
            print(f"Generation {generation}: Best C2 = {best_c2:.6f}")
        
        # Create next generation
        # Keep elite (top 30%)
        elite_indices = np.argsort(fitnesses)[-max(1, pop_size // 3):]
        next_generation = [population[i].copy() for i in elite_indices]
        
        # Select parents
        selected_parents = select_parents(population, fitnesses)
        
        # Generate offspring through crossover and mutation
        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover(parent1, parent2)
            child = mutate_individual(child, mutation_rate, generation)
            next_generation.append(child)
            
        population = next_generation[:pop_size]
    
    return best_individual, best_c2

def optimize_with_local_search(max_time: float = 60.0) -> tuple:
    """
    Hybrid approach combining global search with local refinement.
    """
    start_time = time.time()
    
    # First, do a coarse global search
    print("Starting coarse global search...")
    coarse_individual, coarse_c2 = optimize_step_function_direct(max_time * 0.7)
    
    print(f"Coarse search result: C2 = {coarse_c2:.6f}")
    
    # Then refine with local search around the best solution
    print("Starting local refinement...")
    
    # Create a slightly perturbed version of the best solution
    refined_individual = coarse_individual.copy()
    
    # Local search: try small variations around the best solution
    best_refined_c2 = coarse_c2
    best_refined_individual = refined_individual.copy()
    
    # Try small perturbations to improve
    for i in range(1000):  # Limited number of local searches
        if time.time() - start_time > max_time:
            break
            
        # Make a small change to one element
        idx = random.randint(0, len(refined_individual) - 1)
        old_val = refined_individual[idx]
        
        # Small random perturbation
        new_val = max(0.0, old_val + np.random.normal(0, 0.05))
        refined_individual[idx] = new_val
        
        # Evaluate
        c2 = evaluate_c2(refined_individual)
        
        if c2 > best_refined_c2:
            best_refined_c2 = c2
            best_refined_individual = refined_individual.copy()
        else:
            # Revert if worse
            refined_individual[idx] = old_val
    
    return best_refined_individual, best_refined_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using evolutionary optimization."""
    try:
        # Run hybrid optimization for up to 60 seconds
        best_individual, best_c2 = optimize_with_local_search(60.0)
        print(f"Optimization complete. Best C2 found: {best_c2:.6f}")
        return best_individual if best_individual is not None else [0.0] * 100
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Return default function
        return [0.5] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
